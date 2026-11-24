"""Supervise a bounded Go2 session on an existing AutoDL instance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "rear_run.py"
SHUTDOWN = Path("/usr/bin/shutdown")
TRASH = Path("/root/.local/share/Trash")
KIT = "--kit_args=--/plugins/carb.tasking.plugin/threadCount=8"


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def read_status(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def resources():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8, check=False,
    )
    info = {"time": timestamp(), "gpu_temp_util_mib": result.stdout.strip()}
    for name in ("cpu.pressure", "memory.pressure", "memory.current"):
        file = Path("/sys/fs/cgroup") / name
        if file.is_file():
            info[name] = file.read_text().splitlines()[0]
    return info


def training(args, root, state):
    checkpoint = args.checkpoint
    elapsed = args.elapsed
    deadline = time.monotonic() + args.seconds - elapsed + 240
    for attempt in range(1, 4):
        output = root / f"train-{attempt}"
        argv = [
            sys.executable, "-u", str(TRAINER), "--headless", KIT,
            "--output", str(output), "--duration", str(args.seconds), "--elapsed", str(elapsed),
            "--num_envs", str(args.num_envs),
        ]
        if checkpoint:
            argv.extend(("--checkpoint", str(checkpoint)))
        if args.motion_practice:
            argv.append("--motion-practice")
        if args.route_practice:
            argv.append("--route-practice")
        if args.swing_practice:
            argv.append("--swing-practice")
        state.update(phase="training", attempt=attempt, active_training_dir=output.name)
        record(root / "supervisor.json", state)
        with (root / f"train-{attempt}.log").open("w") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    with (root / "resources.jsonl").open("a") as output_log:
                        print(json.dumps(resources()), file=output_log)
                except (OSError, subprocess.SubprocessError) as error:
                    print(f"{timestamp()} Resource probe failed: {error}", flush=True)
                try:
                    process.wait(timeout=min(45, max(1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        status = read_status(output / "status.json")
        state["attempts"].append({"number": attempt, "return_code": process.returncode, **status})
        if (output / "latest.pt").is_file():
            elapsed = status.get("elapsed_training_seconds", elapsed)
            checkpoint = output / "latest.pt"
        if (output / "final.pt").is_file():
            checkpoint = output / "final.pt"
        state.update(elapsed_training_seconds=elapsed, checkpoint=str(checkpoint) if checkpoint else None)
        record(root / "supervisor.json", state)
        if elapsed >= args.seconds or time.monotonic() >= deadline:
            break
        if process.returncode == 0 or attempt == 3:
            break
        time.sleep(12)
    return checkpoint if elapsed > 0 else None


def evaluate(args, root, checkpoint):
    scenarios = [
        ("stand", "upright", 0.0, 43),
        ("forward-020", "upright", 0.20, 43),
        ("forward-050", "upright", 0.50, 97),
        ("forward-080", "upright", 0.80, 43),
        ("backward-020", "upright", -0.20, 97),
        ("rise-forward-020", "floor", 0.20, 97),
    ]
    results = {}
    for name, start, target, seed in scenarios:
        results[name] = run_evaluation(root, checkpoint, name, start, target, seed, args.eval_envs, 1000, args.route_practice)
        record(root / "evaluation.json", results)

    rise = results["rise-forward-020"] or {}
    options = [
        ("rise-forward-020", "floor", 0.20, 97),
        ("forward-020", "upright", 0.20, 43),
        ("forward-050", "upright", 0.50, 97),
    ]
    name, start, target, seed = max(
        options, key=lambda item: (
            (results[item[0]] or {}).get("completed_hind_walk", 0),
            item[0] == "rise-forward-020" and rise.get("completed_hind_walk", 0) > 0,
            (results[item[0]] or {}).get("hind_only_support_fraction", 0.0),
        ),
    )
    results["video"] = run_evaluation(root, checkpoint, "video", start, target, seed, 1, 650, args.route_practice, video=True)
    record(root / "evaluation.json", results)


def run_evaluation(root, checkpoint, name, start, command, seed, num_envs, steps, route_practice, video=False):
    output = root / "eval" / name
    output.parent.mkdir(exist_ok=True)
    argv = [
        sys.executable, "-u", str(TRAINER), "--headless", KIT, "--evaluate",
        "--checkpoint", str(checkpoint), "--output", str(output), "--start", start,
        "--target", str(command), "--seed", str(seed), "--num_envs", str(num_envs), "--trial-steps", str(steps),
    ]
    if video:
        argv.append("--video")
    if route_practice:
        argv.append("--route-practice")
    with (root / f"eval-{name}.log").open("w") as log:
        try:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
            if result.returncode == 0:
                return read_status(output / "metrics.json")
        except subprocess.TimeoutExpired:
            print(f"{timestamp()} Evaluation timed out: {name}", flush=True)
    return None


def power_off(root, state):
    # The provider's shutdown wrapper empties Trash. Preserve its contents if present.
    if TRASH.is_dir() and any(path.is_file() or path.is_symlink() for path in TRASH.rglob("*")):
        backup = TRASH.with_name(f"Trash.before-autodl-shutdown-{int(time.time())}")
        TRASH.rename(backup)
        state["trash_preserved_at"] = str(backup)
    state.update(phase="shutdown_requested", shutdown_requested_at=timestamp())
    record(root / "supervisor.json", state)
    print(f"{timestamp()} Calling AutoDL shutdown", flush=True)
    os.sync()
    result = subprocess.run(["/bin/bash", str(SHUTDOWN)], check=False, timeout=15)
    state["shutdown_command_exit"] = result.returncode
    record(root / "supervisor.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=18_000)
    parser.add_argument("--elapsed", type=float, default=0.0)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--eval-envs", type=int, default=128)
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-practice", action="store_true")
    motion.add_argument("--route-practice", action="store_true")
    parser.add_argument("--swing-practice", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.elapsed < args.seconds:
        parser.error("Elapsed time must be below the training budget.")
    if args.elapsed and not args.checkpoint:
        parser.error("Resuming elapsed training time requires a checkpoint.")
    if args.swing_practice and not args.route_practice:
        parser.error("Swing practice requires the fixed route.")
    root = args.output.resolve()
    if args.shutdown and (Path("/root/autodl-tmp") not in root.parents or not SHUTDOWN.is_file()):
        parser.error("Shutdown is available only from the configured AutoDL data disk.")
    root.mkdir(parents=True, exist_ok=False)
    state = {
        "phase": "starting", "start_time": timestamp(), "budget_seconds": args.seconds,
        "initial_elapsed_training_seconds": args.elapsed, "attempts": [],
    }
    record(root / "supervisor.json", state)
    try:
        checkpoint = training(args, root, state)
        if checkpoint:
            state["phase"] = "evaluating"
            record(root / "supervisor.json", state)
            evaluate(args, root, checkpoint)
    except Exception as error:
        state["failure"] = repr(error)
        raise
    finally:
        state.update(phase="ready_for_archive", archive_ready_at=timestamp())
        record(root / "supervisor.json", state)
        if args.shutdown:
            deadline = time.monotonic() + 180
            while not (root / "archive-received").exists() and time.monotonic() < deadline:
                time.sleep(3)
            power_off(root, state)


if __name__ == "__main__":
    main()
