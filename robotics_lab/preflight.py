"""Read-only inventory for the documented NVIDIA Go2 simulation workflow.

No package installation, simulator import, model execution or training occurs.
The report is an inventory, not a certification that Isaac Sim will run.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ISAACLAB_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
TASK_FILE = "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


def command(args, cwd=None):
    executable = shutil.which(args[0])
    if executable is None:
        return {"available": False}
    try:
        result = subprocess.run([executable, *args[1:]], cwd=cwd, capture_output=True, text=True, timeout=15)
        return {"available": True, "returncode": result.returncode,
                "output": (result.stdout + result.stderr)[-10000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": str(error)}


def inventory(root=None):
    result = {"scope": "read-only inventory; does not test simulation or training",
              "system": platform.system(), "architecture": platform.machine(),
              "python": sys.version, "cpu_count": os.cpu_count(), "packages": {}}
    for package in ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        result["physical_memory_gib"] = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30, 2)
    except (ValueError, OSError):
        result["physical_memory_gib"] = None
    result["nvidia_smi"] = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu", "--format=csv"])
    result["glibc"] = command(["ldd", "--version"])
    if root is not None:
        root = root.expanduser().resolve()
        commit = command(["git", "rev-parse", "HEAD"], cwd=root)
        task = root / TASK_FILE
        result["isaaclab_source"] = {
            "root": str(root), "commit": commit,
            "expected_commit": ISAACLAB_COMMIT,
            "commit_matches": commit.get("returncode") == 0 and commit.get("output", "").strip() == ISAACLAB_COMMIT,
            "go2_task_registered_in_source": task.is_file() and TASK in task.read_text(),
            "train_entry_exists": (root / "scripts/reinforcement_learning/rsl_rl/train.py").is_file(),
            "play_entry_exists": (root / "scripts/reinforcement_learning/rsl_rl/play.py").is_file(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(inventory(args.isaaclab_root), ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
