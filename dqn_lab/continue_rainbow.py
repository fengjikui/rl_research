"""Sequential, thermally guarded Rainbow segments; never shares the DQN run.

The control file can disable future segments. Scores use held-out epsilon=.05
episodes, not the deterministic three-episode preview. All paths are local.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_lab.guard import snapshot, pressure
from dqn_lab.continue_training import prune


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def verify_scores(run):
    """Independent raw-reward sums, media existence and matching resume counters."""
    stages = read(run / 'stages.json')
    episodes = 0
    for stage in stages:
        directory = run / stage['stage']
        for i, episode in enumerate(stage['evaluation']['episodes']):
            rows = [json.loads(line) for line in (directory / f'episode_{i:02d}.jsonl').open()]
            assert len(rows) == episode['steps'], (stage['stage'], i, 'step count')
            assert abs(sum(row['reward'] for row in rows) - episode['score']) < 1e-6
            assert rows[-1]['score'] == episode['score']
            episodes += 1
        if not stage.get('artifacts_pruned'):
            assert (directory / 'model.pt').stat().st_size > 0
            assert (directory / 'play.mp4').stat().st_size > 0
    heldout = read(run / 'latest_heldout.json')
    directory = run / heldout['stage'] / 'heldout'
    scores = []
    for i, episode in enumerate(heldout['episodes']):
        rows = [json.loads(line) for line in (directory / f'episode_{i:02d}.jsonl').open()]
        assert len(rows) == episode['steps']
        score = sum(row['reward'] for row in rows)
        assert abs(score - episode['score']) < 1e-6
        scores.append(score)
    assert len(scores) >= 10 and heldout['epsilon'] == .05
    assert abs(sum(scores) / len(scores) - heldout['mean_score']) < 1e-6
    state = read(run / 'status.json')
    assert read(run / 'replay/metadata.json')['count'] == state['replay_count'] == state['step']
    value = dict(passed=True, time=now(), stage_evaluation_episodes=episodes,
                 heldout_episodes=len(scores), heldout_mean=heldout['mean_score'],
                 scope='raw reward sums, replay counters, retained model/video existence; not a convergence proof')
    write(run / 'score_verification.json', value)
    return heldout


def record_milestones(state, heldout):
    step = heldout['step']
    if step <= state.get('last_checked_step', -1):
        return
    state['last_checked_step'] = step
    for threshold in (200, 300, 400):
        key = str(threshold)
        if key in state['milestones']:
            continue
        history = state['confirmations'].setdefault(key, [])
        if heldout['mean_score'] >= threshold:
            if not history or step - history[-1]['step'] >= 10000:
                history.append(dict(step=step, mean_score=heldout['mean_score'], stage=heldout['stage'],
                                    capped_episodes=sum(e['capped'] for e in heldout['episodes'])))
        else:
            history.clear()
        if len(history) >= 3:
            state['milestones'][key] = dict(time=now(), confirmations=history[-3:])
            state['pinned'].extend(item['stage'] for item in history[-3:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    run = root / 'run'
    control_path = root / 'continuous_control.json'
    if not control_path.exists():
        write(control_path, dict(enabled=True, segment_seconds=1800, stage_seconds=300,
                                 thermal_pause_at=2, target=400))
    state_path = root / 'continuous_state.json'
    state = read(state_path) if state_path.exists() else dict(milestones={}, confirmations={}, pinned=[], rounds=0)
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

    def status(phase, **extra):
        write(root / 'continuous_status.json', dict(phase=phase, time=now(), controller_pid=os.getpid(),
                                                   milestones=state['milestones'], **extra))

    try:
        while read(control_path)['enabled'] and not stop_requested:
            control = read(control_path)
            resource = snapshot()
            if pressure(resource, control['thermal_pause_at']) or shutil.disk_usage(root).free < 12 * 2**30:
                status('waiting_for_resources', resource=resource)
                time.sleep(10)
                continue
            # Only one owned learner may be alive; Popen.wait below serializes segments.
            round_number = state['rounds'] + 1
            previous_step = read(run / 'status.json')['step'] if (run / 'status.json').exists() else 0
            seconds = int(control['segment_seconds'])
            python = str(Path(sys.executable).absolute())
            command = [python, 'dqn_lab/train_rainbow.py', '--out', str(run), '--device', 'mps',
                       '--seconds', str(seconds), '--stage-seconds', str(control['stage_seconds'])]
            if (run / 'latest.pt').exists():
                command.append('--resume')
            guarded = [python, 'grpo_lab/guard.py', '--log', str(root / 'resources.jsonl'),
                       '--max-seconds', str(seconds + 1200), '--thermal-pause-at',
                       str(control['thermal_pause_at']), '--', *command]
            console = root / f'console_round{round_number:03d}.log'
            with console.open('a') as log:
                process = subprocess.Popen(guarded, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'launch.json', dict(supervisor_pid=process.pid, started_at=now(), command=command,
                                            round=round_number))
            status('training', supervisor_pid=process.pid, round=round_number)
            returncode = process.wait()
            text = console.read_text(errors='replace')
            if returncode:
                # The thermal state can rise between the check and actual launch.
                no_new_data = not (run / 'status.json').exists() or read(run / 'status.json')['step'] == previous_step
                resource_refusal = ('Not starting under resource pressure:' in text or
                                    'Refusing high-load start under resource pressure:' in text)
                if no_new_data and resource_refusal:
                    status('waiting_for_resources', reason='startup guard declined; retrying')
                    time.sleep(10)
                    continue
                raise RuntimeError(f'Rainbow segment exited {returncode}; see {console}')
            learner_status = read(run / 'status.json')
            if learner_status['status'] != 'completed':
                status('stopped', learner_status=learner_status['status'])
                return
            status('verifying_scores', round=round_number)
            heldout = verify_scores(run)
            record_milestones(state, heldout)
            state['rounds'] = round_number
            write(state_path, state)
            archive = run / 'sessions' / f'round{round_number:03d}'
            archive.mkdir(parents=True, exist_ok=False)
            for name in ('config.json', 'status.json', 'result.json', 'latest_heldout.json',
                         'score_verification.json', 'stages.json', 'REPORT.md'):
                shutil.copy2(run / name, archive / name)
            if '400' in state['milestones']:
                status('target_400_confirmed', heldout_mean=heldout['mean_score'])
                return
            prune(root, state['pinned'])
        status('stopped_after_current_segment')
    except BaseException as error:
        status('failed', error_type=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    main()
