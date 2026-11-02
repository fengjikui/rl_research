"""Continue owned Breakout jobs until score milestones, with guarded segments.

No fixed morning deadline. Uses continuous_control.json for the stop switch.
Pruning runs only between segments and retains all score/action JSON records.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grpo_lab.guard import snapshot, pressure


def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def read(path):return json.loads(path.read_text())


def alive(pid):
    try:return psutil.Process(pid).is_running() and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False


def select_retained(stages,run,pinned,current_time):
    keep={stages[0]['stage'],stages[-1]['stage'],max(stages,key=lambda s:s['evaluation']['mean_score'])['stage'],*pinned}
    buckets={}
    for s in stages:
        name=s['stage'];timestamp=(run/name/'evaluation.json').stat().st_mtime
        if current_time-timestamp<7200:keep.add(name)
        bucket=int(timestamp//3600)
        previous=buckets.get(bucket)
        if previous is None or s['evaluation']['mean_score']>previous['evaluation']['mean_score']:
            buckets[bucket]=s
    keep.update(s['stage'] for s in buckets.values())
    return keep


def prune(root,pinned):
    run=root/'run';stages=read(run/'stages.json')
    keep=select_retained(stages,run,pinned,time.time())
    changed=False
    with (root/'retention.jsonl').open('a') as log:
        for s in stages:
            if s['stage'] in keep or s.get('artifacts_pruned'):continue
            path=(run/s['stage']).resolve()
            if path.parent!=run.resolve():raise RuntimeError('Unexpected stage directory')
            removed=[]
            for filename in ['model.pt','play.mp4','preview.gif']:
                file=path/filename
                if file.is_file():file.unlink();removed.append(filename)
            s['artifacts_pruned']=True;s['pruned_at']=now();changed=True
            log.write(json.dumps(dict(stage=s['stage'],removed=removed,time=now(),
                reason='older than 2h; hourly/best/initial/latest/milestone stages retained'))+'\n')
    if changed:write(run/'stages.json',stages)


def archive_round(run):
    parent=run/'sessions';parent.mkdir(exist_ok=True)
    n=1
    while (parent/f'round{n:02d}').exists():n+=1
    directory=parent/f'round{n:02d}';directory.mkdir()
    for name in ['config.json','result.json','verification.json','resource_summary.json','status.json','stages.json','REPORT.md']:
        if (run/name).is_file():shutil.copy2(run/name,directory/name)
    shutil.copytree(run/'source',directory/'source')
    for name in ['result.json','verification.json','resource_summary.json']:
        if (run/name).exists():(run/name).unlink()
    return n


def guarded(root,command,log_name,limit):
    python=str(Path(sys.executable).absolute())
    full=[python,'grpo_lab/guard.py','--log',str(root/'resources.jsonl'),
        '--max-seconds',str(limit),'--thermal-pause-at','2','--',*command]
    with (root/log_name).open('w') as log:
        return subprocess.Popen(full,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def check_milestone(root,state,report_status):
    run=root/'run';stages=read(run/'stages.json')
    target=next((t for t in [200,300,400] if str(t) not in state['milestones']),None)
    if target is None:return True
    candidates=[s for s in stages if not s.get('artifacts_pruned') and s['step']>state.get('last_checked_step',0)
                and s['evaluation']['mean_score']>=target]
    if not candidates:return False
    candidate=max(candidates,key=lambda s:s['step'])
    out=root/'milestone_checks'/f's{candidate["step"]:09d}'
    if not out.exists():
        report_status('checking_milestone',target=target,step=candidate['step'])
        command=[str(Path(sys.executable).absolute()),'dqn_lab/evaluate_saved.py',
                 '--model',str(run/candidate['stage']/'model.pt'),'--out',str(out)]
        process=guarded(root,command,f'milestone_{candidate["step"]}.log',600)
        process.wait()
        if process.returncode or not (out/'evaluation.json').is_file():
            raise RuntimeError('Milestone evaluation failed; see milestone log')
    value=read(out/'evaluation.json')
    state['last_checked_step']=candidate['step']
    # 三个不同训练检查点的独立多局均分均达标，才算稳定达到。
    for threshold in [200,300,400]:
        key=str(threshold)
        if key in state['milestones']:continue
        history=state['confirmations'].setdefault(key,[])
        if value['mean_score']>=threshold:
            if not history or candidate['step']-history[-1]['step']>=10000:
                history.append(dict(step=candidate['step'],mean_score=value['mean_score'],
                    capped_episodes=sum(e['capped'] for e in value['episodes']),
                    score_scope='up to 4500 decisions per episode; capped flags retained',
                    stage=candidate['stage'],evaluation=str(out/'evaluation.json')))
        else:history.clear()
        if len(history)>=3:
            state['milestones'][key]=dict(time=now(),confirmations=history[-3:])
            state['pinned'].extend(x['stage'] for x in history[-3:])
    write(root/'continuous_state.json',state)
    return '400' in state['milestones']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--initial-manifest',required=True,type=Path)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();root=args.root;run=root/'run'
    controls=root/'continuous_control.json'
    defaults=dict(enabled=True,segment_seconds=1800,stage_seconds=300,final_target=400,
                  milestones=[200,300,400],thermal_pause_at=2)
    if not controls.exists():write(controls,defaults)
    if args.plan_only:
        print(read(controls));return
    state_path=root/'continuous_state.json'
    state=read(state_path) if state_path.exists() else dict(milestones={},confirmations={},pinned=[])
    manifest=read(args.initial_manifest)
    interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def report_status(phase,**extra):
        write(root/'continuous_status.json',dict(phase=phase,time=now(),controller_pid=__import__('os').getpid(),
            milestones=state['milestones'],current_supervisor_pid=manifest['supervisor_pid'],**extra))
    try:
        while not interrupted and read(controls)['enabled']:
            # 等待本任务创建的当前训练与结果核验结束，不启动重叠的训练进程。
            report_status('waiting_current_segment')
            while alive(manifest['supervisor_pid']) or (manifest.get('verification_pid') and alive(manifest['verification_pid'])):
                if interrupted or not read(controls)['enabled']:
                    report_status('stop_after_current_segment');return
                time.sleep(10)
            status=read(run/'status.json')
            if status['status']!='completed':raise RuntimeError('Previous training segment did not complete')
            if not (run/'verification.json').is_file() or not read(run/'verification.json')['passed']:
                raise RuntimeError('Previous segment did not pass independent verification')
            if check_milestone(root,state,report_status):
                report_status('target_400_confirmed');return
            while True:
                resource=snapshot()
                if not pressure(resource,2) and shutil.disk_usage(root).free>=12*2**30:break
                report_status('waiting_for_resources',resource=resource)
                if interrupted or not read(controls)['enabled']:return
                time.sleep(10)
            prune(root,state['pinned'])
            round_number=archive_round(run)
            control=read(controls)
            seconds=int(control['segment_seconds'])
            python=str(Path(sys.executable).absolute())
            command=[python,'dqn_lab/train_breakout.py','--out',str(run),'--device','mps',
                     '--seconds',str(seconds),'--stage-seconds',str(control['stage_seconds']),'--resume']
            process=guarded(root,command,f'console_round{round_number+1:02d}.log',seconds+600)
            manifest=dict(supervisor_pid=process.pid,started_at=now(),command=command,round=round_number+1)
            write(root/'continuous_launch.json',manifest)
            # 避免新核验进程误读上一段的 completed 状态。
            deadline=time.monotonic()+60
            deferred_for_resources=False
            while time.monotonic()<deadline:
                current=read(run/'status.json')
                if current['status'] in ['initializing','evaluating','training'] and current['pid']!=status['pid']:break
                if process.poll() is not None:
                    # 外层检查与 guard 启动之间，热状态仍可能升高。
                    # guard 尚未启动训练时的保护性拒绝应等待重试，不能丢失续训链路。
                    startup_log=(root/f'console_round{round_number+1:02d}.log').read_text()
                    if startup_log.startswith('Not starting under resource pressure:'):
                        archived=run/'sessions'/f'round{round_number:02d}'
                        for name in ['result.json','verification.json','resource_summary.json']:
                            if (archived/name).is_file():shutil.copy2(archived/name,run/name)
                        report_status('waiting_for_resources',reason='guard declined startup; retrying')
                        deferred_for_resources=True
                        time.sleep(10)
                        break
                    raise RuntimeError('New segment failed to start')
                time.sleep(1)
            else:raise RuntimeError('New segment did not publish a running status')
            if deferred_for_resources:continue
            with (root/f'verification_round{round_number+1:02d}.log').open('w') as log:
                verifier=subprocess.Popen(['nice','-n','10',python,'dqn_lab/verify_atari_run.py',str(run),
                    '--wait','--supervisor-pid',str(process.pid)],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest['verification_pid']=verifier.pid
            write(root/'continuous_launch.json',manifest)
            report_status('training',round=round_number+1)
        report_status('stopped_after_current_segment')
    except BaseException as error:
        report_status('failed',error_type=type(error).__name__,message=str(error))
        raise


if __name__=='__main__':main()
