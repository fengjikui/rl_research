"""Recompute actual saved scores/losses and check checkpoint/video provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import psutil
import torch

from breakout import NatureQNet, make_env


def verify(directory):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    stages = json.loads((directory/'stages.json').read_text())
    checks = []
    raw_episodes = raw_rows = 0
    for stage in stages:
        path = directory/stage['stage']
        pruned=bool(stage.get('artifacts_pruned'))
        if not pruned:
            assert hashlib.sha256((path/'model.pt').read_bytes()).hexdigest()==stage['model_sha256']
        scores=[]
        for i,episode in enumerate(stage['evaluation']['episodes']):
            rows=[json.loads(x) for x in (path/f'episode_{i:02d}.jsonl').read_text().splitlines()]
            assert len(rows)==episode['steps']
            cumulative=0.0
            for row in rows:
                cumulative+=row['reward']
                assert abs(cumulative-row['score'])<1e-6
                if row['forced_serve']: assert row['action']==1
            assert abs(cumulative-episode['score'])<1e-6
            scores.append(cumulative); raw_episodes+=1; raw_rows+=len(rows)
        assert abs(float(np.mean(scores))-stage['evaluation']['mean_score'])<1e-6
        if pruned:
            checks.append(dict(stage=stage['stage'],scores=scores,artifacts_pruned=True,
                raw_scores_recomputed=True,reloaded_model_inference_matches=None))
            continue
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,width,height','-of','json',str(path/'play.mp4')]))['streams'][0]
        assert int(video['nb_frames'])==stage['evaluation']['episodes'][0]['steps']
        # 重新加载磁盘上的权重，在相同初态独立前向，核对实际记录的 Q 值。
        model=NatureQNet(4)
        model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True))
        env=make_env()
        obs,_=env.reset(seed=stage['evaluation']['episodes'][0]['seed'])
        first_rows=[json.loads(x) for x in (path/'episode_00.jsonl').read_text().splitlines()[:3]]
        with torch.no_grad():
            for row in first_rows:
                q=model(torch.as_tensor(np.asarray(obs))[None])[0].numpy()
                np.testing.assert_allclose(q,row['q_values'],rtol=5e-4,atol=5e-4)
                obs,reward,done,truncated,_=env.step(row['action'])
                assert reward==row['reward']
                if done or truncated:break
        env.close()
        checks.append(dict(stage=stage['stage'],scores=scores,video_frames=int(video['nb_frames']),
                           model_hash_matches=True,reloaded_model_inference_matches=True))
    audit_path=directory/'first_batch/loss_batch.json'
    if audit_path.is_file():
        audit=json.loads(audit_path.read_text())
        assert not audit['label_requires_grad']
        losses=[]
        for row in audit['rows']:
            expected=row['reward']+row['gamma']*(1-row['terminated'])*row['next_max_q']
            assert abs(expected-row['regression_label'])<1e-5
            error=abs(row['prediction_before_update']-row['regression_label'])
            loss=.5*error**2 if error<1 else error-.5
            assert abs(loss-row['smooth_l1_loss'])<1e-5
            losses.append(loss)
        assert abs(float(np.mean(losses))-audit['mean_loss'])<1e-5
    latest=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    assert latest['replay_count']==json.loads((directory/'replay/metadata.json').read_text())['count']
    if json.loads((directory/'status.json').read_text())['status']=='completed':
        weights=torch.load(directory/stages[-1]['stage']/'model.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,weights[k]) for k,v in latest['online'].items())
    result=dict(passed=True,stage_count=len(stages),raw_episodes=raw_episodes,raw_rows=raw_rows,checks=checks,
        first_batch_labels_and_losses_recomputed=audit_path.is_file(),checkpoint_replay_counts_match=True)
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--supervisor-pid',type=int)
    args=parser.parse_args()
    start=time.monotonic()
    if args.wait:
        while True:
            status_path=args.directory/'status.json'
            state=json.loads(status_path.read_text()).get('status') if status_path.is_file() else None
            if state in ('completed','stopped','failed'):break
            if time.monotonic()-start>2500:raise TimeoutError('Training completion wait exceeded')
            if args.supervisor_pid and not psutil.pid_exists(args.supervisor_pid):
                raise RuntimeError('Training supervisor exited before a terminal status was saved')
            time.sleep(15)
        if state=='failed':raise RuntimeError('Training failed; see error.json')
    result=verify(args.directory)
    resource_path=args.directory.parent/'resources.jsonl'
    if resource_path.is_file():
        samples=[json.loads(x) for x in resource_path.read_text().splitlines()]
        summary=dict(samples=len(samples),thermal_states=sorted({s['thermal_state'] for s in samples},key=str),
            max_cpu_percent=max(s['cpu_percent'] for s in samples),
            min_available_gib=min(s['available_gib'] for s in samples),
            max_task_rss_gib=max(s['task_rss_gib'] for s in samples),
            paused_samples=sum(bool(s.get('paused')) for s in samples))
        (args.directory/'resource_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
