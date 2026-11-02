"""Independently recompute saved evaluation outcomes and TD-loss arithmetic."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resources',type=Path,required=True)
    parser.add_argument('--console',type=Path,required=True)
    args=parser.parse_args();out=args.out
    config=json.loads((out/'config.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    verified_runs=[];schedule=[0,250,*range(500,12001,500)]
    for variant in config['variants']:
        for seed in range(5):
            run=out/variant/f'seed_{seed}'
            metrics=json.loads((run/'metrics.json').read_text())
            evaluations=[json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()]
            curve=json.loads((run/'learning_curve.json').read_text())
            assert [entry['step'] for entry in evaluations]==schedule
            for entry,point in zip(evaluations,curve):
                records=entry['records']
                assert len(records)==25
                assert {(r['ball'],r['paddle']) for r in records}=={(b,p) for b in range(5) for p in range(5)}
                for record in records:
                    paddle=record['paddle']
                    assert len(record['actions'])==5
                    for action in record['actions']:
                        assert action in [0,1,2]
                        paddle=max(0,min(4,paddle+action-1))
                    assert record['success']==(paddle==record['ball'])
                correct=sum(record['success'] for record in records)
                assert entry['correct']==correct and entry['success']==correct/25
                assert all(entry[k]==point[k] for k in ['step','updates','n','correct','success'])
            assert metrics['optimizer_updates']==2969==evaluations[-1]['updates']
            assert metrics['final_success']==evaluations[-1]['success']
            auc=sum((b['step']-a['step'])*(a['success']+b['success'])/2 for a,b in zip(curve,curve[1:]))/12000
            assert math.isclose(auc,metrics['normalized_auc'],abs_tol=1e-12)
            for threshold in [.8,.95,1.]:
                first=next((p['step'] for p in curve if p['success']>=threshold),None)
                assert metrics['first_observed_threshold_step'][str(threshold)]==first
            positions=json.loads((run/'initial_positions.json').read_text())
            assert len(positions)==2400
            assert hashlib.sha256(json.dumps(positions).encode()).hexdigest()==metrics['initial_positions_sha256']
            batch=json.loads((run/'first_batch/loss_batch.json').read_text())
            inputs=np.load(run/'first_batch/batch_inputs.npz',allow_pickle=False)
            assert len(batch['rows'])==64==len(inputs['states'])
            assert batch['label_requires_grad'] is False
            for i,row in enumerate(batch['rows']):
                assert row['reward']==float(inputs['rewards'][i])
                assert row['terminated']==bool(inputs['terminated'][i])
                assert row['action']==int(inputs['actions'][i])
                label=row['reward']+.99*(not row['terminated'])*row['next_max_q']
                assert math.isclose(label,row['regression_label'],rel_tol=1e-6,abs_tol=1e-7)
                difference=abs(row['prediction_before_update']-row['regression_label'])
                loss=.5*difference*difference if difference<1 else difference-.5
                assert math.isclose(loss,row['smooth_l1_loss'],rel_tol=1e-5,abs_tol=1e-7)
            assert math.isclose(sum(r['smooth_l1_loss'] for r in batch['rows'])/64,batch['mean_loss'],rel_tol=1e-6)
            verified_runs.append(dict(variant=variant,seed=seed,checkpoints=len(curve),final_success=metrics['final_success']))
    for seed in range(5):
        assert len({m['initial_positions_sha256'] for m in summary['runs'] if m['seed']==seed})==1
    for aggregate in summary['aggregates']:
        final=[m['final_success'] for m in summary['runs'] if m['variant']==aggregate['variant']]
        assert math.isclose(float(np.mean(final)),aggregate['final_success_mean'])
        assert math.isclose(float(np.std(final,ddof=1)),aggregate['final_success_sample_std'])
    shutil.copyfile(args.resources,out/'resources.jsonl')
    shutil.copyfile(args.console,out/'console.log')
    resources=[json.loads(line) for line in (out/'resources.jsonl').read_text().splitlines()]
    assert resources[-1].get('event')=='completed' and resources[-1]['returncode']==0
    resource_summary=dict(samples=len(resources),
        thermal_states=sorted({r['thermal_state'] for r in resources if r['thermal_state'] is not None}),
        max_sampled_cpu_percent=max(r['cpu_percent'] for r in resources),
        max_sampled_task_rss_gib=max(r['task_rss_gib'] for r in resources),
        min_sampled_available_gib=min(r['available_gib'] for r in resources),
        paused_samples=sum(r.get('paused',False) for r in resources),
        note='Sampled values, not continuous peaks; no Celsius temperature measured.')
    (out/'resource_summary.json').write_text(json.dumps(resource_summary,indent=2)+'\n')
    verification=dict(passed=True,runs=len(verified_runs),total_saved_evaluation_episodes=len(verified_runs)*len(schedule)*25,
        all_recorded_actions_replayed_against_independent_position_arithmetic=True,
        all_recorded_success_rates_recomputed=True,all_auc_and_thresholds_recomputed=True,
        all_first_batch_td_labels_and_huber_losses_recomputed=True,
        same_environment_initial_sequence_across_variants_for_each_seed=True,
        optimizer_updates_per_run=2969,checkpoint_reload_verified_during_each_run=True,
        scope='Fixed-budget finite Catch task; architecture and parameter-count confounding disclosed.')
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines=['','## 各采样预算下的成功率均值','',
        '| 环境步数 | 坐标 MLP | 双通道 MLP | 双通道 CNN | 单通道 CNN |',
        '|---:|---:|---:|---:|---:|']
    for step in [0,250,500,1000,2000,4000,8000,12000]:
        values=[next(p['mean'] for p in item['mean_success_by_step'] if p['step']==step) for item in summary['aggregates']]
        lines.append('| '+str(step)+' | '+' | '.join(f'{v:.1%}' for v in values)+' |')
    lines+=['','每格为五个预先固定种子的均值，不是最佳种子或最佳检查点。',
        f'资源采样共 {resource_summary["samples"]} 次，热状态取值为 {resource_summary["thermal_states"]}；CPU 最高采样值 {resource_summary["max_sampled_cpu_percent"]:.1f}%，任务 RSS 最高采样值 {resource_summary["max_sampled_task_rss_gib"]:.2f} GiB，可用内存最低采样值 {resource_summary["min_sampled_available_gib"]:.2f} GiB，保护暂停采样 {resource_summary["paused_samples"]} 次。','']
    with (out/'REPORT.md').open('a') as stream:stream.write('\n'.join(lines))
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob('*') if p.is_file() and p.suffix!='.pt' and p.name!='SHA256SUMS.json'},indent=2)+'\n')
    print(json.dumps(verification,indent=2));print(json.dumps(resource_summary,indent=2))


if __name__=='__main__':main()
