"""Summarize only measured results and resource telemetry."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    game = json.loads((ROOT/'runs/game/results.json').read_text())
    evaluations = []
    for p in sorted((ROOT/'runs/llm_reverse').glob('validation_*.json')):
        r = json.loads(p.read_text()); evaluations.append((r['step'],r['success'],r['n']))
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    history = game['history']
    axes[0].plot([r['step'] for r in history],[r['success'] for r in history],marker='o',label='GRPO policy')
    axes[0].axhline(.95,linestyle='--',color='gray',label='target 95%')
    axes[0].set(title='Catch: all 25 initial states',xlabel='Environment transitions',ylabel='Greedy success',ylim=(0,1.05))
    for n in sorted({x[2] for x in evaluations}):
        points=[x for x in evaluations if x[2]==n]
        axes[1].plot([x[0] for x in points],[x[1] for x in points],marker='o',label=f'Validation n={n}')
    axes[1].axhline(.8,linestyle='--',color='gray',label='development target 80%')
    axes[1].set(title='Qwen 0.5B: development scores (sample sizes differ)',xlabel='GRPO updates',ylabel='Exact match',ylim=(0,1.05))
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT/'runs/progress.png',dpi=160); plt.close(fig)
    rows=[]
    for path in sorted((ROOT/'runs').glob('*resources.jsonl')):
        records=[json.loads(line) for line in path.read_text().splitlines()]
        rows.append(dict(stage=path.stem,samples=len(records),
                  observed_max_cpu_percent=max(r['cpu_percent'] for r in records),
                  observed_max_task_rss_gib=max(r['task_rss_gib'] for r in records),
                  min_available_gib=min(r['available_gib'] for r in records),
                  thermal_states=sorted({r['thermal_state'] for r in records if r['thermal_state'] is not None}),
                  paused_samples=sum(r.get('paused',False) for r in records)))
    (ROOT/'runs/resource_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'game_final':game['metrics']['final_success'],'validation':evaluations,'resources':rows},indent=2))


if __name__ == '__main__': main()
