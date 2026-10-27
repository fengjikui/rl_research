"""Run the complete small CPU curriculum. No pretrained model downloads."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = [
    ('00_bandit.py', 'bandit', []),
    ('01_q_table.py', 'q_table', []),
    ('02_dqn.py', 'dqn_mlp', ['--encoder', 'mlp']),
    ('02_dqn.py', 'dqn_cnn', ['--encoder', 'cnn']),
    ('03_policy_gradient.py', 'reinforce', ['--algo', 'reinforce']),
    ('03_policy_gradient.py', 'ac', ['--algo', 'ac']),
    ('03_policy_gradient.py', 'ppo', ['--algo', 'ppo']),
    ('04_memory.py', 'memory_mlp', ['--encoder', 'mlp']),
    ('04_memory.py', 'memory_gru', ['--encoder', 'gru']),
    ('05_tiny_lm.py', 'tiny_sft', ['--algo', 'sft']),
    ('05_tiny_lm.py', 'tiny_dpo', ['--algo', 'dpo']),
    ('05_tiny_lm.py', 'tiny_grpo', ['--algo', 'grpo']),
    ('06_tool_agent.py', 'tool_agent', []),
]

if __name__ == '__main__':
    subprocess.run([sys.executable, 'tutorials/code/check_invariants.py'], cwd=ROOT, check=True)
    for script, name, args in JOBS:
        print(f'\nRunning {name}', flush=True)
        subprocess.run([sys.executable, f'tutorials/code/{script}', *args,
                        '--out', f'tutorials/runs/{name}'], cwd=ROOT, check=True)
