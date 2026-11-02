"""Read-only server inventory for a company Agent; no install or training.

Run inside the intended environment, then send the JSON back with secrets removed.
No environment variables, SSH configuration or credentials are collected.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(name, args):
    if not shutil.which(name):
        return dict(available=False)
    try:
        result = subprocess.run([name, *args], capture_output=True, text=True, timeout=20)
        return dict(available=True, returncode=result.returncode,
                    output=(result.stdout + result.stderr)[-12000:])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(available=True, error=str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = dict(python=sys.version, executable=sys.executable, system=platform.system(),
                  architecture=platform.machine(), cpu_count=os.cpu_count())
    result['packages'] = {}
    for package in ('torch', 'torch-npu', 'gymnasium', 'craftium', 'numpy', 'Box2D'):
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    try:
        import psutil
        memory = psutil.virtual_memory()
        result['memory_gib'] = dict(total=round(memory.total/2**30, 2), available=round(memory.available/2**30, 2))
    except ImportError:
        result['memory_gib'] = None
    for key, name, options in (('npu_smi', 'npu-smi', ['info']),
                               ('nvidia_smi', 'nvidia-smi', ['--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv']),
                               ('graphics', 'glxinfo', ['-B'])):
        result[key] = command(name, options)
    try:
        import torch
        if importlib.util.find_spec('torch_npu'):
            import torch_npu  # Registers the device backend; no tensors or learner are started.
        result['torch_devices'] = dict(cuda_available=torch.cuda.is_available(),
                                       cuda_count=torch.cuda.device_count(),
                                       npu_available=bool(hasattr(torch, 'npu') and torch.npu.is_available()))
        if result['torch_devices']['npu_available']:
            result['torch_devices']['npu_count'] = torch.npu.device_count()
    except Exception as error:
        result['torch_probe_error'] = f'{type(error).__name__}: {error}'
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
