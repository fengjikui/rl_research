"""Explicit accelerator selection and portable, CPU-valued checkpoints."""
import importlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


def select_device(name):
    if name=='cpu':return torch.device('cpu')
    if name=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
        torch.cuda.set_device(0);return torch.device('cuda:0')
    if name=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS requested but unavailable; refusing CPU fallback')
        return torch.device('mps')
    if name=='npu':
        try:importlib.import_module('torch_npu')
        except ImportError as error:raise RuntimeError('NPU requested: install matching CANN/PyTorch/torch_npu first') from error
        if not torch.npu.is_available():raise RuntimeError('Ascend NPU requested but unavailable; refusing CPU fallback')
        torch.npu.set_device('npu:0');return torch.device('npu:0')
    raise ValueError(f'Unknown device {name}')


def setup(seed,threads=2):
    if threads<1:raise ValueError('threads must be positive')
    os.environ.setdefault('OMP_NUM_THREADS',str(threads));os.environ.setdefault('OPENBLAS_NUM_THREADS',str(threads))
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


def atomic_torch(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(cpu_copy(value),temporary);temporary.replace(path)


def rng_state(device):
    state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    backend=device.type
    if backend in ('cuda','mps','npu'):
        state['accelerator']=getattr(torch,backend).get_rng_state()
    return state


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if 'accelerator' in state and device.type in ('cuda','mps','npu'):
        getattr(torch,device.type).set_rng_state(state['accelerator'])


def local_resource_snapshot():
    # Shared project guard provides the macOS thermal enum where available.
    root=str(Path(__file__).resolve().parents[1])
    if root not in sys.path:sys.path.insert(0,root)
    try:
        from grpo_lab.guard import snapshot
        return snapshot(os.getpid())
    except ImportError:
        import psutil
        memory=psutil.virtual_memory()
        return dict(thermal_state=None,cpu_percent=psutil.cpu_percent(.2),
                    available_gib=round(memory.available/2**30,2),task_rss_gib=round(psutil.Process().memory_info().rss/2**30,2))


def resource_pressure(snapshot):
    return ((snapshot['thermal_state'] is not None and snapshot['thermal_state']>=2)
        or snapshot['cpu_percent']>75 or snapshot['available_gib']<6 or snapshot['task_rss_gib']>9)
