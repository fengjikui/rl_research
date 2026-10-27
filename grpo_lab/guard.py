"""Low-load launcher; log macOS thermal state and pause only our process group."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil


def thermal_state():
    if os.uname().sysname != 'Darwin': return None
    try:
        ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send = objc.objc_msgSend
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        send.restype = ctypes.c_void_p
        process = send(objc.objc_getClass(b'NSProcessInfo'), objc.sel_registerName(b'processInfo'))
        send.restype = ctypes.c_long
        return int(send(process, objc.sel_registerName(b'thermalState')))
    except (OSError, AttributeError): return None


def snapshot(pid=None):
    mem = psutil.virtual_memory()
    rss = 0
    if pid:
        try:
            p = psutil.Process(pid)
            rss = sum(x.memory_info().rss for x in [p, *p.children(recursive=True)] if x.is_running())
        except psutil.Error: pass
    return dict(time=datetime.datetime.now().astimezone().isoformat(),
                thermal_state=thermal_state(), cpu_percent=psutil.cpu_percent(interval=.2),
                available_gib=round(mem.available/2**30, 2), memory_percent=mem.percent,
                task_rss_gib=round(rss/2**30, 2), load1=os.getloadavg()[0])


def pressure(s):
    return (s['thermal_state'] is not None and s['thermal_state'] >= 1) or s['cpu_percent'] > 75 or s['available_gib'] < 6 or s['task_rss_gib'] > 9


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--max-seconds', type=int, default=1800)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: p.error('provide a command after --')
    log = Path(args.log); log.parent.mkdir(parents=True, exist_ok=True)
    initial = snapshot()
    if pressure(initial): raise SystemExit(f'Not starting under resource pressure: {initial}')
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    process = subprocess.Popen(['nice', '-n', '10', *command], env=env, start_new_session=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'guard interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    start, paused, good = time.monotonic(), False, 0
    try:
        while process.poll() is None:
            s = snapshot(process.pid)
            if pressure(s) and not paused:
                os.killpg(process.pid, signal.SIGSTOP)
                paused = True; good = 0
                print('GUARD pause', s, flush=True)
            elif paused:
                good = good+1 if not pressure(s) else 0
                if good >= 4:
                    os.killpg(process.pid, signal.SIGCONT)
                    paused = False
                    print('GUARD resume', s, flush=True)
            s['paused'] = paused
            with log.open('a') as f: f.write(json.dumps(s)+'\n')
            if time.monotonic()-start > args.max_seconds:
                raise TimeoutError('stage time budget exceeded; use last saved checkpoint')
            time.sleep(5)
        final = snapshot()
        final.update(event='completed', returncode=process.returncode)
        with log.open('a') as f: f.write(json.dumps(final)+'\n')
        raise SystemExit(process.returncode)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGCONT)
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__': main()
