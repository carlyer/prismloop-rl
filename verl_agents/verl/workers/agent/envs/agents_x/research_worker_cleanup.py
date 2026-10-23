"""Bounded cleanup of only owned Python tools; never terminate a Ray actor."""
import time
import warnings

_pending = []


def reap_pending():
    for process in list(_pending):
        process.join(timeout=0)
        if not process.is_alive():
            process.close()
            _pending.remove(process)
    if len(_pending) >= 4:
        raise RuntimeError('Tool worker quarantine full; refusing to leak more workers')


def terminate_worker(worker):
    process = worker.process
    if process is None:
        return
    try:
        if process.is_alive():
            worker.input_queue.put(None)
            process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.is_alive():
            # Some inherited I/O can remain temporarily uninterruptible. Keep
            # the handle and bound future creation rather than aborting PPO.
            _pending.append(process)
            warnings.warn(f'Tool PID {process.pid} quarantined after SIGKILL', RuntimeWarning)
        else:
            process.join(timeout=0)
            process.close()
    finally:
        worker.process = None
        for channel in (worker.input_queue, worker.output_queue):
            channel.cancel_join_thread()
            channel.close()
