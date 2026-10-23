"""CPU-only shutdown and conservative live-cleanup gate checks."""
import ast
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import unittest

import patch_tool_cleanup as patch
import reclaim_expired_image_tools as cleanup


namespace = {}
exec('class Worker:\n' + patch.REPLACEMENT, namespace)
Worker = namespace['Worker']


def graceful(channel):
    while channel.get() is not None:
        pass


def ignores_term(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(1)


class CleanupTests(unittest.TestCase):
    def make_worker(self, target, args=()):
        worker = Worker()
        worker.input_queue, worker.output_queue = mp.Queue(), mp.Queue()
        worker.process = mp.Process(target=target, args=args or (worker.input_queue,))
        worker.process.start()
        self.addCleanup(self.emergency_cleanup, worker)
        return worker

    def emergency_cleanup(self, worker):
        if worker.process is not None:
            if worker.process.is_alive():
                worker.process.kill()
            worker.process.join(timeout=3)
            worker.process.close()
        for channel in (worker.input_queue, worker.output_queue):
            channel.cancel_join_thread()
            channel.close()

    def test_graceful_reap_and_idempotence(self):
        worker = self.make_worker(graceful)
        pid = worker.process.pid
        worker.terminate()
        worker.terminate()
        self.assertIsNone(worker.process)
        self.assertFalse(Path(f'/proc/{pid}').exists())

    def test_dead_process_reaped(self):
        worker = self.make_worker(time.sleep, (0.01,))
        pid = worker.process.pid
        time.sleep(0.1)
        worker.terminate()
        self.assertIsNone(worker.process)
        self.assertFalse(Path(f'/proc/{pid}').exists())

    def test_stuck_process_escalates_and_reaps(self):
        ready = mp.Event()
        worker = self.make_worker(ignores_term, (ready,))
        self.assertTrue(ready.wait(3))
        pid = worker.process.pid
        start = time.monotonic()
        worker.terminate()
        self.assertGreaterEqual(time.monotonic() - start, 6.8)
        self.assertLess(time.monotonic() - start, 12)
        self.assertFalse(Path(f'/proc/{pid}').exists())
        self.assertIsNone(worker.process)

    def child(self, birth, **changes):
        value = dict(pid=123, state='S', ppid=next(iter(cleanup.PARENTS)),
                     start_ticks=int((birth-cleanup.BOOT)*cleanup.TICKS))
        return value | changes

    def test_expired_child_eligible(self):
        now = time.time()
        self.assertTrue(cleanup.eligible(self.child(now-3600), now-100, now))

    def test_current_batch_rejected(self):
        now = time.time()
        self.assertFalse(cleanup.eligible(self.child(now-1000), now-1200, now))

    def test_boundary_guard_rejected(self):
        now = time.time()
        self.assertFalse(cleanup.eligible(self.child(now-1050), now-1000, now))

    def test_young_rejected(self):
        now = time.time()
        self.assertFalse(cleanup.eligible(self.child(now-500), now, now))

    def test_unrelated_or_zombie_rejected(self):
        now = time.time()
        self.assertFalse(cleanup.eligible(self.child(now-3600, ppid=os.getpid()), now, now))
        self.assertFalse(cleanup.eligible(self.child(now-3600, state='Z'), now, now))
        self.assertFalse(cleanup.eligible(None, now, now))

    def test_exact_stack_required(self):
        stack = ('_worker_loop (' + str(cleanup.TOOL) + ':72)\nexec_code (' +
                 str(cleanup.TOOL) + ':227)\nwrite '
                 '(/opt/pyvision-rl-venv/lib/python3.10/site-packages/ray/_private/utils.py:424)\n'
                 '_showwarnmsg_impl (/usr/lib/python3.10/warnings.py:30)')
        self.assertTrue(cleanup.blocked_tool(stack))
        self.assertFalse(cleanup.blocked_tool(stack.replace('ray/_private/utils.py', 'unrelated.py')))
        self.assertFalse(cleanup.blocked_tool(stack.replace(str(cleanup.TOOL), '/unrelated/tool.py')))
        kmeans = (f'_worker_loop ({cleanup.TOOL}:72)\nexec_code ({cleanup.TOOL}:227)\n'
                  '_kmeans_single_lloyd (/opt/pyvision-rl-venv/lib/python3.10/site-packages/'
                  'sklearn/cluster/_kmeans.py:700)')
        self.assertTrue(cleanup.blocked_tool(kmeans))
        self.assertFalse(cleanup.blocked_tool(kmeans.replace('_worker_loop', 'unrelated')))

    def test_only_terminate_method_changes(self):
        for name in patch.NAMES:
            path = patch.DIRECTORY / name
            current = path.read_text()
            before = ((patch.ROOT / 'references/tool-cleanup-original' / name).read_text()
                      if patch.REPLACEMENT in current else current)
            after = before.replace(patch.ORIGINAL, patch.REPLACEMENT)
            self.assertNotEqual(before, after)
            def without_terminate(text):
                tree = ast.parse(text)
                worker = next(node for node in tree.body
                              if isinstance(node, ast.ClassDef) and node.name == 'PersistentWorker')
                worker.body = [node for node in worker.body
                               if not isinstance(node, ast.FunctionDef) or node.name != 'terminate']
                return ast.dump(tree)
            self.assertEqual(without_terminate(before), without_terminate(after))
            self.assertIn(current, (before, after))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CleanupTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  scope='Real CPU subprocesses: graceful exit, already dead, SIGTERM-ignoring worker; '
                        'negative eligibility/stack gates and AST equivalence outside terminate(). '
                        'No training process, model, or benchmark inference used.')
    (patch.ROOT / 'manifests/tool-cleanup-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
