"""Reap persistent tool workers after the original graceful shutdown interval."""
from download_preparation_queue import ROOT, digest, write

NAMES = [
    'safe_persis_python_exe_tool_w_image_hint.py',
    'safe_persis_python_exe_tool_wo_image_hint.py',
    'safe_persis_python_exe_tool_wo_video_hint.py',
]
DIRECTORY = ROOT.parent / 'verl_agents/verl/workers/agent/envs/agents_x'
ORIGINAL = '''    def terminate(self):
        """终止工作进程"""
        if self.process and self.process.is_alive():
            self.input_queue.put(None)
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
'''
REPLACEMENT = '''    def terminate(self):
        """Reap the tool subprocess, including workers stuck in inherited I/O."""
        process = self.process
        if process is None:
            return
        if process.is_alive():
            self.input_queue.put(None)
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        else:
            process.join(timeout=0)
        if process.is_alive():
            raise RuntimeError("Tool subprocess did not exit after forced shutdown")
        self.process = None
        process.close()
        for channel in (self.input_queue, self.output_queue):
            channel.cancel_join_thread()
            channel.close()
'''


def main():
    records = []
    for name in NAMES:
        target = DIRECTORY / name
        backup = ROOT / 'references/tool-cleanup-original' / name
        source = target.read_text()
        if REPLACEMENT not in source:
            if source.count(ORIGINAL) != 1:
                raise RuntimeError(f'Unexpected tool cleanup source: {target}')
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_text(source)
            elif backup.read_text() != source:
                raise RuntimeError(f'Existing original backup differs: {target}')
            target.write_text(source.replace(ORIGINAL, REPLACEMENT))
        records.append(dict(target=str(target), original_sha256=digest(backup),
                            patched_sha256=digest(target)))
    write(ROOT / 'manifests/tool-cleanup-patch.json', dict(
        files=records, rl_algorithm_modified=False,
        tool_execution_timeout_modified=False,
        purpose='Keep original graceful shutdown, then terminate/join, kill/join and close queues; '
                'no reward, filtering, decoding or trajectory content changes.',
        running_image_actors_reload_patch=False,
        validation='manifests/tool-cleanup-validation.json'))
    print('Applied persistent tool subprocess cleanup patch')


if __name__ == '__main__':
    main()
