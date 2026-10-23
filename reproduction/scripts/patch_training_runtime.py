"""Parallel independent tool cleanup and optionally save the first full step."""
from download_preparation_queue import ROOT, digest, write

PATCHES = [
    ('parallel_env', 'verl_agents/verl/workers/agent/parallel_env.py',
'''        # 正确关闭每个工具
        for tool in self.tools:
            if tool is not None and hasattr(tool, 'close'):
                tool.close()
            elif tool is not None and hasattr(tool, '__del__'):
                tool.__del__()
''',
'''        # Local reproduction: independent tool processes retain their original
        # per-process shutdown rules, while cleanup runs concurrently.
        def close_one(tool):
            if tool is not None and hasattr(tool, 'close'):
                tool.close()
            elif tool is not None and hasattr(tool, '__del__'):
                tool.__del__()
        if self.tools:
            with ThreadPoolExecutor(max_workers=min(32, len(self.tools))) as pool:
                list(pool.map(close_one, self.tools))
'''),
    ('ray_trainer', 'verl_agents/verl/trainer/ppo/ray_trainer.py',
'''                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
''',
'''                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                        or (self.global_steps == 1 and self.config.trainer.get("save_first_step", False))
'''),
]


def main():
    records = []
    for name, relative, old, new in PATCHES:
        target = ROOT.parent / relative
        backup = ROOT / 'references' / (name + '.original.py')
        source = target.read_text()
        if new not in source:
            if source.count(old) != 1:
                raise RuntimeError('Upstream runtime source changed: ' + relative)
            if not backup.exists():
                backup.write_text(source)
            target.write_text(source.replace(old, new))
        records.append(dict(target=str(target), original_sha256=digest(backup), patched_sha256=digest(target)))
    write(ROOT / 'manifests/training-runtime-patches.json', dict(
        patches=records, rl_rewards_filters_advantages_modified=False,
        changes=['Concurrent cleanup of independent tool processes', 'Optional step-1 checkpoint'],
        validation='manifests/tool-cleanup-validation.json'))
    print('Applied independent tool cleanup and first-step checkpoint options')


if __name__ == '__main__':
    main()
