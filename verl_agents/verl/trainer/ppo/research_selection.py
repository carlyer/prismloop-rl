"""Task quotas and group-aware selection, with complete candidate accounting."""
from collections import Counter, defaultdict
import math


def source_counts(batch):
    return Counter(str(x) for x in batch.non_tensor_batch['data_source'])


def select_stratified_indices(batch, target, weights):
    if len(batch) < target:
        raise ValueError('Insufficient eligible trajectories')
    sources = batch.non_tensor_batch['data_source']
    groups = defaultdict(list)
    for i, (source, uid) in enumerate(zip(sources, batch.non_tensor_batch['uid'])):
        groups[(str(source), str(uid))].append(i)
    ordered = defaultdict(list)
    stds = batch.non_tensor_batch['sample_level_stds']
    for (source, uid), indices in groups.items():
        ordered[source].append((float(stds[indices[0]]), uid, indices))
    for source in ordered:
        ordered[source].sort(key=lambda x: (-x[0], x[1]))
    positive = {str(k): float(v) for k, v in weights.items() if float(v) > 0}
    if not positive:
        raise ValueError('Source weights must be positive')
    total = sum(positive.values())
    quotas = {k: math.floor(target * v / total) for k, v in positive.items()}
    for k in sorted(positive, key=lambda k: (-(target * positive[k] / total - quotas[k]), k))[:target-sum(quotas.values())]:
        quotas[k] += 1
    selected, used = [], set()
    counts = Counter()
    # Whole surviving sibling sets first, ranked only within their task.
    for source, quota in quotas.items():
        for _, uid, indices in ordered.get(source, []):
            if counts[source] + len(indices) <= quota:
                selected.extend(indices)
                counts[source] += len(indices)
                used.add((source, uid))
    # Fill unused capacity without comparing raw reward variance across tasks.
    remaining = {s: [g for g in gs if (s, g[1]) not in used] for s, gs in ordered.items()}
    partial = 0
    while len(selected) < target:
        available = [s for s, gs in remaining.items() if gs]
        if not available:
            raise RuntimeError('Selection exhausted eligible trajectories')
        source = min(available, key=lambda s: (counts[s] / max(quotas.get(s, 0), 1), s))
        _, _, indices = remaining[source].pop(0)
        room = target - len(selected)
        chosen = indices[:room]
        partial += int(len(chosen) != len(indices))
        selected.extend(chosen)
        counts[source] += len(chosen)
    assert len(selected) == len(set(selected)) == target
    metrics = {'research/partial_groups_at_capacity': partial}
    for source in sorted(set(quotas) | set(counts)):
        metrics[f'research/selected/{source}'] = counts[source]
        metrics[f'research/quota/{source}'] = quotas.get(source, 0)
        metrics[f'research/quota_shortfall/{source}'] = max(0, quotas.get(source, 0)-counts[source])
    return selected, metrics
