"""Group-total weighted sampling and explicit expected/actual exposure."""
from collections import Counter
import math
import torch
from torch.utils.data import WeightedRandomSampler


def build_sampler(rows, group_weights, *, generator, sample_ratio=None, draws_per_epoch=None):
    if not rows or not isinstance(group_weights, dict) or not group_weights or not isinstance(generator, torch.Generator):
        raise ValueError('Rows, group weights and explicit generator required')
    if sample_ratio is not None and draws_per_epoch is not None:
        raise ValueError('Specify ratio or draws, not both')
    if draws_per_epoch is None:
        ratio = 1.0 if sample_ratio is None else sample_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('Invalid sample ratio')
        draws = math.floor(len(rows) * ratio)
    else:
        if type(draws_per_epoch) is not int:
            raise ValueError('draws_per_epoch must be an integer')
        draws = draws_per_epoch
    if draws < 1:
        raise ValueError('At least one draw required')
    counts = Counter(r['group'] for r in rows)
    if not set(counts) <= group_weights.keys():
        raise ValueError('Missing group weight')
    for group, value in group_weights.items():
        if not isinstance(group, str) or not group or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid group weight')
    denominator = sum(group_weights[g] for g in counts)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError('No positive nonempty group weight')
    probabilities = [group_weights[r['group']] / counts[r['group']] / denominator for r in rows]
    root_probability = Counter()
    for row, probability in zip(rows, probabilities):
        # A multi-root sample contributes its full probability to each root: root
        # exposures are overlapping events and need not sum to one.
        for root in set(row['lineage_roots']):
            root_probability[root] += probability
    report = {'balance_unit': 'sample', 'replacement': True, 'draws_per_epoch': draws,
              'empty_weighted_groups': sorted(g for g, w in group_weights.items() if w > 0 and not counts[g]),
              'groups': {g: {'samples': counts[g], 'probability': group_weights[g] / denominator if counts[g] else 0.0,
                              'expected_draws': draws * group_weights[g] / denominator if counts[g] else 0.0} for g in group_weights},
              'samples': {r['sample_id']: {'probability': p, 'expected_draws': p * draws} for r, p in zip(rows, probabilities)},
              'roots': {r: {'probability': p, 'expected_draws': p * draws} for r, p in root_probability.items()},
              'root_semantics': 'A multi-root sample counts fully toward each root; root probabilities may sum above one.'}
    return WeightedRandomSampler(probabilities, draws, replacement=True, generator=generator), report


def exposure_report(rows, indices):
    samples, groups, roots = Counter(), Counter(), Counter()
    frames = 0
    sequence = []
    for index in indices:
        if type(index) is not int or index < 0 or index >= len(rows):
            raise ValueError('Invalid sampled index')
        row = rows[index]
        sequence.append(row['sample_id']); samples[row['sample_id']] += 1; groups[row['group']] += 1
        roots.update(set(row['lineage_roots'])); frames += row['num_frames']
    return {'sequence': sequence, 'draws': len(sequence), 'samples': dict(samples), 'groups': dict(groups),
            'roots': dict(roots), 'unique_samples': len(samples), 'unique_roots': len(roots), 'effective_frames': frames}
