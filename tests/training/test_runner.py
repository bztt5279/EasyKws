"""Tiny CPU training, full-epoch recovery, and restricted serialization."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_data_sampling import feature_bundle
from easykws.training import runner
from easykws.training.checkpoint import restricted_load, migrate_checkpoint, TrainingError


def config():
    return {'model': {'hidden_dim': 8, 'num_layers': 1, 'kernel_size': 3},
            'training': {'num_epochs': 2, 'batch_size': 2},
            'sampler': {'group_weights': {'positive': 1, 'negative': 1, 'empty': 4}, 'draws_per_epoch': 5}}


def saved(output, status):
    return restricted_load(output / status['last_checkpoint'], status['last_checkpoint_sha256'])


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def test_epoch_resume_exact_and_real_updates(feature_bundle, tmp_path):
    continuous, stopped = tmp_path / 'continuous', tmp_path / 'stopped'
    a = runner.run(feature_bundle, config(), continuous)
    first = runner.run(feature_bundle, config(), stopped, stop_after_epoch=1)
    assert first['state'] == 'interrupted' and first['global_step'] == 3
    b = runner.resume(stopped, feature_bundle)
    assert a['state'] == b['state'] == 'completed'
    assert a['history'] == b['history']
    assert a['global_step'] == 6 and len(a['candidates']) == 2
    assert a['selected_checkpoint'] is None and a['best_val_loss_checkpoint']
    assert a['purpose'] == b['purpose'] == 'synthetic-test'
    ca, cb = saved(continuous, a), saved(stopped, b)
    for key in ('model', 'optimizer', 'scheduler', 'rng', 'identity'):
        equal(ca[key], cb[key])
    initial = restricted_load(continuous / 'initial.pt')
    assert not torch.equal(initial['model']['classifier.weight'], ca['model']['classifier.weight'])
    logs = lambda path: [json.loads(s) for p in sorted(path.glob('attempt_*.jsonl')) for s in p.read_text().splitlines()]
    batches_a = [r for r in logs(continuous) if r['event'] == 'batch']
    batches_b = [r for r in logs(stopped) if r['event'] == 'batch']
    assert batches_a == batches_b  # Includes actual masked feature hashes and LR.
    for history in a['history']:
        assert history['draws'] == 5 and history['batches'] == 3
        assert history['actual_frames'] == history['exposure']['effective_frames']
    assert a['expected_exposure']['empty_weighted_groups'] == ['empty']


def test_mid_epoch_failure_replays_from_last_complete(feature_bundle, tmp_path, monkeypatch):
    continuous, interrupted = tmp_path / 'baseline', tmp_path / 'fault'
    a = runner.run(feature_bundle, config(), continuous)
    original = runner.max_pooling_loss
    calls = 0
    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:  # One optimizer step into epoch two, after epoch-one validation.
            raise RuntimeError('injected failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'max_pooling_loss', fail_once)
    with pytest.raises(RuntimeError, match='injected'):
        runner.run(feature_bundle, config(), interrupted)
    state = json.loads((interrupted / 'status.json').read_text())
    assert state['state'] == 'failed' and state['last_complete_epoch'] == 1 and state['global_step'] == 3
    assert state['attempts'][0]['attempted_global_step'] == 4
    oldlog = (interrupted / 'attempt_0001.jsonl').read_bytes()
    monkeypatch.setattr(runner, 'max_pooling_loss', original)
    b = runner.resume(interrupted, feature_bundle)
    assert oldlog == (interrupted / 'attempt_0001.jsonl').read_bytes()
    assert b['history'] == a['history'] and len(b['attempts']) == 2
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        equal(saved(continuous, a)[key], saved(interrupted, b)[key])


def test_failure_before_first_epoch_can_restore_initial(feature_bundle, tmp_path, monkeypatch):
    output = tmp_path / 'initial_recovery'
    original = runner.max_pooling_loss
    def fail(*args, **kwargs): raise RuntimeError('first epoch failure')
    monkeypatch.setattr(runner, 'max_pooling_loss', fail)
    with pytest.raises(RuntimeError): runner.run(feature_bundle, config(), output)
    state = json.loads((output / 'status.json').read_text())
    assert state['last_checkpoint'] == 'initial.pt' and state['global_step'] == 0
    monkeypatch.setattr(runner, 'max_pooling_loss', original)
    assert runner.resume(output, feature_bundle)['global_step'] == 6


def test_resume_rejects_config_code_and_partial_checkpoint(feature_bundle, tmp_path, monkeypatch):
    output = tmp_path / 'protected'
    state = runner.run(feature_bundle, config(), output, stop_after_epoch=1)
    original_status = (output / 'status.json').read_bytes()
    with pytest.raises(ValueError, match='empty'): runner.run(feature_bundle, config(), output)
    changed = config(); changed['optimizer'] = {'lr': .02}
    with pytest.raises(ValueError, match='identity'): runner.resume(output, feature_bundle, changed)
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'code_identity', lambda: 'different-code')
        with pytest.raises(ValueError, match='identity'): runner.resume(output, feature_bundle)
    assert (output / 'status.json').read_bytes() == original_status
    feature = next((feature_bundle / 'features').glob('*.npy'))
    original_feature = feature.read_bytes()
    feature.write_bytes(original_feature + b'changed')
    with pytest.raises(ValueError, match='hash mismatch'): runner.resume(output, feature_bundle)
    feature.write_bytes(original_feature)
    assert (output / 'status.json').read_bytes() == original_status
    (output / state['last_checkpoint']).write_bytes(b'partial')
    with pytest.raises(ValueError, match='hash mismatch'): runner.resume(output, feature_bundle)
    partial = tmp_path / 'partial.tmp'; partial.write_bytes(b'partial')
    with pytest.raises(ValueError, match='partial'): restricted_load(partial)


def test_explicit_trusted_legacy_migration_only(tmp_path):
    legacy, output = tmp_path / 'own_legacy.pt', tmp_path / 'converted.pt'
    torch.save({'array': np.arange(4), 'model': {'weight': torch.ones(2)}}, legacy)
    with pytest.raises(TrainingError): restricted_load(legacy)
    with pytest.raises(TrainingError, match='trust'): migrate_checkpoint(legacy, output)
    result = migrate_checkpoint(legacy, output, trust_legacy_input=True)
    value = restricted_load(output)
    assert result['resumable'] is False and value['resumable'] is False
    assert torch.equal(value['legacy_state']['array'], torch.arange(4))


@pytest.mark.parametrize('bad_hash', ['missing', None, '', 'a' * 63, 'g' * 64, 42])
def test_resume_requires_valid_checkpoint_hash(feature_bundle, tmp_path, bad_hash):
    output = tmp_path / 'missing_hash'
    runner.run(feature_bundle, config(), output, stop_after_epoch=1)
    status_file = output / 'status.json'
    status = json.loads(status_file.read_text())
    if bad_hash == 'missing':
        del status['last_checkpoint_sha256']
    else:
        status['last_checkpoint_sha256'] = bad_hash
    status_file.write_text(json.dumps(status), encoding='utf-8')
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(TrainingError, match='last_checkpoint_sha256'):
        runner.resume(output, feature_bundle)
    assert before == {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}


@pytest.mark.parametrize('recovering', [False, True])
def test_log_creation_failure_records_failed_and_remains_resumable(feature_bundle, tmp_path, monkeypatch, recovering):
    output = tmp_path / 'log_failure'
    if recovering:
        runner.run(feature_bundle, config(), output, stop_after_epoch=1)
    previous = (output / 'attempt_0001.jsonl').read_bytes() if recovering else None
    original_open = Path.open
    def deny_log(path, *args, **kwargs):
        if path.name.startswith('attempt_') and args and args[0] == 'x':
            raise PermissionError('injected log creation failure')
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', deny_log)
        with pytest.raises(PermissionError, match='injected'):
            if recovering:
                runner.resume(output, feature_bundle)
            else:
                runner.run(feature_bundle, config(), output)
    state = json.loads((output / 'status.json').read_text())
    assert state['state'] == state['attempts'][-1]['state'] == 'failed'
    assert state['attempts'][-1]['error'] == 'PermissionError'
    assert 'injected log creation failure' in state['attempts'][-1]['reason']
    assert state['last_complete_epoch'] == (1 if recovering else 0)
    if previous is not None:
        assert (output / 'attempt_0001.jsonl').read_bytes() == previous
    assert runner.resume(output, feature_bundle)['state'] == 'completed'


def test_config_rejects_unsupported_modes_before_output():
    for change in ({'device': 'cuda'}, {'training': {'num_workers': 2}},
                   {'sampler': {'draws_per_epoch': 2, 'sample_ratio': 1}},
                   {'spec_augment': {'max_t': 0}}):
        cfg = config()
        for key, value in change.items():
            if isinstance(value, dict): cfg.setdefault(key, {}).update(value)
            else: cfg[key] = value
        with pytest.raises(ValueError): runner.configure(cfg)
