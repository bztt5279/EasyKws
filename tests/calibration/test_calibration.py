import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from test_data_sampling import feature_bundle
from easykws.calibration import calibrate, evaluate, configure
from easykws.calibration.metrics import event_metrics, sample_metrics, choose
from easykws.calibration.sessions import load_sessions
from easykws.training import runner
from easykws.training.data import load_bundle


def session_manifest(tmp_path):
    path = tmp_path / 'session.wav'
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(np.zeros(16000, dtype='<i2').tobytes())
    session = {'id': 'one', 'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'duration_samples': 16000,
               'recording_kind': 'synthetic-concatenation', 'speaker_ids': [],
               'positive_windows': [{'id': 'wake', 'start_sec': 0, 'end_sec': .5}],
               'negative_intervals': [{'start_sec': .5, 'end_sec': 1}]}
    value = {'schema': 'easykws-calibration-sessions/v1', 'purpose': 'synthetic-test', 'split': 'calibration',
             'sample_rate': 16000, 'channels': 1, 'clock': 'samples', 'boundary_policy': 'reset-each-session',
             'gap_policy': 'split-session', 'provenance': {'source_id': 'generated', 'description': 'synthetic silence'}, 'sessions': [session]}
    manifest = tmp_path / 'sessions.json'
    manifest.write_text(json.dumps(value), encoding='utf-8')
    return manifest, value


def config(candidate):
    return {'candidates': [candidate], 'threshold_grid': [0, 1], 'min_recall': 1, 'max_fah': 0,
            'trigger': {'confirm_frames': 1, 'warmup_frames': 0}, 'chunk_samples': 317}


def test_core_real_scoring_binding_and_cli(feature_bundle, tmp_path):
    torch.set_num_threads(1)
    run = tmp_path / 'run'
    status = runner.run(feature_bundle, {'model': {'hidden_dim': 4, 'num_layers': 1, 'kernel_size': 3}, 'training': {'num_epochs': 1, 'batch_size': 2}, 'sampler': {'group_weights': {'positive': 1, 'negative': 1}}}, run)
    candidate = status['candidates'][0]['path']
    before = (run / 'status.json').read_bytes()
    audio_root = tmp_path / 'sessions'; audio_root.mkdir()
    manifest, _ = session_manifest(audio_root)
    result = calibrate(run, feature_bundle, manifest, tmp_path / 'cal', config(candidate))
    assert result['feasible'] and result['selected_threshold'] == 0
    assert result['records'][0]['metrics']['recall'] == 1
    assert result['records'][1]['metrics']['recall'] == 0
    assert result['records'][0]['binding']['trigger_config']['rules']['comparison'] == 'ge'
    assert (tmp_path / 'cal' / 'selected_checkpoint.json').exists()
    ready = json.loads((tmp_path / 'cal' / 'READY.json').read_text())
    for name, sha in ready['files'].items():
        assert hashlib.sha256((tmp_path / 'cal' / name).read_bytes()).hexdigest() == sha
    impossible = config(candidate); impossible['threshold_grid'] = [1]
    assert not calibrate(run, feature_bundle, manifest, tmp_path / 'impossible', impossible)['feasible']
    assert not (tmp_path / 'impossible' / 'selected_checkpoint.json').exists()
    segmented = evaluate(run, feature_bundle, tmp_path / 'eval', [candidate], [0, 1])
    assert segmented['split'] == 'val'
    assert segmented['records'][0]['metrics'][0]['auc'] is not None
    assert 'protocol_stream_fah' not in segmented['records'][0]['metrics'][0]
    assert before == (run / 'status.json').read_bytes()
    with pytest.raises(ValueError): evaluate(run, feature_bundle, tmp_path / 'eval')
    cfg = tmp_path / 'config.json'; cfg.write_text(json.dumps(config(candidate)))
    command = [sys.executable, '-B', '-c', 'from easykws.training.cli import main; raise SystemExit(main())', 'calibrate', '--run', str(run), '--features', str(feature_bundle), '--manifest', str(manifest), '--config', str(cfg), '--output', str(tmp_path / 'cli')]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / 'cli' / 'READY.json').exists()
    with pytest.raises(ValueError, match='protected'): evaluate(run, feature_bundle, run / 'forbidden')


def test_config_and_metric_boundaries():
    for invalid in (None, {}, {'candidates': ['x'], 'threshold_grid': [True], 'min_recall': 1}, {'candidates': ['x'], 'threshold_grid': [.5], 'max_fah': -1}):
        with pytest.raises(ValueError): configure(invalid)
    assert sample_metrics([0, -1], [.5, .5])['auc'] == .5
    assert choose([{'candidate': 'a', 'threshold': .8, 'metrics': {'recall': 1, 'protocol_stream_fah': 1}}], 1, 0) is None


def test_known_pcm_repack_rejected_with_empty_speakers(feature_bundle, tmp_path):
    bundle = load_bundle(feature_bundle)
    manifest, value = session_manifest(tmp_path)
    source = tmp_path / 'audio' / '0.wav'
    raw = source.read_bytes()
    # Same PCM payload with a different legal WAV wrapper (unknown trailing chunk).
    repack = raw + b'JUNK' + (4).to_bytes(4, 'little') + b'abcd'
    repack = repack[:4] + (len(repack) - 8).to_bytes(4, 'little') + repack[8:]
    (tmp_path / 'session.wav').write_bytes(repack)
    session = value['sessions'][0]
    session.update(sha256=hashlib.sha256(repack).hexdigest(), duration_samples=3200, positive_windows=[], negative_intervals=[{'start_sec': 0, 'end_sec': .2}])
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='PCM'): load_sessions(manifest, bundle)
    value['split'] = 'test'; manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='train/test'): load_sessions(manifest, bundle)

def test_session_protocol_rejections_and_failed_output(feature_bundle, tmp_path):
    bundle = load_bundle(feature_bundle)
    audio_root = tmp_path / 'audio-session'; audio_root.mkdir()
    manifest, value = session_manifest(audio_root)
    value['sessions'][0]['negative_intervals'][0]['start_sec'] = .25
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='overlap'): load_sessions(manifest, bundle)
    value['sessions'][0]['negative_intervals'][0]['start_sec'] = .5
    value['sessions'][0]['sha256'] = '0' * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash'): load_sessions(manifest, bundle)
    output = tmp_path / 'failure'
    with pytest.raises(ValueError): calibrate(tmp_path / 'run', feature_bundle, manifest, output, {})
    assert (output / 'FAILED.json').exists() and not (output / 'READY.json').exists()


def test_missing_denominators_and_tie_rule():
    session = {'id': 'x', 'positive_windows': [], 'negative_intervals': [], 'duration_samples': 16000, 'recording_kind': 'offline-replay'}
    metrics = event_metrics([session], {'x': []})
    assert metrics['recall'] is None and metrics['protocol_stream_fah'] is None
    assert choose([{'candidate': 'x', 'threshold': .5, 'metrics': metrics}], 0, 0) is None
    records = [{'candidate': c, 'threshold': t, 'metrics': {'recall': 1, 'protocol_stream_fah': 0}} for c, t in [('b', .8), ('a', .8), ('a', .5)]]
    assert choose(records, 1, 0) == records[1]
    assert sample_metrics([0, 0, -1, -1], [1, .5, .5, 0])['auc'] == .875
