"""Used augmentation resources remain excluded across WAV repackaging."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocess'))
from test_augmentation_pipeline import bundle, config
from easykws.preprocess import run
from easykws.training.data import load_bundle
from easykws.calibration.sessions import load_sessions
from easykws.prep.core import _audio_info


@pytest.mark.parametrize('resource_split', ['train', 'external'])
@pytest.mark.parametrize('kind,resource_id', [('noise', 'noise_b'), ('rir', 'impulse')])
def test_used_resources_reject_original_and_repacked_pcm(bundle, tmp_path, resource_split, kind, resource_id):
    document, audio, prepare = bundle
    next(r for r in document['resources'] if r['resource_id'] == resource_id)['split'] = resource_split
    prepared, roots = prepare()
    cfg = config([kind])
    cfg['augmentation']['noise']['resource_ids'] = [resource_id] if kind == 'noise' else []
    cfg['augmentation']['rir']['resource_ids'] = [resource_id] if kind == 'rir' else []
    features = tmp_path / 'features'
    run(prepared, None, features, cfg, path_map=roots)
    validated = load_bundle(features)
    used = [r for row in validated['rows']['train'] for r in row.get('augmentation', {}).get('resources', [])]
    assert len(used) == 2 and all(r['resource_id'] == resource_id and r['audio_pcm_sha256'] for r in used)
    session_root = tmp_path / 'sessions'
    session_root.mkdir()
    raw = (audio / (resource_id + '.wav')).read_bytes()
    repacked = raw + b'JUNK' + (4).to_bytes(4, 'little') + b'abcd'
    repacked = repacked[:4] + (len(repacked) - 8).to_bytes(4, 'little') + repacked[8:]
    assert hashlib.sha256(raw).digest() != hashlib.sha256(repacked).digest()

    def manifest_for(data):
        path = session_root / 'resource.wav'
        path.write_bytes(data)
        info = _audio_info(path)
        value = {'schema': 'easykws-calibration-sessions/v1', 'purpose': 'synthetic-test',
                 'split': 'calibration', 'sample_rate': 16000, 'channels': 1, 'clock': 'samples',
                 'boundary_policy': 'reset-each-session', 'gap_policy': 'split-session',
                 'provenance': {'source_id': 'generated', 'description': 'Resource isolation test'},
                 'sessions': [{'id': 'resource', 'path': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                               'duration_samples': info['num_samples'], 'recording_kind': 'offline-replay',
                               'speaker_ids': [], 'positive_windows': [],
                               'negative_intervals': [{'start_sec': 0, 'end_sec': info['num_samples'] / 16000}]}]}
        manifest = session_root / 'sessions.json'
        manifest.write_text(json.dumps(value), encoding='utf-8')
        return manifest, info

    # Check the wrapper counterexample first so the failing baseline proves PCM coverage.
    for data in (repacked, raw):
        manifest, info = manifest_for(data)
        assert info['audio_pcm_sha256'] == used[0]['audio_pcm_sha256']
        with pytest.raises(ValueError, match='duplicates'):
            load_sessions(manifest, validated)

    # Merely being listed in the external resource catalog is not training use.
    manifest, _ = manifest_for((audio / 'noise_a.wav').read_bytes())
    assert load_sessions(manifest, validated)[0]['sessions'][0]['id'] == 'resource'
