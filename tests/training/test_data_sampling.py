import hashlib
import json
import random
import wave

import numpy as np
import pytest
import torch

from easykws.prep import v2
from easykws.preprocess import run
from easykws.training.data import load_bundle, FeatureDataset, collate_features
from easykws.training.augment import SpecAugment
from easykws.training.sampling import build_sampler, exposure_report


@pytest.fixture
def feature_bundle(tmp_path):
    audio = tmp_path / 'audio'; audio.mkdir()
    doc = {'schema': v2.SCHEMA, 'datasets': [{'dataset_id': 'demo', 'source': 'generated', 'version': 'v1', 'license': 'test-only', 'split_policy': 'speaker'}], 'resources': [], 'selection': {'kind': 'full', 'reason': '', 'available_count': 6}, 'rows': []}
    for i in range(6):
        samples = (2500 * np.sin(np.arange(3200 + 160 * i) * (.08 + i * .01))).astype('<i2')
        path = audio / f'{i}.wav'
        with wave.open(str(path), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed')); f.writeframes(samples.tobytes())
        doc['rows'].append({'sample_id': f's{i}', 'dataset_id': 'demo', 'speaker_id': f'p{i // 2}', 'source_recording_id': f'r{i}', 'audio_path': path.name, 'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'label': 0 if i % 2 == 0 else -1, 'group': 'positive' if i % 2 == 0 else 'negative', 'expected_phrase': 'x' if i % 2 == 0 else '', 'review_status': 'pass', 'origin': 'test-fixture', 'purpose': 'synthetic-test', 'parent_sample_ids': [], 'resource_ids': [], 'requested_split': ('train', 'val', 'test')[i // 2], 'generation': None})
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps(doc), encoding='utf-8')
    prepared = tmp_path / 'prep'; output = tmp_path / 'features'
    v2.prepare(manifest, {'demo': audio}, prepared)
    run(prepared, None, output, path_map={'demo': audio})
    return output


def test_full_features_masks_and_purpose(feature_bundle):
    bundle = load_bundle(feature_bundle)
    assert bundle['purpose'] == 'synthetic-test' and 'synthetic-test' in bundle['data_flags']
    raw = FeatureDataset(bundle, 'train')[0][0]
    cache = (feature_bundle / bundle['rows']['train'][0]['feature_path']).read_bytes()
    random.seed(123); masked = FeatureDataset(bundle, 'train', SpecAugment())[0][0]
    random.seed(123); assert torch.equal(masked, FeatureDataset(bundle, 'train', SpecAugment())[0][0])
    assert not torch.equal(masked, raw)
    val = FeatureDataset(bundle, 'val', SpecAugment())[0][0]
    assert torch.equal(val, FeatureDataset(bundle, 'val')[0][0])
    assert cache == (feature_bundle / bundle['rows']['train'][0]['feature_path']).read_bytes()
    x, lengths, labels, ids = collate_features([FeatureDataset(bundle, 'train')[i] for i in range(2)])
    assert lengths.tolist() == [r['num_frames'] for r in bundle['rows']['train']]
    assert x.shape[1] == max(lengths) and len(labels) == len(ids) == 2


def test_sampling_probabilities_and_actual_exposure():
    rows = [{'sample_id': 'a', 'group': 'g', 'lineage_roots': ['r'], 'num_frames': 4}, {'sample_id': 'b', 'group': 'g', 'lineage_roots': ['s'], 'num_frames': 7}, {'sample_id': 'c', 'group': 'h', 'lineage_roots': ['z'], 'num_frames': 10}]
    sampler, expected = build_sampler(rows, {'g': 2, 'h': 1, 'empty': 9}, generator=torch.Generator().manual_seed(8), draws_per_epoch=11)
    assert expected['empty_weighted_groups'] == ['empty'] and expected['groups']['g']['probability'] == pytest.approx(2/3)
    indices = list(sampler)
    sampler2, _ = build_sampler(rows, {'g': 2, 'h': 1, 'empty': 9}, generator=torch.Generator().manual_seed(8), draws_per_epoch=11)
    assert indices == list(sampler2)
    report = exposure_report(rows, indices)
    assert report['draws'] == 11 and report['effective_frames'] == sum(rows[i]['num_frames'] for i in indices)
    extended = rows + [{'sample_id': 'd', 'group': 'g', 'lineage_roots': ['r'], 'num_frames': 4}]
    _, later = build_sampler(extended, {'g': 2, 'h': 1}, generator=torch.Generator().manual_seed(8))
    assert later['roots']['r']['probability'] == pytest.approx(4/9)
    assert expected['roots']['r']['probability'] == pytest.approx(1/3)
    with pytest.raises(ValueError): build_sampler(rows, {'g': 1, 'h': 1}, generator=torch.Generator(), sample_ratio=0.01)
    with pytest.raises(ValueError): build_sampler(rows, {'g': 1}, generator=torch.Generator())
    with pytest.raises(ValueError): build_sampler(rows, {'g': 1, 'h': 1}, generator=torch.Generator(), sample_ratio=1, draws_per_epoch=1)


def test_corrupted_feature_rejected(feature_bundle):
    b = load_bundle(feature_bundle)
    path = feature_bundle / b['rows']['train'][0]['feature_path']
    path.write_bytes(path.read_bytes() + b'bad')
    with pytest.raises(ValueError, match='hash mismatch'): load_bundle(feature_bundle)
    with pytest.raises(ValueError, match='changed'): FeatureDataset(b, 'train')[0]


def test_invalid_mask_parameters():
    with pytest.raises(ValueError): SpecAugment(max_t=0)
    with pytest.raises(ValueError): SpecAugment(num_f_mask=-1)


def test_legacy_v1_preprocessed_bundle_remains_readable(dataset, tmp_path):
    from easykws.prep import prepare
    manifest, audio, _ = dataset
    prepared, output = tmp_path / 'v1_prepared', tmp_path / 'v1_features'
    prepare(manifest, audio, prepared)
    run(prepared, audio, output)
    bundle = load_bundle(output)
    assert bundle['purpose'] == 'synthetic-test'
    assert len(bundle['rows']['train']) == 6
    assert FeatureDataset(bundle, 'train')[0][0].shape[1] == 40

@pytest.mark.parametrize('kind', ['nan', 'dtype', 'shape'])
def test_rehashed_invalid_feature_is_rejected(feature_bundle, kind):
    import io
    ready_path = feature_bundle / 'READY.json'
    ready = json.loads(ready_path.read_text(encoding='utf-8'))
    index_path = feature_bundle / 'train.jsonl'
    rows = [json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines()]
    feature_path = feature_bundle / rows[0]['feature_path']
    x = np.load(feature_path, allow_pickle=False)
    if kind == 'nan': x[0, 0] = np.nan
    elif kind == 'dtype': x = x.astype(np.float64)
    else: x = x[:, :39]
    buffer = io.BytesIO(); np.save(buffer, x, allow_pickle=False); raw = buffer.getvalue()
    feature_path.write_bytes(raw)
    rows[0]['feature_sha256'] = hashlib.sha256(raw).hexdigest()
    ready['files'][rows[0]['feature_path']] = rows[0]['feature_sha256']
    index_path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    ready['files']['train.jsonl'] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    ready_path.write_text(json.dumps(ready), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid feature'): load_bundle(feature_bundle)


def test_purpose_downgrade_even_when_ready_and_plan_agree(feature_bundle):
    ready_path, plan_path = feature_bundle / 'READY.json', feature_bundle / 'plan.json'
    ready, plan = (json.loads(p.read_text(encoding='utf-8')) for p in (ready_path, plan_path))
    ready['purpose'] = plan['purpose'] = 'recorded-data'
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    ready['files']['plan.json'] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    ready_path.write_text(json.dumps(ready), encoding='utf-8')
    with pytest.raises(ValueError, match='downgraded'): load_bundle(feature_bundle)
