"""Validated raw feature bundles and full-length batching."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocess.frontend import FBANK


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Invalid bundle path')
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ('..', '.') for x in p.parts) or ':' in name:
        raise ValueError('Unsafe bundle path')
    path = root.joinpath(*p.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Bundle path escapes root')
    try:
        return path.read_bytes()
    except OSError as e:
        raise ValueError('Cannot read bundle file: ' + name) from e


def _json(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate JSON key')
            out[k] = v
        return out
    try:
        return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    except (ValueError, UnicodeError) as e:
        raise ValueError('Invalid bundle JSON') from e


def load_bundle(path):
    root = Path(path).resolve()
    if (root / 'FAILED.json').exists():
        raise ValueError('Bundle has failure marker')
    ready_raw = _read(root, 'READY.json')
    ready = _json(ready_raw)
    if ready.get('schema') not in ('easykws-preprocessed/v1', 'easykws-preprocessed/v2') or ready.get('feature_format') != 'numpy-float32-raw-v1':
        raise ValueError('Unsupported feature bundle')
    files = ready.get('files')
    if not isinstance(files, dict):
        raise ValueError('Missing file inventory')
    required = {'train.jsonl', 'val.jsonl', 'test.jsonl', 'cmvn.json', 'config.json', 'excluded.jsonl', 'prep_config.json', 'source_ready.json', 'plan.json'}
    if ready['schema'].endswith('/v2'):
        required.add('source_manifest.json')
    if not required <= files.keys():
        raise ValueError('Incomplete file inventory')
    payloads = {}
    for name, expected in files.items():
        raw = _read(root, name)
        if not isinstance(expected, str) or _digest(raw) != expected:
            raise ValueError('File hash mismatch: ' + name)
        # Do not retain the entire feature corpus in RAM during verification.
        if not name.endswith('.npy'):
            payloads[name] = raw
    plan, source = _json(payloads['plan.json']), _json(payloads['source_ready.json'])
    summary = {k: v for k, v in plan.items() if k not in {'rows', 'config', 'schema'}}
    if any(ready.get(k) != v for k, v in summary.items()) or plan.get('config') != _json(payloads['config.json']):
        raise ValueError('Plan and READY disagree')
    if ready.get('config_sha256') != files['config.json'] or ready.get('prepared_ready_sha256') != _digest(payloads['source_ready.json']):
        raise ValueError('Source or config identity mismatch')
    if ready.get('prepared_files') != source.get('files') or files['excluded.jsonl'] != source['files']['excluded.jsonl'] or files['prep_config.json'] != source['files']['config.json']:
        raise ValueError('Source inventory mismatch')
    if ready['schema'].endswith('/v2') and files['source_manifest.json'] != source.get('manifest_sha256'):
        raise ValueError('Source manifest identity mismatch')
    frontend = ready.get('frontend')
    if frontend != {'implementation': 'torchaudio.compliance.kaldi.fbank', 'parameters': FBANK, 'waveform_scale': 32768.0, 'cmvn_applied': False}:
        raise ValueError('Unsupported frontend')
    rows, ids, used = {}, set(), set()
    total = np.zeros(40, dtype=np.float64)
    squares = np.zeros(40, dtype=np.float64)
    count = 0
    for split in ('train', 'val', 'test'):
        rows[split] = [_json(line) for line in payloads[split + '.jsonl'].splitlines() if line.strip()]
        for row in rows[split]:
            sid, name = row.get('sample_id'), row.get('feature_path')
            if not isinstance(sid, str) or not sid or sid in ids or row.get('split') != split or row.get('included') is not True or type(row.get('label')) is not int or row['label'] not in (-1, 0) or not isinstance(row.get('group'), str) or not row['group']:
                raise ValueError('Invalid sample identity, label, group or split')
            ids.add(sid)
            if name in used or name not in files or not isinstance(name, str) or not name.startswith('features/') or not name.endswith('.npy') or row.get('feature_sha256') != files[name]:
                raise ValueError('Invalid feature identity')
            used.add(name)
            try:
                feature_raw = _read(root, name)
                if _digest(feature_raw) != files[name]:
                    raise ValueError('Feature changed during verification')
                arr = np.load(io.BytesIO(feature_raw), allow_pickle=False)
                if not isinstance(arr, np.ndarray) or arr.dtype != np.float32 or arr.ndim != 2 or arr.shape[1] != 40 or arr.shape[0] < 1 or not np.isfinite(arr).all() or row.get('num_frames') != len(arr) or row.get('feature_dim') != 40:
                    raise ValueError('Invalid feature shape or values')
            except (OSError, EOFError, ValueError) as e:
                raise ValueError('Invalid feature data') from e
            if split == 'train':
                f = arr.astype(np.float64)
                count += len(f); total += f.sum(0); squares += (f * f).sum(0)
        if ready.get('splits', {}).get(split) != {'samples': len(rows[split]), 'frames': sum(r['num_frames'] for r in rows[split])}:
            raise ValueError('Split statistics mismatch')
    if set(files) != required | used:
        raise ValueError('Unexpected or unreferenced bundle file')
    planned = plan.get('rows')
    reconstructed = [{k: v for k, v in r.items() if k not in ('feature_path', 'feature_sha256', 'num_frames', 'feature_dim')} | {'expected_frames': r['num_frames']} for s in rows.values() for r in s]
    if not isinstance(planned, list) or sorted(planned, key=lambda r: r['sample_id']) != sorted(reconstructed, key=lambda r: r['sample_id']):
        raise ValueError('Feature index and plan mismatch')
    cmvn = _json(payloads['cmvn.json'])
    try:
        mean, var, istd = (np.asarray(cmvn[k], dtype=np.float64) for k in ('mean', 'var', 'istd'))
        expected_var = np.maximum(squares / count - (total / count) ** 2, 1e-10)
        if not count or cmvn['frame_num'] != count or ready.get('train_frames') != count or cmvn['fit_split'] != 'train' or ready.get('cmvn_fit_split') != 'train' or any(a.shape != (40,) or not np.isfinite(a).all() for a in (mean, var, istd)) or not np.allclose(mean, total / count, rtol=1e-7, atol=1e-7) or not np.allclose(var, expected_var, rtol=1e-6, atol=1e-9) or not np.allclose(istd, var ** -.5, rtol=1e-7):
            raise ValueError('CMVN mismatch')
    except (KeyError, TypeError, ZeroDivisionError) as e:
        raise ValueError('Invalid CMVN') from e
    flags = ready.get('data_flags', [])
    if not isinstance(flags, list) or source.get('purpose') == 'synthetic-test' and ready.get('purpose') != 'synthetic-test' or not set(source.get('data_flags', [])) <= set(flags):
        raise ValueError('Purpose or flags downgraded')
    allrows = [r for values in rows.values() for r in values]
    resources = ready.get('augmentation', {}).get('resources', {})
    for row in allrows:
        augmentation = row.get('augmentation', {})
        used_ids = {op['resource_id'] for op in augmentation.get('operations', []) if 'resource_id' in op}
        actual = augmentation.get('resources', [])
        if not isinstance(actual, list) or {r.get('resource_id') for r in actual} != used_ids or any(resources.get(r.get('resource_id')) != r for r in actual):
            raise ValueError('Augmentation resource identity mismatch')
    if any(r.get('effective_purpose') == 'synthetic-test' or r.get('origin') == 'synthetic' or any(x.get('effective_purpose') == 'synthetic-test' for x in r.get('augmentation', {}).get('resources', [])) for r in allrows):
        if ready.get('purpose') != 'synthetic-test' or ready['schema'].endswith('/v2') and 'synthetic-test' not in flags:
            raise ValueError('Synthetic purpose downgraded')
    ancestry = {r['sample_id']: r for r in (_json(line) for line in payloads['excluded.jsonl'].splitlines() if line.strip())}
    if 'source_manifest.json' in payloads:
        from ..prep import v2
        from ..preprocess.core import AUDIO_FIELDS
        manifest = _json(payloads['source_manifest.json'])
        ancestry = {r['sample_id']: r for r in manifest['rows']}
        representatives = {}
        for row in allrows:
            source_id = row.get('augmentation', {}).get('source_sample_id', row['sample_id'])
            representatives.setdefault(source_id, row)
        def metadata(row):
            actual = representatives[row['sample_id']]
            return {key: actual[key] for key in AUDIO_FIELDS}
        try:
            rebuilt = v2.evaluate(manifest, _json(payloads['prep_config.json']), metadata, lambda resource: None)
        except Exception as e:
            raise ValueError('Invalid source manifest semantics') from e
        expected = {k: v for k, v in rebuilt.items() if k not in {'config', 'rows'}}
        expected['manifest_sha256'] = files['source_manifest.json']
        if {k: v for k, v in source.items() if k != 'files'} != expected:
            raise ValueError('Source READY semantics mismatch')
        original = {r['sample_id']: r for r in rebuilt['rows']}
        for row in allrows:
            sid = row.get('augmentation', {}).get('source_sample_id', row['sample_id'])
            if sid not in original or not original[sid]['included']:
                raise ValueError('Feature has unknown source')
            src = original[sid]
            for key in ('label', 'group', 'split', 'origin', 'purpose', 'dataset_id', 'source_recording_id'):
                if row.get(key) != src.get(key):
                    raise ValueError('Feature source metadata mismatch')
            if src['effective_purpose'] == 'synthetic-test' and row.get('effective_purpose') != 'synthetic-test':
                raise ValueError('Source purpose downgraded')
    for r in allrows:
        ancestry[r['sample_id']] = r
    for r in allrows:
        def roots(sid, stack):
            if sid in stack or sid not in ancestry:
                raise ValueError('Invalid lineage')
            node = ancestry[sid]
            parents = node.get('parent_sample_ids', [node['parent_sample_id']] if node.get('parent_sample_id') else [])
            if not parents:
                source = node.get('source_recording_id')
                return {json.dumps([node.get('dataset_id', 'legacy'), source], ensure_ascii=False, separators=(',', ':')) if source else sid}
            return set().union(*(roots(p, stack | {sid}) for p in parents))
        r['lineage_roots'] = sorted(roots(r['sample_id'], set()))
    return {'rows': rows, 'cmvn': cmvn, 'identity': _digest(ready_raw), 'cmvn_identity': files['cmvn.json'], 'purpose': ready['purpose'], 'data_flags': flags, 'frontend': frontend, 'selection': ready.get('selection'), 'root': root}


class FeatureDataset(Dataset):
    def __init__(self, bundle, split, spec_augment=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        self.bundle, self.rows, self.split, self.spec_augment = bundle, bundle['rows'][split], split, spec_augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = _read(self.bundle['root'], row['feature_path'])
        if _digest(raw) != row['feature_sha256']:
            raise ValueError('Feature changed after verification')
        x = torch.from_numpy(np.load(io.BytesIO(raw), allow_pickle=False).copy())
        if self.split == 'train' and self.spec_augment is not None:
            x = self.spec_augment(x)
        return x, row['label'], row['sample_id']


def collate_features(batch):
    if not batch:
        raise ValueError('Empty batch')
    xs, labels, ids = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    return torch.nn.utils.rnn.pad_sequence(xs, batch_first=True), lengths, torch.tensor(labels, dtype=torch.long), list(ids)
