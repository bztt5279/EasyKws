"""Validation-only candidate evaluation and explicit session calibration."""
import copy
import json
import math
from pathlib import Path

import torch

from ..training.checkpoint import require, file_hash
from ..training.data import load_bundle, FeatureDataset, _json
from ..training import runner
from ..streaming import StreamingScorer
from ..trigger import FreshFrameTrigger, trigger_config
from . import binding
from .sessions import load_sessions, read_audio
from .metrics import sample_metrics, event_metrics, choose

RULE = 'recall-desc/fah-asc/threshold-desc/candidate-asc'


def _grid(values):
    require(isinstance(values, list) and values and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in values), 'Explicit finite threshold grid in [0,1] required')
    require(len(set(values)) == len(values), 'Duplicate thresholds')
    return list(values)


def _names(values):
    require(isinstance(values, list) and values and all(isinstance(x, str) and x for x in values) and len(set(values)) == len(values), 'Explicit unique candidate names required')
    return list(values)


def configure(config=None):
    if isinstance(config, (str, Path)):
        config = _json(Path(config).read_bytes())
    require(isinstance(config, dict), 'Explicit calibration config required')
    config = copy.deepcopy(config)
    allowed = {'schema', 'candidates', 'threshold_grid', 'min_recall', 'max_fah', 'trigger', 'chunk_samples', 'selection_rule'}
    require(not set(config) - allowed, 'Unknown calibration config field')
    require(config.get('schema', 'easykws-calibration-config/v1') == 'easykws-calibration-config/v1', 'Invalid calibration config schema')
    config['schema'] = 'easykws-calibration-config/v1'
    config['candidates'] = _names(config.get('candidates'))
    config['threshold_grid'] = _grid(config.get('threshold_grid'))
    for key, maximum in [('min_recall', 1), ('max_fah', math.inf)]:
        value = config.get(key)
        require(value is None or type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum, 'Invalid constraint: ' + key)
        config[key] = value
    require(config['min_recall'] is not None or config['max_fah'] is not None, 'At least one calibration constraint required')
    config['trigger'] = trigger_config(config.get('trigger'))
    config.setdefault('chunk_samples', 512)
    require(type(config['chunk_samples']) is int and config['chunk_samples'] > 0, 'Invalid chunk_samples')
    config.setdefault('selection_rule', RULE)
    require(config['selection_rule'] == RULE, 'Unsupported selection rule')
    return config


def _write(root, name, value):
    (root / name).write_bytes(runner.encode(value))


def _finish(root, report):
    _write(root, 'report.json', report)
    _write(root, 'READY.json', {'schema': 'easykws-calibration-output/v1', 'files': {p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()}, 'purpose': report['purpose'], 'data_flags': report['data_flags']})
    return report


def _begin(output, protected):
    root = Path(output).resolve()
    for source in protected:
        source = Path(source).resolve()
        require(not root.is_relative_to(source) and not source.is_relative_to(root), 'Output overlaps protected input path')
    require(not root.exists(), 'Output directory already exists')
    root.mkdir(parents=True)
    return root


def calibrate(run_dir, features, manifest, output, config=None, path_root=None):
    root = _begin(output, [run_dir, features, Path(path_root).resolve() if path_root is not None else Path(manifest).resolve().parent])
    try:
        config = configure(config)
        bundle = load_bundle(features)
        sessions, manifest_sha, paths, _ = load_sessions(manifest, bundle, path_root)
        identity = {'schema': 'easykws-calibration-identity/v1', 'manifest_sha256': manifest_sha,
                    'inputs': {s['id']: s['sha256'] for s in sessions['sessions']},
                    'protocol': {k: v for k, v in sessions.items() if k != 'sessions'},
                    'grid': config['threshold_grid'], 'constraints': {k: config[k] for k in ('min_recall', 'max_fah')},
                    'selection_policy': config['selection_rule'], 'candidates': config['candidates'], 'chunk_samples': config['chunk_samples']}
        records = []
        for candidate in config['candidates']:
            model, metadata = binding.load_candidate(run_dir, candidate, bundle)
            frames = {}
            for session in sessions['sessions']:
                audio = read_audio(paths[session['id']], session['sha256'], session['duration_samples'])
                scorer = StreamingScorer(model)
                rows = []
                for start in range(0, len(audio), config['chunk_samples']):
                    rows.extend(scorer.push(audio[start:start + config['chunk_samples']]))
                rows.extend(scorer.flush())
                frames[session['id']] = rows
            _write(root, 'scores_%04d.json' % len(records), {'candidate': candidate, 'sessions': frames})
            for threshold in config['threshold_grid']:
                events = {}
                for sid, rows in frames.items():
                    trigger = FreshFrameTrigger(threshold, config['trigger'])
                    events[sid] = [event for row in rows for event in trigger.update(row['score'], row['t_audio_sec'])]
                records.append({'candidate': candidate, 'threshold': threshold, 'metrics': event_metrics(sessions['sessions'], events),
                                'binding': binding.build_binding(model, metadata, {'threshold': threshold, 'rules': config['trigger']}, identity)})
        selected = choose(records, config['min_recall'], config['max_fah'])
        # A restricted calibration purpose is itself a marker that must travel
        # with the report; a restricted model can never be calibrated clean.
        session_flags = {sessions['purpose']} if sessions['purpose'] in binding.RESTRICTED_PURPOSES else set()
        report = {'schema': 'easykws-calibration-report/v1', 'purpose': sessions['purpose'],
                  'data_flags': sorted(set(bundle['data_flags']) | session_flags),
                  'config': config, 'calibration_identity': identity, 'feasible': selected is not None, 'records': records,
                  'selected_checkpoint': selected['candidate'] if selected else None, 'selected_threshold': selected['threshold'] if selected else None}
        if selected:
            _write(root, 'selected_checkpoint.json', selected)
        return _finish(root, report)
    except Exception as exc:
        _write(root, 'FAILED.json', {'error': type(exc).__name__, 'message': str(exc)})
        raise


def evaluate(run_dir, features, output, candidates=None, thresholds=None):
    root = _begin(output, [run_dir, features])
    try:
        bundle = load_bundle(features)
        names = _names(candidates if candidates is not None else [r['path'] for r in runner._read_status(Path(run_dir))['candidates']])
        thresholds = _grid([.5] if thresholds is None else thresholds)
        dataset = FeatureDataset(bundle, 'val')
        require(len(dataset) > 0, 'Validation split is empty')
        records = []
        for candidate in names:
            model, metadata = binding.load_candidate(run_dir, candidate, bundle)
            labels, scores, ids = [], [], []
            for index in range(len(dataset)):
                features_tensor, label, sample_id = dataset[index]
                with torch.inference_mode():
                    probabilities, _ = model(features_tensor.unsqueeze(0))
                labels.append(label); scores.append(float(probabilities.max())); ids.append(sample_id)
            records.append({'candidate': candidate, 'sample_ids': ids, 'max_scores': scores, 'labels': labels,
                            'source_checkpoint_sha256': metadata['source_checkpoint_sha256'],
                            'metrics': [sample_metrics(labels, scores, threshold) for threshold in thresholds]})
        return _finish(root, {'schema': 'easykws-segmented-evaluation/v1', 'split': 'val', 'purpose': bundle['purpose'], 'data_flags': bundle['data_flags'], 'records': records})
    except Exception as exc:
        _write(root, 'FAILED.json', {'error': type(exc).__name__, 'message': str(exc)})
        raise
