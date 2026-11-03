"""Explicit, observed calibration sessions. Silence is valid audio, gaps are not."""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

from ..training.checkpoint import require
from ..training.data import _json
from ..prep.core import _audio_info

SCHEMA = 'easykws-calibration-sessions/v1'
KINDS = {'continuous-recording', 'offline-replay', 'synthetic-concatenation'}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _path(root, name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name, 'Session path must be relative POSIX')
    p = PurePosixPath(name)
    require(not p.is_absolute() and all(x not in ('.', '..') for x in p.parts), 'Unsafe session path')
    path = root.joinpath(*p.parts).resolve()
    require(path.is_relative_to(root) and path.suffix.lower() == '.wav', 'Session path escapes audio root or is not WAV')
    return path


def read_audio(path, expected_hash, duration_samples):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_hash, 'Session audio hash mismatch')
    with sf.SoundFile(io.BytesIO(raw)) as stream:
        require(stream.format == 'WAV' and stream.subtype in {'PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32'}, 'Session audio must be PCM WAV')
        require(stream.samplerate == 16000 and stream.channels == 1, 'Session audio must be 16 kHz mono')
        require(len(stream) == duration_samples, 'Session duration_samples mismatch')
        audio = stream.read(dtype='float32', always_2d=False)
    require(audio.ndim == 1 and len(audio) > 0 and np.isfinite(audio).all(), 'Invalid session audio')
    return audio


def load_sessions(manifest, bundle, path_root=None):
    manifest = Path(manifest).resolve()
    raw = manifest.read_bytes()
    value = _json(raw)
    require(isinstance(value, dict) and value.get('schema') == SCHEMA, 'Unsupported calibration session schema')
    required = {'schema', 'purpose', 'split', 'sample_rate', 'channels', 'clock', 'boundary_policy', 'gap_policy', 'provenance', 'sessions'}
    require(set(value) == required, 'Calibration manifest fields mismatch')
    require(value['purpose'] in {'recorded-data', 'training-data', 'synthetic-test', 'diagnostic-subset'},
            'Invalid calibration purpose')
    require(value['split'] in {'val', 'calibration'}, 'Calibration selection rejects train/test split')
    require((value['sample_rate'], value['channels'], value['clock'], value['boundary_policy'], value['gap_policy']) ==
            (16000, 1, 'samples', 'reset-each-session', 'split-session'), 'Unsupported calibration clock/boundary/gap policy')
    provenance = value['provenance']
    require(isinstance(provenance, dict) and set(provenance) == {'source_id', 'description'} and
            all(isinstance(v, str) and v.strip() for v in provenance.values()), 'Explicit calibration provenance required')
    root = Path(path_root).resolve() if path_root is not None else manifest.parent
    sessions = value['sessions']
    require(isinstance(sessions, list) and sessions, 'Calibration requires sessions')
    forbidden_hashes, forbidden_speakers, forbidden_pcm = set(), set(), set()
    # Raw source manifest catches exact source files even when features were augmented.
    original_splits = {}
    for split, rows in bundle['rows'].items():
        for row in rows:
            sid = row.get('augmentation', {}).get('source_sample_id', row['sample_id'])
            original_splits[sid] = split
            if split in {'train', 'test'}:
                forbidden_hashes.update(row[k] for k in ('audio_file_sha256', 'audio_sha256') if row.get(k))
                if row.get('audio_pcm_sha256'): forbidden_pcm.add(row['audio_pcm_sha256'])
                forbidden_speakers.update(row[k] for k in ('actual_speaker_id', 'speaker_id') if row.get(k))
                # Actual augmentation use determines exposure, even when the
                # resource catalog declares the asset as external. Preserve
                # PCM identity so changing the WAV wrapper cannot hide reuse.
                for resource in row.get('augmentation', {}).get('resources', []):
                    forbidden_hashes.update(resource[k] for k in ('sha256', 'audio_file_sha256', 'audio_sha256') if resource.get(k))
                    if resource.get('audio_pcm_sha256'):
                        forbidden_pcm.add(resource['audio_pcm_sha256'])
                    forbidden_speakers.update(resource[k] for k in ('actual_speaker_id', 'speaker_id') if resource.get(k))
    source_path = bundle['root'] / 'source_manifest.json'
    if source_path.exists():
        source = _json(source_path.read_bytes())
        for row in source.get('rows', []):
            if original_splits.get(row['sample_id'], row.get('split') or row.get('requested_split')) in {'train', 'test'}:
                if row.get('audio_sha256'): forbidden_hashes.add(row['audio_sha256'])
                if row.get('audio_pcm_sha256'): forbidden_pcm.add(row['audio_pcm_sha256'])
                if row.get('speaker_id'): forbidden_speakers.add(row['speaker_id'])
        for resource in source.get('resources', []):
            if resource.get('split') in {'train', 'test'}:
                forbidden_hashes.add(resource['sha256'])
                if resource.get('speaker_id'): forbidden_speakers.add(resource['speaker_id'])
    # Prepared excluded rows can retain assigned source split and identities.
    excluded_path = bundle['root'] / 'excluded.jsonl'
    if excluded_path.exists():
        for line in excluded_path.read_bytes().splitlines():
            if not line.strip(): continue
            row = _json(line)
            if row.get('split') in {'train', 'test'} or row.get('requested_split') in {'train', 'test'}:
                forbidden_hashes.update(row[k] for k in ('audio_file_sha256', 'audio_sha256') if row.get(k))
                if row.get('audio_pcm_sha256'): forbidden_pcm.add(row['audio_pcm_sha256'])
                forbidden_speakers.update(row[k] for k in ('actual_speaker_id', 'speaker_id') if row.get(k))
    seen_ids, seen_hashes, seen_pcm, paths = set(), set(), set(), {}
    for session in sessions:
        require(isinstance(session, dict) and set(session) == {'id', 'path', 'sha256', 'duration_samples', 'recording_kind', 'positive_windows', 'negative_intervals', 'speaker_ids'}, 'Session fields mismatch')
        sid = session['id']
        require(isinstance(sid, str) and sid and sid not in seen_ids, 'Duplicate or invalid session ID')
        seen_ids.add(sid)
        sha = session['sha256']
        require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'Invalid session SHA256')
        require(sha not in seen_hashes and sha not in forbidden_hashes, 'Session duplicates another session or train/test audio')
        seen_hashes.add(sha)
        speakers = session['speaker_ids']
        require(isinstance(speakers, list) and all(isinstance(x, str) and x for x in speakers) and len(set(speakers)) == len(speakers), 'Invalid speaker_ids')
        require(not set(speakers) & forbidden_speakers, 'Calibration speaker overlaps train/test')
        require(session['recording_kind'] in KINDS, 'Invalid recording_kind')
        require(value['purpose'] == 'synthetic-test' or session['recording_kind'] != 'synthetic-concatenation', 'Synthetic sessions require synthetic-test purpose')
        n = session['duration_samples']
        require(type(n) is int and n > 0, 'duration_samples must be positive integer')
        intervals, ids = [], set()
        for key in ('positive_windows', 'negative_intervals'):
            require(isinstance(session[key], list), 'Intervals must be lists')
            for interval in session[key]:
                fields = {'start_sec', 'end_sec'} | ({'id'} if key == 'positive_windows' else set())
                require(isinstance(interval, dict) and set(interval) == fields, 'Interval fields mismatch')
                a, b = interval['start_sec'], interval['end_sec']
                require(_number(a) and _number(b) and 0 <= a < b <= n / 16000, 'Invalid or out-of-duration interval')
                if key == 'positive_windows':
                    iid = interval['id']
                    require(isinstance(iid, str) and iid and iid not in ids, 'Invalid positive window ID')
                    ids.add(iid)
                intervals.append((a, b))
        intervals.sort()
        require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), 'Calibration intervals overlap')
        path = _path(root, session['path'])
        read_audio(path, sha, n)
        pcm = _audio_info(path)['audio_pcm_sha256']
        require(pcm not in forbidden_pcm and pcm not in seen_pcm, 'Session duplicates known train/test PCM or another session')
        seen_pcm.add(pcm)
        paths[sid] = path
    require(bundle['purpose'] != 'synthetic-test' or value['purpose'] == 'synthetic-test',
            'Calibration purpose cannot downgrade synthetic model')
    require(bundle['purpose'] != 'diagnostic-subset'
            or value['purpose'] in {'diagnostic-subset', 'synthetic-test'},
            'Calibration purpose cannot downgrade diagnostic-subset model')
    return value, hashlib.sha256(raw).hexdigest(), paths, root
