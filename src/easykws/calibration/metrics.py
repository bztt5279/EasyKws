"""Separate sample classification statistics and observed stream-event statistics."""
import math

from ..training.checkpoint import require


def sample_metrics(labels, scores, threshold=.5, comparison='ge'):
    require(len(labels) == len(scores) and all(x in (0, -1) for x in labels), 'Invalid sample labels')
    require(all(type(x) in (int, float) and math.isfinite(x) for x in scores), 'Invalid sample scores')
    require(comparison in {'ge', 'gt'}, 'Invalid comparison')
    positives = [s for y, s in zip(labels, scores) if y == 0]
    negatives = [s for y, s in zip(labels, scores) if y == -1]
    hit = (lambda s: s >= threshold) if comparison == 'ge' else (lambda s: s > threshold)
    # Mann-Whitney interpretation explicitly awards half credit to equal scores.
    auc = None
    if positives and negatives:
        ranked = sorted(zip(scores, labels))
        below = 0; pairs = 0.0; index = 0
        while index < len(ranked):
            end = index + 1
            while end < len(ranked) and ranked[end][0] == ranked[index][0]: end += 1
            group = ranked[index:end]
            pos = sum(y == 0 for _, y in group); neg = len(group) - pos
            pairs += pos * (below + .5 * neg)
            below += neg; index = end
        auc = pairs / (len(positives) * len(negatives))
    return {'positive_samples': len(positives), 'negative_samples': len(negatives), 'auc': auc,
            'sample_recall': sum(map(hit, positives)) / len(positives) if positives else None,
            'segmented_negative_fpr': sum(map(hit, negatives)) / len(negatives) if negatives else None,
            'threshold': threshold, 'comparison': comparison}


def event_metrics(sessions, session_events):
    hits = duplicates = false_events = excluded = positives = 0
    negative_sec = 0.0
    details = []
    for session in sessions:
        windows, negative = session['positive_windows'], session['negative_intervals']
        positives += len(windows)
        negative_sec += sum(i['end_sec'] - i['start_sec'] for i in negative)
        matched = set()
        classified = []
        for event in session_events[session['id']]:
            t = event['t_audio_sec']
            require(type(t) in (int, float) and math.isfinite(t) and 0 <= t <= session['duration_samples'] / 16000, 'Invalid event timestamp')
            window = next((w for w in windows if w['start_sec'] <= t < w['end_sec']), None)
            if window is not None:
                if window['id'] in matched:
                    duplicates += 1; category = 'duplicate-positive-window'
                else:
                    matched.add(window['id']); hits += 1; category = 'positive-hit'
            elif any(i['start_sec'] <= t < i['end_sec'] for i in negative):
                false_events += 1; category = 'negative-event'
            else:
                excluded += 1; category = 'out-of-scope'
            classified.append(dict(event, category=category))
        details.append({'session_id': session['id'], 'recording_kind': session['recording_kind'],
                        'matched_windows': sorted(matched), 'events': classified})
    return {'positive_windows': positives, 'positive_hits': hits, 'duplicate_positive_events': duplicates,
            'negative_events': false_events, 'excluded_events': excluded, 'negative_duration_sec': negative_sec,
            'recall': hits / positives if positives else None,
            'protocol_stream_fah': false_events * 3600 / negative_sec if negative_sec else None,
            'interval_rule': '[start_sec,end_sec)', 'scope': 'observed explicit negative intervals; protocol replay, not long-term field performance',
            'sessions': details}


def choose(records, min_recall, max_fah):
    feasible = [r for r in records if
                (min_recall is None or r['metrics']['recall'] is not None and r['metrics']['recall'] >= min_recall) and
                (max_fah is None or r['metrics']['protocol_stream_fah'] is not None and r['metrics']['protocol_stream_fah'] <= max_fah)]
    def key(r):
        recall, fah = r['metrics']['recall'], r['metrics']['protocol_stream_fah']
        return (-(recall if recall is not None else -1), fah if fah is not None else math.inf, -r['threshold'], r['candidate'])
    return min(feasible, key=key) if feasible else None
