"""Rule and synthetic replay checks, without recognition-quality claims."""
import copy

import numpy as np
import pytest
import torch

from easykws.preprocess.frontend import extract_fbank
from easykws.streaming import StreamingScorer
from easykws.training.model import KWSModel
from easykws.trigger import FreshFrameTrigger, trigger_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_default_rules_have_no_threshold_and_do_not_mutate_input():
    config = {"comparison": "gt"}
    assert trigger_config(config) == {"confirm_frames": 3, "warmup_frames": 50,
                                     "rearm_sec": .3, "cooldown_sec": 2., "comparison": "gt"}
    assert config == {"comparison": "gt"}
    assert "threshold" not in trigger_config()
    with pytest.raises(TypeError):
        FreshFrameTrigger()


@pytest.mark.parametrize("config", [{"comparison": ">="}, {"confirm_frames": 0},
                                    {"warmup_frames": -1}, {"rearm_sec": 0},
                                    {"cooldown_sec": -1}, {"warmup_frames": True},
                                    {"rearm_sec": float("nan")}, {"threshold": .5}])
def test_invalid_trigger_rules(config):
    with pytest.raises(ValueError):
        FreshFrameTrigger(.5, config)


def test_warmup_uses_only_new_frames_and_confirmation_spans_warmup():
    trigger = FreshFrameTrigger(.5)
    for index in range(49):
        assert trigger.update(.8, .025 + .01 * index) == []
    events = trigger.update(.8, .515)
    assert len(events) == 1
    assert events[0]["frame_index"] == 49
    assert trigger.frames_seen == 50
    for index in range(50, 500):
        assert trigger.update(.8, .025 + .01 * index) == []
    assert trigger.event_count == 1
    assert len(trigger.history) == 200


@pytest.mark.parametrize("comparison,expected", [("ge", 1), ("gt", 0)])
def test_explicit_comparison(comparison, expected):
    trigger = FreshFrameTrigger(.5, {"warmup_frames": 0, "comparison": comparison})
    assert sum(len(trigger.update(.5, .025 + i * .01)) for i in range(3)) == expected


def test_rearm_and_held_high_during_cooldown_do_not_duplicate_episode():
    trigger = FreshFrameTrigger(.5, {"warmup_frames": 0, "confirm_frames": 1})
    assert len(trigger.update(.9, .025)) == 1
    assert trigger.update(.1, .1) == []
    assert trigger.update(.1, .39) == []  # shorter than the rearm interval
    assert trigger.update(.9, .4) == []
    assert trigger.update(.1, .5) == []
    assert trigger.update(.1, .8) == []  # rearmed while still in cooldown
    assert trigger.update(.9, .9) == []
    assert trigger.update(.9, 2.024) == []
    assert len(trigger.update(.9, 2.025)) == 1
    assert trigger.update(.9, 10.) == []
    assert trigger.event_count == 2


def test_low_interrupts_confirmation_and_invalid_frames_leave_state_unchanged():
    trigger = FreshFrameTrigger(.5, {"warmup_frames": 0})
    for score, t in [(.9, .025), (.9, .035), (.1, .045), (.9, .055), (.9, .065)]:
        assert trigger.update(score, t) == []
    before = copy.deepcopy(trigger.__dict__)
    for score, t in [(float("nan"), .075), (.9, float("inf")), (.9, .065), (1.1, .075)]:
        with pytest.raises(ValueError):
            trigger.update(score, t)
        assert trigger.__dict__ == before
    assert len(trigger.update(.9, .075)) == 1
    trigger.reset()
    assert trigger.frames_seen == trigger.event_count == 0
    assert not trigger.history and not trigger.events
    assert trigger.update(.9, .025) == []


def test_recent_event_diagnostics_are_bounded():
    trigger = FreshFrameTrigger(.5, {"confirm_frames": 1, "warmup_frames": 0,
                                      "cooldown_sec": 0, "rearm_sec": .01})
    for i in range(250):
        event = trigger.update(.9, i * .1)
        assert event[0]["index"] == i
        event[0]["score"] = -1
        assert trigger.events[-1]["score"] == .9
        trigger.update(.1, i * .1 + .02)
        trigger.update(.1, i * .1 + .04)
    assert trigger.event_count == 250
    assert len(trigger.events) == len(trigger.history) == 200


def small_model():
    torch.manual_seed(17)
    return KWSModel(hidden_dim=8, num_layers=2, kernel_size=3, dropout=0,
                    cmvn_mean=torch.zeros(40), cmvn_istd=torch.ones(40)).eval()


def proxy_audio():
    tone = (.1 * np.sin(2 * np.pi * 370 * np.arange(12000) / 16000)).astype(np.float32)
    return np.concatenate((np.zeros(8000, np.float32), tone, np.zeros(8321, np.float32)))


def replay(model, audio, chunk):
    scorer = StreamingScorer(model)
    rows = []
    trigger = FreshFrameTrigger(.0)
    events = []
    for start in range(0, len(audio), chunk):
        current = scorer.push(audio[start:start + chunk])
        rows.extend(current)
        for row in current:
            events.extend(trigger.update(row["score"], row["t_audio_sec"]))
        assert len(scorer.buffer) < 400
    assert scorer.sample_count == len(audio)
    assert scorer.audio_sec == len(audio) / 16000
    return scorer, rows, events


def test_chunk_lengths_silence_signal_silence_frame_count_scores_events():
    model = small_model()
    audio = proxy_audio()
    references = replay(model, audio, 160)
    expected_frames = 1 + (len(audio) - 400) // 160
    assert len(references[1]) == expected_frames
    assert [r["t_audio_sec"] for r in references[1]] == [(i * 160 + 400) / 16000 for i in range(expected_frames)]
    assert references[2][0]["frame_index"] == 49
    for chunk in (512, 4096, len(audio)):
        scorer, rows, events = replay(model, audio, chunk)
        assert rows == references[1]  # identical one-frame arithmetic on this CPU
        assert events == references[2]
        assert scorer.frames_processed == expected_frames
        assert scorer.flush() == scorer.flush() == []
        assert len(scorer.buffer) == 0


def test_offline_frontend_and_full_causal_forward_alignment():
    model = small_model()
    audio = proxy_audio()
    _, rows, _ = replay(model, audio, 512)
    full_features = extract_fbank(audio)
    individual = np.vstack([extract_fbank(audio[start:start + 400])
                            for start in range(0, len(audio) - 399, 160)])
    np.testing.assert_allclose(individual, full_features, rtol=0, atol=2e-5)
    with torch.inference_mode():
        expected, _ = model(torch.from_numpy(full_features).unsqueeze(0))
    np.testing.assert_allclose([row["score"] for row in rows], expected.numpy().reshape(-1), rtol=0, atol=2e-6)


def test_digital_silence_is_processed_and_empty_block_does_not_advance():
    class AlwaysHigh(torch.nn.Module):
        def forward(self, features, cache):
            return torch.full((1, 1, 1), .8), torch.zeros((1, 1, 1))

    scorer = StreamingScorer(AlwaysHigh())
    assert scorer.push(np.empty(0, np.float32)) == []
    assert scorer.sample_count == scorer.frames_processed == 0
    trigger = FreshFrameTrigger(.5)
    rows = scorer.push(np.zeros(16000, np.float32))
    events = [event for row in rows for event in trigger.update(row["score"], row["t_audio_sec"])]
    assert len(rows) == 98 and len(events) == 1
    assert scorer.audio_sec == 1 and events[0]["frame_index"] == 49


def test_empty_midstream_preserves_overlap_cache_and_next_frame():
    audio = proxy_audio()[:1041]
    reference = StreamingScorer(small_model())
    expected = reference.push(audio)
    scorer = StreamingScorer(small_model())
    first = scorer.push(audio[:512])
    before = (scorer.sample_count, scorer.frames_processed,
              scorer.buffer.copy(), scorer.cache.clone())
    for _ in range(3):
        assert scorer.push(np.empty(0, np.float32)) == []
        assert (scorer.sample_count, scorer.frames_processed) == before[:2]
        np.testing.assert_array_equal(scorer.buffer, before[2])
        assert torch.equal(scorer.cache, before[3])
    assert first + scorer.push(audio[512:]) == expected
    assert scorer.sample_count == 1041
    assert scorer.frames_processed == 5
    assert scorer.audio_sec == 1041 / 16000


def test_frame_clock_waits_for_complete_window_and_flush_never_pads():
    scorer = StreamingScorer(small_model())
    assert scorer.push(np.zeros(399, np.float32)) == []
    first = scorer.push(np.zeros(1, np.float32))
    assert [(r["frame_index"], r["t_audio_sec"]) for r in first] == [(0, .025)]
    assert scorer.push(np.zeros(159, np.float32)) == []
    second = scorer.push(np.zeros(1, np.float32))
    assert [(r["frame_index"], r["t_audio_sec"]) for r in second] == [(1, .035)]
    assert scorer.push(np.zeros(159, np.float32)) == []
    assert scorer.flush() == []
    assert scorer.frames_processed == 2
    assert scorer.sample_count == 719
    assert scorer.audio_sec == 719 / 16000


@pytest.mark.parametrize("invalid", [np.zeros((2, 1), np.float32), np.zeros(3, np.float64),
                                     np.array([np.nan], np.float32), np.array([np.inf], np.float32),
                                     np.array([1.01], np.float32), [0., 0.]])
def test_invalid_audio_leaves_state_unchanged(invalid):
    scorer = StreamingScorer(small_model())
    scorer.push(np.zeros(512, np.float32))
    before = (scorer.sample_count, scorer.frames_processed, scorer.buffer.copy(), scorer.cache.clone())
    with pytest.raises(ValueError):
        scorer.push(invalid)
    assert (scorer.sample_count, scorer.frames_processed) == before[:2]
    np.testing.assert_array_equal(scorer.buffer, before[2])
    assert torch.equal(scorer.cache, before[3])


def test_reset_flush_and_partial_frames():
    scorer = StreamingScorer(small_model())
    assert scorer.push(np.zeros(399, np.float32)) == []
    assert scorer.flush() == [] and scorer.frames_processed == 0
    with pytest.raises(ValueError, match="flushed"):
        scorer.push(np.zeros(1, np.float32))
    scorer.reset()
    first = scorer.push(proxy_audio())
    scorer.reset()
    assert scorer.sample_count == scorer.frames_processed == 0
    assert scorer.cache.numel() == 0
    assert scorer.push(proxy_audio()) == first


def test_unsupported_model_precision_and_autocast_rejected():
    with pytest.raises(ValueError, match="CPU FP32"):
        StreamingScorer(small_model().double())
    scorer = StreamingScorer(small_model())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            scorer.push(np.zeros(400, np.float32))
    assert scorer.sample_count == 0
