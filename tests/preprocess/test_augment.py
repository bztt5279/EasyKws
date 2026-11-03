import numpy as np
import pytest
from scipy.signal import butter, resample_poly, sosfilt

from easykws.preprocess.augment import prepare_resource, rms, stable_seed, transform
from easykws.preprocess.errors import PreprocessError


def waveform(n=16000):
    time = np.arange(n) / 16000
    return (0.1 * np.sin(2 * np.pi * 440 * time) +
            0.03 * np.sin(2 * np.pi * 731 * time)).astype(np.float32)


def test_clean_preserves_complete_waveform_without_peak_normalization():
    x = waveform(80000)
    x[100], x[40000], x[-100] = 0.99, -0.98, 0.97
    y, info = transform(x, [], peak_limit=0.2)
    np.testing.assert_array_equal(y, x)
    assert not np.shares_memory(x, y)
    assert info["safety_gain"] == 1 and info["samples"] == 80000
    assert info["operations"] == []


def test_seed_is_stable_across_row_order_and_sensitive_to_identity():
    rows = ["sample-0", "sample-1", "sample-2"]
    a = {row: stable_seed(17, row, 3) for row in rows}
    b = {row: stable_seed(17, row, 3) for row in reversed(rows)}
    assert a == b
    assert len({stable_seed(17, "sample-0", 3), stable_seed(18, "sample-0", 3),
                stable_seed(17, "sample-1", 3), stable_seed(17, "sample-0", 4)}) == 4


def test_gain_peak_guard_occurs_once_after_chain():
    x = waveform()
    ops = [{"kind": "gain", "value": 20.0}, {"kind": "gain", "value": 0.02}]
    y, info = transform(x, ops)
    np.testing.assert_array_equal(y, (x * 20.0 * 0.02).astype(np.float32))
    assert info["safety_gain"] == 1
    loud, info = transform(x, ops[:1], peak_limit=0.7)
    assert np.max(np.abs(loud)) == pytest.approx(0.7, abs=1e-7)
    assert info["safety_gain"] < 1


def test_speed_changes_both_length_and_pitch():
    x = waveform()
    y, info = transform(x, [{"kind": "speed", "value": 1.25}])
    np.testing.assert_array_equal(y, resample_poly(x, 4, 5))
    assert len(y) == 12800
    frequency = np.fft.rfftfreq(len(y), 1 / 16000)[np.argmax(abs(np.fft.rfft(y)))]
    assert frequency == pytest.approx(550)
    assert info["operations"][0]["resample_up"] == 4


def test_filter_is_fourth_order_causal_butterworth_sos():
    x = waveform()
    expected = sosfilt(butter(4, [100, 3200], fs=16000, btype="bandpass", output="sos"), x)
    y, info = transform(x, [{"kind": "filter", "low": 100, "high": 3200}])
    np.testing.assert_array_equal(y, expected.astype(np.float32))
    assert info["operations"][0]["implementation"] == "sosfilt"


def test_short_noise_repeats_then_selects_segment_with_target_snr():
    x = waveform(3000)
    noise = np.random.default_rng(1).normal(0, 0.1, 1200).astype(np.float32)
    y, info = transform(x, [{"kind": "noise", "resource_id": "background",
                            "snr_db": 13.0, "position": 0.5}],
                        resources={"background": (noise, 16000)})
    step = info["operations"][0]
    assert (step["repeats"], step["start_sample"], step["end_sample"]) == (3, 300, 3300)
    assert 20 * np.log10(rms(x) / rms(y - x)) == pytest.approx(13, abs=1e-5)


def test_rir_selects_channel_removes_peak_delay_and_retains_tail():
    x = waveform(2000)
    impulse = np.zeros((71, 2), dtype=np.float32)
    impulse[2, 0] = 1
    impulse[10, 1], impulse[70, 1] = 1, 0.4
    y, info = transform(x, [{"kind": "rir", "resource_id": "room", "channel": 1}],
                        resources={"room": (impulse, 16000)})
    step = info["operations"][0]
    assert step["peak_sample"] == 10
    assert len(y) == len(x) + len(impulse) - 1 - 10
    assert np.any(abs(y[len(x):]) > 1e-5)
    assert rms(y[:len(x)]) == pytest.approx(rms(x), rel=1e-6)


def test_resource_channel_conversion_precedes_resampling():
    x = np.stack([waveform(3000), waveform(3000) * 0.5], axis=1)
    noise, info = prepare_resource(x, 24000)
    rir, _ = prepare_resource(x, 24000, channel=1)
    np.testing.assert_array_equal(noise, resample_poly(x.mean(axis=1), 2, 3))
    np.testing.assert_array_equal(rir, resample_poly(x[:, 1], 2, 3))
    assert info["source_channels"] == 2 and info["processed_samples"] == 2000


def test_explicit_crop_records_interval_and_strict_length_trigger():
    op = {"kind": "crop", "length": 2.5, "position": 0.25, "only_if_longer_than": 3}
    x = waveform(80000)
    y, info = transform(x, [op])
    np.testing.assert_array_equal(y, x[10000:50000])
    step = info["operations"][0]
    assert (step["start_sample"], step["end_sample"]) == (10000, 50000)
    assert (step["original_duration_sec"], step["retained_duration_sec"],
            step["discarded_duration_sec"]) == (5, 2.5, 2.5)
    boundary = waveform(48000)
    unchanged, info = transform(boundary, [op])
    np.testing.assert_array_equal(unchanged, boundary)
    assert info["operations"][0]["applied"] is False


@pytest.mark.parametrize("op", [
    {"kind": "gain", "value": float("nan")},
    {"kind": "speed", "value": 0},
    {"kind": "filter", "low": 100, "high": 8000},
    {"kind": "noise", "resource_id": "missing", "snr_db": 10, "position": 0},
    {"kind": "noise", "path": "background.wav", "snr_db": 10, "position": 0},
    {"kind": "unknown"},
])
def test_invalid_operations_raise_domain_error(op):
    with pytest.raises(PreprocessError):
        transform(waveform(1000), [op])


def test_zero_resource_and_out_of_range_channel_rejected():
    with pytest.raises(PreprocessError):
        prepare_resource(np.zeros(100), 16000)
    with pytest.raises(PreprocessError):
        prepare_resource(waveform(100), 16000, channel=1)
