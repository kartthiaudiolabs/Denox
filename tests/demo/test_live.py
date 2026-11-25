# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The microphone uses one stream, preserving chunk boundaries and levels."""

import json
import threading

import numpy as np
import pytest

from inworld import voice_isolation
import live


def _isolator(network):
    return voice_isolation.VoiceIsolator(network,
                                         device="cpu",
                                         precision="fp32",
                                         mode="streaming",
                                         lookahead_frames=2,
                                         block_frames=4)


def test_incremental_microphone_matches_a_whole_stream(windowed_model):
    isolator = _isolator(windowed_model)
    samples = np.random.RandomState(7).normal(0, 0.02, 4003).astype(np.float32)
    expected = isolator.isolate(samples, 48000)
    session = live.LiveSession(isolator)

    for chunk in np.array_split(samples, 7):
        update = session.push((48000, chunk))
    rate, actual = session.finish()

    assert not update.finished
    assert update.output_seconds > 0
    assert rate == 48000
    assert actual.shape == samples.shape
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(session.finish()[1], actual)
    assert session.push((48000, samples)).finished
    np.testing.assert_array_equal(session.finish()[1], actual)


@pytest.mark.parametrize("rate", [16000, 44100])
def test_resampling_preserves_fractional_positions_and_flushes_tail(
        identity_model, rate):
    soxr = pytest.importorskip("soxr")
    samples = (0.05 * np.sin(np.arange(1103) * 0.07)).astype(np.float32)
    expected = soxr.resample(samples, rate, 48000)
    session = live.LiveSession(_isolator(identity_model))

    for chunk in np.array_split(samples, 13):
        session.push((rate, chunk))
    out_rate, actual = session.finish()

    assert out_rate == 48000
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert session.snapshot().input_seconds == pytest.approx(
        len(samples) / rate)


def test_integer_stereo_audio_keeps_its_absolute_level(identity_model):
    pcm = np.tile(
        np.array([[-32768, 0], [0, 32767], [1000, 3000]], dtype=np.int16),
        (101, 1))
    expected = (pcm.astype(np.float32) / 32768).mean(axis=1)
    session = live.LiveSession(_isolator(identity_model))

    session.push((48000, pcm))
    _, actual = session.finish()

    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert actual.max() < 0.51


def test_two_browsers_have_independent_streams_and_recordings(windowed_model):
    isolator = _isolator(windowed_model)
    first = np.random.RandomState(2).normal(0, 0.02, 2033).astype(np.float32)
    second = np.random.RandomState(3).normal(0, 0.04, 2033).astype(np.float32)
    expected = [isolator.isolate(x, 48000) for x in (first, second)]
    model_lock = threading.Lock()
    sessions = [
        live.LiveSession(isolator, model_lock=model_lock) for _ in range(2)
    ]

    for offset in range(0, first.size, 137):
        for session, samples in zip(sessions, (first, second)):
            session.push((48000, samples[offset:offset + 137]))
    actual = [session.finish()[1] for session in sessions]

    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=2e-6)
    assert not np.allclose(actual[0], actual[1])


@pytest.mark.parametrize("rate", [44100, 48000])
def test_recording_limit_caps_audio_and_automatically_finishes(
        identity_model, rate):
    if rate != 48000:
        pytest.importorskip("soxr")
    session = live.LiveSession(_isolator(identity_model), max_seconds=0.02)
    samples = np.zeros(rate // 10, dtype=np.float32)

    update = session.push((rate, samples))
    _, actual = session.finish()

    assert update.limit_reached and update.finished
    assert update.input_seconds == pytest.approx(0.02)
    assert actual.shape == (960,)
    session.push((rate, samples))
    np.testing.assert_array_equal(session.finish()[1], actual)


def test_visualizations_are_bounded_aligned_and_json_serializable(
        identity_model):
    session = live.LiveSession(_isolator(identity_model), window_seconds=0.01)
    samples = np.full(1600, 0.03, dtype=np.float32)

    session.push((48000, samples))
    session.finish()
    update = session.snapshot()
    data = update.visualization

    assert data["end_seconds"] - data["start_seconds"] == pytest.approx(0.01)
    assert data["end_seconds"] == pytest.approx(samples.size / 48000)
    assert len(data["input"]["waveform"]) == live.WAVEFORM_BINS
    spec = np.asarray(data["output"]["spectrogram"])
    assert spec.shape == (live.FREQUENCY_BINS, live.TIME_BINS)
    assert 0 <= spec.min() <= spec.max() <= 255
    np.testing.assert_allclose(data["input"]["waveform"],
                               data["output"]["waveform"],
                               atol=1e-7)
    assert json.loads(json.dumps(data, allow_nan=False))["sample_rate"] == 48000


def test_changed_sample_rate_requires_a_new_recording(identity_model):
    session = live.LiveSession(_isolator(identity_model))
    samples = np.zeros(100, dtype=np.float32)
    session.push((48000, samples))

    with pytest.raises(ValueError, match="sample rate changed"):
        session.push((44100, samples))

    assert session.finish()[1].size == samples.size


@pytest.mark.parametrize("audio", [
    (0, np.zeros(10)),
    (48000.5, np.zeros(10)),
    (48000, np.zeros((2, 3, 4))),
    (48000, np.array([np.nan])),
    (48000, np.array([np.inf])),
    (48000, np.array(["not audio"])),
])
def test_invalid_microphone_input_does_not_enter_the_stream(
        identity_model, audio):
    session = live.LiveSession(_isolator(identity_model))

    with pytest.raises(ValueError):
        session.push(audio)

    assert session.finish()[1].size == 0


def test_empty_recording_and_reset_release_the_session(identity_model):
    session = live.LiveSession(_isolator(identity_model))

    assert session.push(None).input_seconds == 0
    assert session.finish()[1].size == 0
    session.reset()

    update = session.snapshot()
    assert update.finished and update.output_seconds == 0
    assert update.visualization["input"]["waveform"] == []
    assert session.finish()[1].size == 0


def test_live_session_rejects_an_offline_model(identity_model):
    isolator = voice_isolation.VoiceIsolator(identity_model, device="cpu")

    with pytest.raises(ValueError, match="realtime model"):
        live.LiveSession(isolator)


@pytest.mark.parametrize("rate", [48000, 16000])
def test_the_recorded_input_is_the_streamed_audio_aligned_with_the_voice(
        windowed_model, rate):
    soxr = pytest.importorskip("soxr")
    samples = (0.05 * np.sin(np.arange(2207) * 0.05)).astype(np.float32)
    expected = (samples if rate == 48000 else soxr.resample(
        samples, rate, 48000))
    session = live.LiveSession(_isolator(windowed_model))

    for chunk in np.array_split(samples, 5):
        session.push((rate, chunk))
    _, voice = session.finish()
    recorded_rate, recorded = session.recorded()
    session.reset()

    assert recorded_rate == 48000
    assert recorded.shape == voice.shape == expected.shape
    np.testing.assert_allclose(recorded, expected, atol=1e-7)
    assert session.recorded()[1].size == 0
