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
"""Browser event ordering and isolated microphone recordings."""

import base64
import threading
import time
from unittest import mock

import numpy as np
import pytest
import soundfile as sf

from inworld import voice_isolation
import live

pytest.importorskip("gradio")

# pylint: disable=wrong-import-position
import gradio as gr

import online

# pylint: enable=wrong-import-position


@pytest.fixture(name="server")
def online_server(identity_model, monkeypatch, tmp_path):
    """A real tiny model with player files confined to the test directory."""
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    model = voice_isolation.VoiceIsolator(identity_model,
                                          device="cpu",
                                          precision="fp32",
                                          mode="streaming",
                                          lookahead_frames=2)
    return online.Online(model, threading.RLock())


def _event(take="first", **data):
    return gr.EventData(None, {"take": take, **data})


def _pcm(samples):
    """Base64 little-endian 16-bit PCM, as the browser sends it."""
    scaled = np.round(np.asarray(samples, dtype=np.float64) * 32768)
    pcm = np.clip(scaled, -32768, 32767).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode()


def _request(browser):
    return gr.Request(session_hash=browser)


def _push(server, recording, sequence, samples, rate=48000, take="first"):
    return server.push(
        recording, _event(take, sequence=sequence, pcm=_pcm(samples),
                          rate=rate))


def _preview(audio):
    samples, rate = sf.read(audio.value["path"], dtype="int16")
    assert rate == 48000
    return samples


def test_ordered_chunks_finish_once_and_keep_playback_at_its_own_level(server):
    recording, ready, audio, original = server.start(None, _event())
    first = np.full(503, 0.03, dtype=np.float32)
    last = np.full(211, -0.02, dtype=np.float32)

    recording, active = _push(server, recording, 1, first)
    recording, _ = _push(server, recording, 2, last)
    same, final, player, original_after = server.finish(
        recording, _event(sequence=2, listen=True))
    _, repeated, repeated_player, _ = server.finish(
        recording, _event(sequence=2, listen=False))

    assert ready["phase"] == "ready" and ready["sequence"] == 0
    assert audio is None and original is None
    assert active["phase"] == "active" and active["sequence"] == 1
    assert same is recording
    assert final["phase"] == repeated["phase"] == "finished"
    # The browser starts playback itself once the player has loaded.
    assert not player.autoplay and not repeated_player.autoplay
    expected = np.round(np.concatenate((first, last)) * 32767).astype(np.int16)
    np.testing.assert_allclose(_preview(player), expected, atol=1)
    np.testing.assert_array_equal(_preview(repeated_player), _preview(player))
    np.testing.assert_allclose(_preview(original_after), expected, atol=1)
    assert not original_after.autoplay
    assert final["input_seconds"] == pytest.approx(expected.size / 48000)


def test_a_new_take_releases_the_previous_recording(server):
    previous, *_ = server.start(None, _event("previous"))
    _push(server, previous, 1, np.ones(200, dtype=np.float32), take="previous")

    current, response, audio, original = server.start(previous,
                                                      _event("current"))

    assert current is not previous
    assert current.sequence == 0
    assert response["take"] == "current" and response["input_seconds"] == 0
    assert audio is None and original is None
    assert previous.session.snapshot().finished
    assert previous.session.finish()[1].size == 0


def test_late_events_from_a_previous_take_do_not_touch_the_current_one(server):
    current, *_ = server.start(None, _event("current"))

    state, response = _push(server, current, 1, np.ones(100), take="previous")
    ended, final, player, original = server.finish(
        current, _event("previous", sequence=0))

    assert state is ended is current
    for answer in (response, final):
        assert answer["phase"] == "error" and answer["take"] == "previous"
    assert player == original == gr.skip()
    assert current.sequence == 0
    assert current.session.snapshot().input_seconds == 0


@pytest.mark.parametrize("sequence", [1, 3])
def test_duplicate_or_missing_chunks_abort_instead_of_replaying_audio(
        server, sequence):
    recording, *_ = server.start(None, _event())
    _push(server, recording, 1, np.zeros(100, dtype=np.float32))

    _, response = _push(server, recording, sequence,
                        np.zeros(100, dtype=np.float32))

    assert response["phase"] == "error"
    assert "order" in response["message"]
    assert recording.session.snapshot().finished
    assert recording.session.finish()[1].size == 0


def test_finish_before_the_last_chunk_is_acknowledged_reports_it(server):
    recording, *_ = server.start(None, _event())
    _push(server, recording, 1, np.zeros(100, dtype=np.float32))

    _, early, voice, original = server.finish(recording, _event(sequence=2))

    assert early["phase"] == "error" and "did not reach" in early["message"]
    assert voice == original == gr.skip()
    assert not recording.session.snapshot().finished
    _push(server, recording, 2, np.zeros(37, dtype=np.float32))
    _, response, audio, _ = server.finish(recording, _event(sequence=2))
    assert response["phase"] == "finished"
    assert _preview(audio).size == 137


def test_two_browser_states_remain_independent_on_one_server(server):
    first, *_ = server.start(None, _event("first"))
    second, *_ = server.start(None, _event("second"))

    _push(server, first, 1, np.full(200, 0.01, dtype=np.float32), take="first")
    _push(server,
          second,
          1,
          np.full(307, -0.02, dtype=np.float32),
          take="second")
    _, _, first_audio, _ = server.finish(first, _event("first", sequence=1))
    _, _, second_audio, _ = server.finish(second, _event("second", sequence=1))

    assert _preview(first_audio).shape == (200,)
    assert _preview(second_audio).shape == (307,)
    np.testing.assert_allclose(_preview(first_audio),
                               round(0.01 * 32767),
                               atol=1)
    np.testing.assert_allclose(_preview(second_audio),
                               round(-0.02 * 32767),
                               atol=1)


def test_the_duration_cap_reports_to_the_browser_and_ignores_late_audio(server):
    session = live.LiveSession(server.model,
                               max_seconds=0.02,
                               model_lock=server.model_lock)
    recording = online.Recording("first", session)

    _, capped = _push(server, recording, 1, np.zeros(1200, dtype=np.float32))
    _push(server, recording, 2, np.ones(100, dtype=np.float32))
    _, final, audio, _ = server.finish(recording, _event(sequence=2))

    assert capped["limit_reached"] and final["limit_reached"]
    assert final["input_seconds"] == pytest.approx(0.02)
    np.testing.assert_array_equal(_preview(audio), np.zeros(960,
                                                            dtype=np.int16))


@pytest.mark.parametrize("data", [
    {
        "rate": 0,
        "pcm": _pcm([0.0])
    },
    {
        "rate": 48000.5,
        "pcm": _pcm([0.0])
    },
    {
        "rate": 48000,
        "pcm": [0.0]
    },
    {
        "rate": 48000,
        "pcm": "not audio!"
    },
    {
        "rate": 48000,
        "pcm": base64.b64encode(b"odd").decode()
    },
    {
        "rate": 48000,
        "pcm": _pcm(np.zeros(48001))
    },
])
def test_invalid_browser_audio_is_rejected_before_sequence_advances(
        server, data):
    recording, *_ = server.start(None, _event())

    _, response = server.push(recording, _event(sequence=1, **data))

    assert response["phase"] == "error"
    assert recording.sequence == 0
    assert recording.session.snapshot().input_seconds == 0
    assert recording.session.snapshot().finished


@pytest.mark.parametrize("sequence", [True, 1.0])
def test_chunk_sequence_is_an_integer_counter(server, sequence):
    recording, *_ = server.start(None, _event())

    _, response = _push(server, recording, sequence, np.zeros(10))

    assert response["phase"] == "error"
    assert recording.sequence == 0


def test_expiring_browser_state_releases_its_recording(server):
    recording, *_ = server.start(None, _event())
    _push(server, recording, 1, np.full(500, 0.01, dtype=np.float32))

    online.close_recording(recording)
    online.close_recording(None)

    assert recording.session.snapshot().finished
    assert recording.session.finish()[1].size == 0


def test_finish_returns_the_unprocessed_input_beside_the_voice(server):
    labelled = online.Online(server.model, server.model_lock, "Metal GPU")
    recording, ready, *_ = labelled.start(None, _event())
    noisy = np.linspace(-0.5, 0.5, 701, dtype=np.float32)
    recording, active = _push(labelled, recording, 1, noisy)
    silent = (48000, np.zeros(noisy.size, dtype=np.float32))

    with mock.patch.object(recording.session,
                           "finish",
                           autospec=True,
                           return_value=silent):
        _, final, voice, original = labelled.finish(
            recording, _event(sequence=1, listen=True))

    np.testing.assert_array_equal(_preview(voice), np.zeros(701, np.int16))
    np.testing.assert_allclose(_preview(original),
                               np.round(noisy * 32767),
                               atol=1)
    assert not voice.autoplay and not original.autoplay
    assert ready["device"] == active["device"] == final["device"] == "Metal GPU"


def test_a_busy_server_refuses_a_take_until_one_finishes(server):
    first, *_ = server.start(None, _event("first"), _request("a"))
    server.start(None, _event("second"), _request("b"))

    refused, response, *_ = server.start(None, _event("third"), _request("c"))
    _push(server, first, 1, np.zeros(100, dtype=np.float32))
    server.finish(first, _event("first", sequence=1))
    third, ready, *_ = server.start(None, _event("third"), _request("c"))

    assert server.max_takes == 2
    assert refused is None
    assert response["phase"] == "error" and response["take"] == "third"
    assert "busy" in response["message"]
    assert ready["phase"] == "ready" and third.take == "third"


def test_closing_the_tab_releases_its_take_at_once(server):
    recording, *_ = server.start(None, _event(), _request("tab"))
    _push(server, recording, 1, np.full(500, 0.01, dtype=np.float32))
    other, *_ = server.start(None, _event("other"), _request("other tab"))

    server.release(_request("tab"))
    server.release(_request("never recorded"))

    assert recording.session.snapshot().finished
    assert recording.session.finish()[1].size == 0
    assert not other.session.snapshot().finished


def test_a_tab_whose_state_expired_still_releases_its_last_take(server):
    previous, *_ = server.start(None, _event("previous"), _request("tab"))

    current, *_ = server.start(None, _event("current"), _request("tab"))

    assert previous.session.snapshot().finished
    assert not current.session.snapshot().finished


def test_a_failed_flush_ends_the_take_with_an_error(server):
    recording, *_ = server.start(None, _event())
    _push(server, recording, 1, np.zeros(100, dtype=np.float32))

    with mock.patch.object(recording.session,
                           "finish",
                           autospec=True,
                           side_effect=RuntimeError("no full recording")):
        _, response, voice, original = server.finish(recording,
                                                     _event(sequence=1))

    assert response["phase"] == "error" and response["take"] == "first"
    assert "no full recording" in response["message"]
    assert voice == original == gr.skip()
    assert recording.session.snapshot().finished


def test_a_second_of_audio_fits_one_event_at_any_rate(server):
    recording, *_ = server.start(None, _event())

    _, response = _push(server, recording, 1, np.zeros(44100), rate=44100)

    assert response["phase"] == "active"
    assert response["input_seconds"] == pytest.approx(1.0)


def test_a_take_that_cannot_start_reports_why(server):
    with mock.patch.object(live,
                           "LiveSession",
                           autospec=True,
                           side_effect=RuntimeError("MPS out of memory")):
        recording, response, voice, original = server.start(None, _event())

    assert recording is None and voice is None and original is None
    assert response["phase"] == "error" and response["take"] == "first"
    assert "out of memory" in response["message"]


def test_a_take_does_not_wait_long_for_a_busy_model(server):
    impatient = online.Online(server.model, server.model_lock, start_wait=0.05)
    held, release = threading.Event(), threading.Event()

    def hold():
        with server.model_lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    try:
        recording, response, *_ = impatient.start(None, _event())
    finally:
        release.set()
        holder.join()

    assert recording is None
    assert response["phase"] == "error" and "busy" in response["message"]
    assert impatient.start(None, _event("later"))[1]["phase"] == "ready"


def test_a_silent_take_gives_up_its_place(server, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(online.time, "monotonic", lambda: now[0])
    first, *_ = server.start(None, _event("first"), _request("a"))
    server.start(None, _event("second"), _request("b"))
    _push(server, first, 1, np.zeros(100, dtype=np.float32))

    now[0] += online.IDLE_SECONDS + 1
    third, response, *_ = server.start(None, _event("third"), _request("c"))

    assert response["phase"] == "ready" and third.take == "third"


def test_an_idle_take_is_released_and_refused_when_another_starts(
        server, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(online.time, "monotonic", lambda: now[0])
    idle, *_ = server.start(None, _event("idle"), _request("a"))
    _push(server, idle, 1, np.zeros(100, dtype=np.float32), take="idle")
    other, *_ = server.start(None, _event("other"), _request("b"))

    now[0] += online.IDLE_SECONDS + 1
    third, ready, *_ = server.start(None, _event("third"), _request("c"))
    _, late = _push(server,
                    idle,
                    2,
                    np.zeros(100, dtype=np.float32),
                    take="idle")
    _, finished, voice, original = server.finish(idle, _event("idle",
                                                              sequence=1))

    assert ready["phase"] == "ready" and third.take == "third"
    assert idle.session.finished and other.session.finished
    assert late["phase"] == "error" and "without audio" in late["message"]
    assert finished["phase"] == "error"
    assert voice == original == gr.skip()


def test_a_take_still_sending_audio_is_never_idle(server, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(online.time, "monotonic", lambda: now[0])
    waiting, *_ = server.start(None, _event("waiting"), _request("a"))
    waiting.pushing = 1

    now[0] += online.IDLE_SECONDS + 1
    server.start(None, _event("other"), _request("b"))

    assert not waiting.expired and not waiting.session.finished


def test_a_tab_closed_while_its_take_starts_leaves_nothing_behind(server):
    slow = online.Online(server.model, server.model_lock, start_wait=5)
    held, free = threading.Event(), threading.Event()
    answer = {}

    def hold():
        with server.model_lock:
            held.set()
            free.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    starter = threading.Thread(target=lambda: answer.update(result=slow.start(
        None, _event(), _request("tab"))))
    starter.start()
    time.sleep(0.2)
    slow.release(_request("tab"))
    free.set()
    holder.join()
    starter.join()

    recording, response, *_ = answer["result"]
    assert recording is None and response["phase"] == "error"
    for browser in ("x", "y"):
        assert slow.start(None, _event(browser),
                          _request(browser))[1]["phase"] == "ready"
