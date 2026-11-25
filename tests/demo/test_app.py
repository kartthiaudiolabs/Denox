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
"""Two fixed-mode demo models, playback levels and browser file boundaries."""

import asyncio
import io
import json
import pathlib
import sys
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch

from inworld import voice_isolation

pytest.importorskip("gradio")

# pylint: disable=wrong-import-position
import fastapi
from fastapi import testclient
from gradio import route_utils
import gradio as gr

import app
import example_audio
import online

# pylint: enable=wrong-import-position


@pytest.fixture(name="models")
def loaded_models(tiny):
    """Two independent tiny networks behind the actual public facade."""
    offline = voice_isolation.VoiceIsolator(tiny.identity(),
                                            device="cpu",
                                            precision="fp32",
                                            mode="offline")
    realtime = voice_isolation.VoiceIsolator(tiny.identity(),
                                             device="cpu",
                                             precision="fp32",
                                             mode="streaming",
                                             lookahead_frames=2,
                                             block_frames=4)
    return app.DemoModels(offline, realtime)


def test_models_load_once_in_fixed_modes_with_no_shared_network(models):
    with mock.patch.object(voice_isolation.VoiceIsolator,
                           "load",
                           autospec=True,
                           side_effect=[models.offline,
                                        models.realtime]) as load:
        loaded = app.load_models("offline.safetensors", "online.safetensors",
                                 "cpu", "fp32")
        app.build_demo(loaded)
        app.build_demo(loaded)

    assert load.call_args_list == [
        mock.call("offline.safetensors",
                  device="cpu",
                  mode="offline",
                  precision="fp32"),
        mock.call("online.safetensors",
                  device="cpu",
                  mode="streaming",
                  precision="fp32",
                  lookahead_frames=2,
                  block_frames=4),
    ]
    assert loaded.offline is models.offline
    assert loaded.realtime is models.realtime
    assert (loaded.offline_name,
            loaded.realtime_name) == ("offline.safetensors",
                                      "online.safetensors")
    assert loaded.offline.model is not loaded.realtime.model
    assert loaded.offline_lock is not loaded.realtime_lock


@pytest.mark.parametrize("message", ["checkpoint only streams", "bad checksum"])
def test_checkpoint_errors_fail_startup_without_mode_fallback(message):
    with mock.patch.object(voice_isolation.VoiceIsolator,
                           "load",
                           autospec=True,
                           side_effect=ValueError(message)) as load:
        with pytest.raises(ValueError, match=message):
            app.load_models("wrong.safetensors", "online.safetensors")

    load.assert_called_once_with("wrong.safetensors",
                                 device=None,
                                 mode="offline",
                                 precision="auto")


@pytest.mark.parametrize("failure", ["mode", "shared network"])
def test_startup_refuses_wrong_modes_and_shared_networks(models, failure):
    if failure == "mode":
        realtime = models.offline
        message = "offline and a realtime"
    else:
        realtime = voice_isolation.VoiceIsolator(models.offline.model,
                                                 device="cpu",
                                                 mode="streaming",
                                                 lookahead_frames=2)
        message = "separate model instance"

    with mock.patch.object(voice_isolation.VoiceIsolator,
                           "load",
                           autospec=True,
                           side_effect=[models.offline, realtime]):
        with pytest.raises(ValueError, match=message):
            app.load_models()


def test_pcm16_preserves_quiet_levels_and_clips_without_peak_normalizing():
    samples = np.array([0.05, 0.5, -1.5, 1.5], dtype=np.float32)

    played = app.pcm16(samples)

    assert played.dtype == np.int16
    assert played.tolist() == [
        round(0.05 * 32767),
        round(0.5 * 32767), -32767, 32767
    ]
    assert app.pcm16(np.zeros(0, dtype=np.float32)).shape == (0,)


@pytest.mark.parametrize("input_level,voice_level", [(0.1, 0.025), (0.5, 1.25),
                                                     (-0.5, 1.25)])
def test_offline_residual_uses_original_levels_before_playback_clipping(
        models, input_level, voice_level):
    audio = np.full(2400, input_level, dtype=np.float32)
    voice = np.full(audio.shape, voice_level, dtype=np.float32)
    background = audio - voice

    with (mock.patch.object(models.offline,
                            "isolate",
                            autospec=True,
                            return_value=voice) as
          isolate, mock.patch.object(models.realtime, "isolate",
                                     autospec=True) as realtime,
          mock.patch.object(app,
                            "spectrogram_png",
                            autospec=True,
                            side_effect=["voice-image",
                                         "background-image"]) as image):
        output = app.run_offline((48000, audio), models)

    preview, voice_image, residual, background_image, timing = output
    assert isolate.call_count == 1
    np.testing.assert_array_equal(isolate.call_args.args[0], audio)
    assert isolate.call_args.args[1] == 48000
    assert isolate.call_args.kwargs == {"output_sample_rate": 48000}
    realtime.assert_not_called()
    assert preview[0] == residual[0] == 48000
    np.testing.assert_array_equal(preview[1], app.pcm16(voice))
    np.testing.assert_array_equal(residual[1], app.pcm16(background))
    np.testing.assert_array_equal(image.call_args_list[0].args[0], voice)
    np.testing.assert_array_equal(image.call_args_list[1].args[0], background)
    assert (voice_image, background_image) == ("voice-image",
                                               "background-image")
    assert "processed in" in timing
    if max(abs(voice_level), abs(float(background[0]))) <= 1:
        rebuilt = (preview[1].astype(np.float32) + residual[1]) / 32767
        np.testing.assert_allclose(rebuilt, audio, rtol=0, atol=1 / 32767)


def test_offline_aligns_integer_stereo_before_background_subtraction(models):
    rate = 22050
    left = np.arange(-10000, 10000, 20, dtype=np.int16)
    stereo = np.column_stack((left, np.flip(left) // 3))
    mono = (stereo.astype(np.float32) / np.iinfo(stereo.dtype).max).mean(axis=1)
    aligned = voice_isolation.resample(torch.from_numpy(mono), rate,
                                       48000).numpy()
    voice = aligned * 0.3

    with mock.patch.object(models.offline,
                           "isolate",
                           autospec=True,
                           return_value=voice) as isolate:
        output = app.run_offline((rate, stereo), models)

    np.testing.assert_array_equal(isolate.call_args.args[0], mono)
    assert isolate.call_args.args[1] == rate
    assert isolate.call_args.kwargs == {"output_sample_rate": 48000}
    assert output[0][0] == output[2][0] == 48000
    np.testing.assert_array_equal(output[0][1], app.pcm16(voice))
    np.testing.assert_array_equal(output[2][1], app.pcm16(aligned - voice))


@pytest.mark.parametrize("audio", [
    None, (48000, np.zeros(0)), (48000, np.array([np.nan])),
    (48000, np.zeros((2, 2, 2)))
])
def test_invalid_recordings_fail_before_inference(models, audio):
    with mock.patch.object(models.offline, "isolate", autospec=True) as isolate:
        with pytest.raises(gr.Error):
            app.run_offline(audio, models)

    isolate.assert_not_called()


def test_offline_rejects_misaligned_output_and_real_model_runs(models):
    audio = np.full(2400, 0.05, dtype=np.float32)
    with mock.patch.object(models.offline,
                           "isolate",
                           autospec=True,
                           return_value=audio[:-1]):
        with pytest.raises(gr.Error, match="same length"):
            app.run_offline((48000, audio), models)

    result = app.run_offline((48000, audio), models)

    assert result[0][1].shape == result[2][1].shape == audio.shape
    np.testing.assert_allclose(result[0][1].astype(float) / 32767,
                               audio,
                               atol=2 / 32767,
                               rtol=0)


def test_two_tabs_wire_online_events_to_per_browser_state(models):
    demo = app.build_demo(models)
    components = {item["id"]: item for item in demo.config["components"]}
    events = demo.config["dependencies"]
    tabs = [
        item["props"]["label"]
        for item in components.values()
        if item["type"] == "tabitem"
    ]
    state = next(
        item for item in components.values() if item["type"] == "state")
    studio = next(item for item in components.values()
                  if item["props"].get("elem_id") == "live-studio")
    player = next(item for item in components.values()
                  if item["props"].get("elem_id") == "online-result")
    original = next(item for item in components.values()
                    if item["props"].get("elem_id") == "online-input")

    assert tabs == ["Offline", "Online"]
    assert demo.blocks[state["id"]].value is None
    assert demo.blocks[state["id"]].delete_callback is online.close_recording
    assert demo.blocks[state["id"]].time_to_live == 120
    for name, method, outputs in [
        ("record_start", "start",
         [state["id"], studio["id"], player["id"], original["id"]]),
        ("audio_chunk", "push", [state["id"], studio["id"]]),
        ("record_finish", "finish",
         [state["id"], studio["id"], player["id"], original["id"]]),
    ]:
        event = next(
            item for item in events
            if [studio["id"], name] in [list(t) for t in item["targets"]])
        assert event["inputs"] == [state["id"]]
        assert event["outputs"] == outputs
        function = demo.fns[event["id"]]
        assert function.fn.__name__ == method
        assert function.fn.__self__.model is models.realtime
        assert function.fn.__self__.model_lock is models.realtime_lock
        assert event["api_visibility"] == "private"
        assert event["queue"] is False
        if name == "audio_chunk":
            # A response watcher can send the next chunk before Gradio marks
            # the previous event complete; "once" silently drops that chunk.
            assert event["trigger_mode"] == "multiple"
    assert not any(item["type"] == "dataset" for item in components.values())
    unload = next(item for item in events
                  if any(target[1] == "unload" for target in item["targets"]))
    assert demo.fns[unload["id"]].fn.__name__ == "release"


def test_a_new_input_cancels_its_isolation_without_waiting_for_the_model(
        models):
    demo = app.build_demo(models)
    events = demo.config["dependencies"]
    click = next(
        event for event in events if event["api_name"] == "isolate_offline")
    changes = [
        event for event in events
        if any(target[1] == "change" for target in event["targets"])
    ]

    preview = demo.fns[changes[0]["id"]]

    assert preview.fn is app.input_changed
    assert preview.concurrency_id != demo.fns[click["id"]].concurrency_id
    assert any(click["id"] in event["cancels"] for event in changes)


def test_custom_examples_show_only_listed_audio(models, example_directory):
    (example_directory / "private-note.txt").write_text("PRIVATE_NOTE",
                                                        encoding="utf-8")
    examples = example_audio.load_examples(example_directory)

    demo = app.build_demo(models, examples)

    config = json.dumps(demo.config, default=str)
    assert "synthetic_silence" in config
    assert "PRIVATE_NOTE" not in config
    assert any(item["type"] == "dataset" for item in demo.config["components"])


def test_offline_downloads_keep_levels_and_input_changes_clear_results(
        models, tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "cache"))
    source = tmp_path / "input.wav"
    sf.write(source,
             np.full(2400, 3277, dtype=np.int16),
             48000,
             subtype="PCM_16")
    voice = np.full(2400, 0.025, dtype=np.float32)
    demo = app.build_demo(models)
    click = next(event for event in demo.config["dependencies"]
                 if event["api_name"] == "isolate_offline")
    change = next(event for event in demo.config["dependencies"]
                  if any(target[1] == "change" for target in event["targets"]))
    web_app = gr.mount_gradio_app(fastapi.FastAPI(), demo, path="/")

    with (testclient.TestClient(web_app) as client,
          mock.patch.object(models.offline,
                            "isolate",
                            autospec=True,
                            return_value=voice)):
        with source.open("rb") as recording:
            upload = client.post(
                demo.config.get("api_prefix", "") + "/upload",
                files={"files": (source.name, recording, "audio/wav")})
        assert upload.status_code == 200
        audio = {"path": upload.json()[0], "meta": {"_type": "gradio.FileData"}}
        result = asyncio.run(demo.process_api(click["id"], [audio]))["data"]
        for metadata, level in ((result[0], 0.025), (result[2],
                                                     3277 / 32767 - 0.025)):
            assert pathlib.Path(metadata["path"]).is_relative_to(
                pathlib.Path(demo.GRADIO_CACHE))
            response = client.get(metadata["url"])
            assert response.status_code == 200
            samples, rate = sf.read(io.BytesIO(response.content), dtype="int16")
            assert rate == 48000
            assert samples.tolist() == [round(level * 32767)] * len(voice)
        changed = asyncio.run(demo.process_api(change["id"], [audio]))["data"]
        cleared = asyncio.run(demo.process_api(change["id"], [None]))["data"]

    assert changed[0] is not None
    assert changed[1:] == [None, None, None, None, ""]
    assert cleared == [None, None, None, None, None, ""]


@pytest.mark.parametrize("bundled", [True, False])
def test_main_loads_once_and_allows_only_named_example_files(
        models, tmp_path, monkeypatch, bundled):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOICE_ISOLATION_EXAMPLES", raising=False)
    monkeypatch.setattr(sys, "argv",
                        ["app.py"] if bundled else ["app.py", "--examples", ""])
    demo = mock.create_autospec(gr.Blocks, instance=True, spec_set=True)
    demo.launch.return_value = (None, "http://localhost", None)
    examples = example_audio.load_examples(
        example_audio.DEFAULT_DIRECTORY) if bundled else None

    calls = mock.Mock()
    with (mock.patch.object(app,
                            "load_models",
                            autospec=True,
                            return_value=models) as
          load, mock.patch.object(app, "warm_up", autospec=True) as warm,
          mock.patch.object(app, "build_demo", autospec=True,
                            return_value=demo) as build):
        calls.attach_mock(load, "load")
        calls.attach_mock(warm, "warm_up")
        calls.attach_mock(build, "build_demo")
        app.main()

    load.assert_called_once_with(app.registry.REPO_ID, app.registry.REPO_ID,
                                 None, "auto")
    warm.assert_called_once_with(models)
    build.assert_called_once_with(models, examples)
    assert [call[0] for call in calls.mock_calls
           ] == ["load", "warm_up", "build_demo"]
    launch = demo.launch.call_args.kwargs
    assert launch["allowed_paths"] == ([row[0] for row in examples.rows]
                                       if examples else [])
    assert "api" not in launch["footer_links"]
    assert ".show-api-divider" in launch["css"]
    demo.block_thread.assert_called_once_with()


@pytest.mark.parametrize("failure", ["no audio", "non-audio", "bad auth"])
def test_main_rejects_invalid_startup_before_loading_models(
        example_directory, monkeypatch, failure):
    path = example_directory / "synthetic_silence.wav"
    arguments = ["app.py", "--share", "--examples", str(example_directory)]
    if failure == "no audio":
        path.unlink()
    elif failure == "non-audio":
        path.write_bytes(b'{"private": "a disguised sidecar"}')
    else:
        arguments.extend(["--auth", "invalid"])
    monkeypatch.setattr(sys, "argv", arguments)

    with mock.patch.object(app, "load_models", autospec=True) as load:
        with pytest.raises(SystemExit):
            app.main()

    load.assert_not_called()


@pytest.mark.parametrize("variable",
                         ["GRADIO_TEMP_DIR", "GRADIO_EXAMPLES_CACHE"])
@pytest.mark.parametrize("inside_cache", [False, True])
def test_main_rejects_examples_in_gradio_cache(example_directory, monkeypatch,
                                               capsys, variable, inside_cache):
    cache = example_directory.parent if inside_cache else example_directory
    monkeypatch.setenv(variable, str(cache))
    monkeypatch.setattr(
        sys, "argv",
        ["app.py", "--examples", str(example_directory)])

    with mock.patch.object(app, "load_models", autospec=True) as load:
        with pytest.raises(SystemExit) as error:
            app.main()

    assert error.value.code == 2
    assert "outside Gradio" in capsys.readouterr().err
    load.assert_not_called()


def test_routes_allow_examples_but_deny_private_sidecars(
        models, example_directory, tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    private_note = example_directory / "private-note.txt"
    private_note.write_text("Private test note.", encoding="utf-8")
    nested = example_directory / "private"
    nested.mkdir()
    nested_note = nested / "note.txt"
    nested_note.write_text("Nested test note.", encoding="utf-8")
    nested_audio = nested / "unlisted.wav"
    nested_audio.write_bytes(
        (example_directory / "synthetic_silence.wav").read_bytes())
    linked_note = example_directory / "linked-note.txt"
    linked_note.symlink_to(private_note)
    examples = example_audio.load_examples(example_directory)
    demo = app.build_demo(models, examples)
    web_app = gr.mount_gradio_app(
        fastapi.FastAPI(),
        demo,
        path="/",
        allowed_paths=[row[0] for row in examples.rows])
    prefix = demo.config.get("api_prefix", "") + "/file="

    with testclient.TestClient(web_app) as client:
        audio = client.get(prefix + examples.rows[0][0])
        denied = [
            client.get(prefix + str(path))
            for path in (private_note, nested_note, nested_audio, linked_note)
        ]

    assert audio.status_code == 200
    assert audio.content == pathlib.Path(examples.rows[0][0]).read_bytes()
    assert all(response.status_code == 403 for response in denied)


@pytest.mark.parametrize("device,kind,name,detail", [
    ("cpu", "cpu", "CPU", "Apple M4 Max · 3 threads"),
    ("mps", "metal", "Metal GPU", "Apple M4 Max"),
    ("cuda", "gpu", "CUDA GPU", "NVIDIA L4"),
])
def test_devices_are_named_for_people(monkeypatch, device, kind, name, detail):
    monkeypatch.setattr(app, "processor_name", lambda: "Apple M4 Max")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda device: "NVIDIA L4")

    info = app.describe_device(device)

    assert info == app.DeviceInfo(kind, name, detail)


def test_processor_name_falls_back_to_the_machine_type(monkeypatch):

    def unavailable(*args, **kwargs):
        raise OSError("no sysctl")

    monkeypatch.setattr(app.subprocess, "run", unavailable)
    monkeypatch.setattr(app.platform, "processor", lambda: "")
    monkeypatch.setattr(app.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(app.sys, "platform", "darwin")
    app.processor_name.cache_clear()

    try:
        name = app.processor_name()
    finally:
        app.processor_name.cache_clear()

    assert name == "arm64"


def test_the_header_names_the_inference_device(models, monkeypatch):
    monkeypatch.setattr(app, "processor_name", lambda: "Chip <b>&</b>")

    header = app.header_html(models)
    demo = app.build_demo(models)

    assert header.count('class="device-badge device-cpu"') == 1
    assert "Running on <strong>CPU</strong>" in header
    assert "Chip &lt;b&gt;&amp;&lt;/b&gt;" in header and "<b>" not in header
    assert "device-badge" in json.dumps(demo.config, default=str)


def test_models_on_two_devices_get_one_badge_each(models, monkeypatch):
    infos = {
        "offline": app.DeviceInfo("cpu", "CPU", "Apple M4 Max · 4 threads"),
        "realtime": app.DeviceInfo("metal", "Metal GPU", "Apple M4 Max"),
    }
    monkeypatch.setattr(
        app, "describe_device", lambda device: infos[
            "offline" if device is models.offline.device else "realtime"])
    monkeypatch.setattr(models.realtime, "device", torch.device("meta"))

    header = app.header_html(models)

    assert "Offline: Running on <strong>CPU</strong>" in header
    assert "Online: Running on <strong>Metal GPU</strong>" in header


def test_offline_timing_names_the_device(models):
    audio = np.full(2400, 0.05, dtype=np.float32)

    timing = app.run_offline((48000, audio), models)[-1]

    assert timing.endswith("× real time on CPU")


def test_input_spectrograms_share_the_outputs_frequency_axis():
    rate = 16000
    audio = (0.3 * np.sin(2 * np.pi * 3000 * np.arange(rate) / rate)).astype(
        np.float32)
    at_48k = voice_isolation.resample(torch.from_numpy(audio), rate,
                                      48000).numpy()

    image = app.spectrogram_png(audio, rate)

    assert image.shape == app.spectrogram_png(at_48k).shape
    rows = image.shape[0]
    loudest = int(np.argmax(image.astype(int).sum(axis=(1, 2))))
    # 3 kHz sits an eighth of the way up a 0-24 kHz axis (row 0 is the top).
    assert abs((rows - 1 - loudest) / (rows - 1) - 3000 / 24000) < 0.01


def test_warm_up_runs_both_models_through_a_crossfade(models):
    with (mock.patch.object(models.offline,
                            "isolate",
                            wraps=models.offline.isolate) as isolate,
          mock.patch.object(models.realtime,
                            "stream",
                            wraps=models.realtime.stream) as stream):
        app.warm_up(models)

    isolate.assert_called_once()
    assert isolate.call_args.args[1] == 48000
    stream.assert_called_once_with(window_frames=8, crossfade_frames=4)


def test_two_models_on_the_apple_gpu_share_one_lock(models, monkeypatch):
    monkeypatch.setattr(models.offline, "device", torch.device("mps"))
    monkeypatch.setattr(models.realtime, "device", torch.device("mps"))

    shared = app.DemoModels(models.offline, models.realtime)

    assert shared.realtime_lock is shared.offline_lock
    assert models.realtime_lock is not models.offline_lock


def test_the_cache_sweep_keeps_the_samples(models, example_directory):
    examples = example_audio.load_examples(example_directory)
    demo = app.build_demo(models, examples)
    copies = set().union(*demo.temp_file_sets)

    # Gradio's hourly sweep, with every file old enough to go.
    route_utils.delete_files_created_by_app(demo, 0)

    assert copies
    assert all(pathlib.Path(path).exists() for path in copies)


@pytest.mark.parametrize("name,subtype,file_format", [
    ("extensible.wav", "FLOAT", "WAVEX"),
    ("double.wav", "DOUBLE", "WAV"),
    ("lossless.flac", "PCM_24", "FLAC"),
])
def test_uploads_decode_at_their_own_level(tmp_path, name, subtype,
                                           file_format):
    path = tmp_path / name
    tone = (0.5 * np.sin(np.arange(9600) / 7)).astype(np.float32)
    sf.write(path,
             np.column_stack((tone, tone)),
             96000,
             subtype=subtype,
             format=file_format)

    rate, audio = app.load_input(str(path))

    assert rate == 96000
    np.testing.assert_allclose(audio, tone, atol=1e-6)


def test_offline_refuses_inputs_longer_than_its_limit(tmp_path, monkeypatch,
                                                      models):
    monkeypatch.setattr(app, "MAX_OFFLINE_SECONDS", 1)
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(96000, dtype=np.float32), 48000)

    with mock.patch.object(models.offline, "isolate", autospec=True) as isolate:
        with pytest.raises(gr.Error, match="takes up to"):
            app.run_offline(str(path), models)

    isolate.assert_not_called()


def test_rates_sharing_few_factors_with_48k_are_resampled_first(models):
    rate = 7993
    audio = np.random.RandomState(3).normal(0, 0.1, rate).astype(np.float32)

    with mock.patch.object(models.offline,
                           "isolate",
                           wraps=models.offline.isolate) as isolate:
        voice, _, background, _, _ = app.run_offline((rate, audio), models)

    assert isolate.call_args.args[1] == 48000
    assert voice[1].shape == background[1].shape
    assert abs(voice[1].size - 48000) <= 1


def test_an_unusable_input_still_clears_the_previous_results(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), 48000)

    with mock.patch.object(app.gr, "Warning", autospec=True) as warning:
        cleared = app.input_changed(str(path))

    assert cleared == (None, None, None, None, None, "")
    warning.assert_called_once()


@pytest.mark.parametrize("spec,mode,name", [
    ("inworld/voice-isolator", "offline", "voice-isolator-v1"),
    ("inworld/voice-isolator", "streaming", "voice-isolator-v1-realtime"),
    ("voice-isolator-v1-realtime", "streaming", "voice-isolator-v1-realtime"),
    ("hf://org/repo/sub/realtime.safetensors@main", "streaming",
     "sub/realtime.safetensors"),
    ("/checkpoints/run7/ema.safetensors", "offline", "ema.safetensors"),
])
def test_checkpoints_are_named_as_released_or_by_file(spec, mode, name):
    assert app.model_name(spec, mode) == name


def test_each_tab_names_its_own_model(models):
    named = app.DemoModels(models.offline,
                           models.realtime,
                           offline_name="offline-ckpt.safetensors",
                           realtime_name="live-ckpt.safetensors")
    lookahead_ms = round(models.realtime.lookahead_frames *
                         models.realtime.frame_seconds * 1000)

    demo = app.build_demo(named)

    texts = [
        item["props"].get("value") or ""
        for item in demo.config["components"]
        if item["type"] == "markdown"
    ]
    offline_intro = next(text for text in texts if "offline-ckpt" in text)
    online_intro = next(text for text in texts if "live-ckpt" in text)
    assert "different model" in offline_intro
    assert "live-ckpt" not in offline_intro
    assert "separate model" in online_intro
    assert "offline-ckpt" not in online_intro
    assert f"{lookahead_ms} ms ahead" in online_intro


def test_a_superseded_isolation_gives_way_without_running(models):
    audio = np.full(2400, 0.05, dtype=np.float32)

    with mock.patch.object(models.offline, "isolate", autospec=True) as isolate:
        result = app.run_offline((48000, audio),
                                 models,
                                 superseded=lambda: True)

    isolate.assert_not_called()
    assert list(result) == [gr.skip()] * 5


def test_a_newer_request_supersedes_only_its_own_browsers():
    latest = app.LatestRequests()
    first = latest.begin(gr.Request(session_hash="tab"))
    other = latest.begin(gr.Request(session_hash="other tab"))

    second = latest.begin(gr.Request(session_hash="tab"))
    latest.forget(gr.Request(session_hash="other tab"))

    assert first() and not second()
    assert other()
