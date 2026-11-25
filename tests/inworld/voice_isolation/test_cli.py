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
"""The `voice-isolate` command line.

Argument contract, file mapping, end-to-end on a tiny model.
"""

# pylint: disable=protected-access,redefined-outer-name,unused-argument

import math
import os
import stat
import sys
from unittest import mock

import pytest
import soundfile as sf
import torch

from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import cli

SR = 48000


def _sine(seconds, sr=SR, amp=0.3, hz=440.0):
    t = torch.arange(int(seconds * sr)) / sr
    return amp * torch.sin(2 * math.pi * hz * t)


@pytest.fixture
def ignores_case(tmp_path):
    """Whether tmp_path's file system finds a name in another case, as macOS
    and Windows do by default and Linux does not."""
    probe = tmp_path / "case-probe"
    probe.touch()
    found = (tmp_path / "CASE-PROBE").exists()
    probe.unlink()
    return found


@pytest.fixture
def umask():
    """Runs the test under umask 027, then restores the process's own."""
    previous = os.umask(0o027)
    yield 0o027
    os.umask(previous)


def _run(*argv):
    return cli.main(list(argv) + ["--device", "cpu"])


def _files(folder):
    return sorted(
        str(path.relative_to(folder))
        for path in folder.rglob("*")
        if path.is_file())


def test_parse_defaults():
    a = cli.parse_args(["in.wav", "out.wav"])
    assert (a.model, a.output_sr) == (
        "inworld/voice-isolator",
        48000,
    )
    assert (a.mode, a.lookahead, a.block_frames) == ("offline", 0, 4)
    assert (a.device, a.precision, a.float32) == (None, "auto", False)
    assert (a.chunk_seconds, a.overlap_seconds, a.shard, a.num_shards) == (
        None,
        None,
        0,
        1,
    )
    assert a.offline_passes is None
    assert a.background is None and not a.no_background
    assert cli.parse_args(["in.wav", "out.wav", "--offline-passes",
                           "2"]).offline_passes == 2


def test_parse_flags():
    a = cli.parse_args([
        "d",
        "o",
        "--model",
        "m.safetensors",
        "--mode",
        "streaming",
        "--lookahead",
        "2",
        "--block-frames",
        "2",
        "--window-frames",
        "40",
        "--crossfade-frames",
        "10",
        "--output-sr",
        "0",
        "--device",
        "cpu",
        "--precision",
        "bf16",
        "--chunk-seconds",
        "4",
        "--overlap-seconds",
        "1",
        "--shard",
        "2",
        "--num-shards",
        "3",
        "--float32",
    ])
    assert (a.model, a.output_sr, a.device) == (
        "m.safetensors",
        0,
        "cpu",
    )
    assert (a.mode, a.lookahead, a.block_frames, a.precision) == ("streaming",
                                                                  2, 2, "bf16")
    assert (a.window_frames, a.crossfade_frames) == (40, 10)
    assert a.float32
    assert (a.chunk_seconds, a.overlap_seconds, a.shard, a.num_shards) == (
        4.0,
        1.0,
        2,
        3,
    )
    with pytest.raises(SystemExit):
        cli.parse_args(["d", "o", "--mode", "live"])


def test_outputs_file_mode_is_one_pair_never_resumed(tmp_path):
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    assert cli._outputs(a, b, 0, 1) == ([(a, b, None)], False)
    folder = tmp_path / "outs"
    folder.mkdir()
    assert cli._outputs(a, str(folder), 0,
                        1) == ([(a, str(folder / "a.wav"), None)], False)


def test_outputs_directory_mode_recurses_filters_and_shards(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "sub").mkdir(parents=True)
    for name in ("a.wav", "b.flac", "c.txt", "sub/d.mp3", "sub/e.WAV"):
        (src / name).write_bytes(b"x")
    pairs, resume = cli._outputs(str(src), str(out), 0, 1)
    assert resume and out.is_dir()
    assert [os.path.relpath(dst, out) for _, dst, _ in pairs
           ] == ["a.wav", "b.wav", "sub/d.wav", "sub/e.wav"]
    first, _ = cli._outputs(str(src), str(out), 0, 2)
    second, _ = cli._outputs(str(src), str(out), 1, 2)
    assert len(first) + len(second) == 4 and not set(first) & set(second)


@pytest.mark.parametrize(
    "extra",
    [
        ["--alpha", "0.5"],
        ["--stage2", "s2.safetensors"],
    ],
)
def test_unsupported_restoration_flags_are_refused(extra, capsys):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(["in.wav", "out.wav"] + extra)

    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_directory_mode_resumes_without_loading_a_model(tmp_path, capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    audio_lib.save_audio(str(src / "a.wav"), _sine(0.1), SR, subtype="FLOAT")
    (out / "a.wav").write_bytes(b"done")
    (out / "a_background.wav").write_bytes(b"done background")
    assert cli.main([str(src), str(out), "--model", "no-such-checkpoint"]) == 0
    assert "0 to do" in capsys.readouterr().out
    assert (out / "a.wav").read_bytes() == b"done"
    assert (out / "a_background.wav").read_bytes() == b"done background"


def test_file_mode_replaces_an_existing_output(tmp_path, tiny_ckpt):
    inp, out = tmp_path / "in.wav", tmp_path / "out.wav"
    audio_lib.save_audio(str(inp), _sine(0.1), SR, subtype="FLOAT")
    out.write_bytes(b"an earlier run with other settings")
    assert _run(str(inp), str(out), "--model", tiny_ckpt, "--float32") == 0
    assert sf.info(str(out)).subtype == "FLOAT"


def test_an_interrupted_write_leaves_no_output(tmp_path, tiny_ckpt, monkeypatch,
                                               capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    audio_lib.save_audio(str(src / "a.wav"), _sine(0.2), SR, subtype="FLOAT")
    write = sf.write

    def full_disk(path, data, *args, **kwargs):
        # a partial file, then the error
        write(path, data[:len(data) // 2], *args, **kwargs)
        raise OSError("No space left on device")

    monkeypatch.setattr(sf, "write", full_disk)
    assert _run(str(src), str(out), "--model", tiny_ckpt) == 1
    assert not list(out.iterdir())  # nothing a resumed run would take as done
    monkeypatch.setattr(sf, "write", write)
    assert _run(str(src), str(out), "--model", tiny_ckpt) == 0
    assert sf.info(str(out / "a.wav")).frames == int(0.2 * SR)


def test_voice_only_pcm16_default_and_float32(tmp_path, tiny_ckpt):
    x = _sine(1.0)
    inp = tmp_path / "in.wav"
    audio_lib.save_audio(str(inp), x, SR, subtype="FLOAT")
    out = tmp_path / "out.wav"
    assert _run(str(inp), str(out), "--model", tiny_ckpt,
                "--no-background") == 0
    info = sf.info(str(out))
    assert (info.subtype, info.samplerate, info.frames) == (
        "PCM_16",
        SR,
        len(x),
    )
    y, _ = audio_lib.load_audio(str(out))
    # identity-at-init model, 16-bit quantisation
    assert torch.allclose(y, x, atol=2e-4)
    out32 = tmp_path / "out32.wav"
    assert _run(str(inp), str(out32), "--model", tiny_ckpt, "--float32",
                "--no-background") == 0
    assert sf.info(str(out32)).subtype == "FLOAT"
    y32, _ = audio_lib.load_audio(str(out32))
    assert torch.allclose(y32, x, atol=1e-4)
    assert not (tmp_path / "out_background.wav").exists()
    assert not (tmp_path / "out32_background.wav").exists()


def test_streaming_mode_runs_end_to_end(tmp_path, tiny_ckpt):
    x = _sine(0.3)
    inp = tmp_path / "in.wav"
    audio_lib.save_audio(str(inp), x, SR, subtype="FLOAT")
    out = tmp_path / "out.wav"
    assert _run(str(inp), str(out), "--model", tiny_ckpt, "--mode", "streaming",
                "--lookahead", "2", "--float32") == 0
    assert _run(str(inp), str(out), "--model", tiny_ckpt, "--mode", "streaming",
                "--lookahead", "2", "--window-frames", "40",
                "--crossfade-frames", "10", "--float32") == 0
    y, _ = audio_lib.load_audio(str(out))
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert not (tmp_path / "out_background.wav").exists()
    with pytest.raises(ValueError, match="lookahead_frames must be one of"):
        _run(str(inp), str(out), "--model", tiny_ckpt, "--mode", "streaming",
             "--lookahead", "3")


def test_pcm16_output_above_full_scale_is_scaled_not_clipped(
        tmp_path, tiny_ckpt):
    x = _sine(0.5, amp=1.5)
    inp, out = tmp_path / "hot.wav", tmp_path / "out.wav"
    audio_lib.save_audio(str(inp), x, SR, subtype="FLOAT")
    assert _run(str(inp), str(out), "--model", tiny_ckpt,
                "--no-background") == 0
    y, _ = audio_lib.load_audio(str(out))
    assert 0.999 <= float(y.abs().max()) <= 1.0
    assert torch.allclose(y / y.abs().max(), x / x.abs().max(),
                          atol=2e-3)  # same shape, no flat tops
    out32 = tmp_path / "out32.wav"
    assert _run(str(inp), str(out32), "--model", tiny_ckpt, "--float32",
                "--no-background") == 0
    y32, _ = audio_lib.load_audio(str(out32))
    # float output is untouched
    assert float(y32.abs().max()) == pytest.approx(1.5, abs=1e-3)


def test_output_rate_default_resamples_and_zero_keeps_input_rate(
        tmp_path, tiny_ckpt):
    x = _sine(0.5, sr=24000)
    inp = tmp_path / "in24.wav"
    audio_lib.save_audio(str(inp), x, 24000, subtype="FLOAT")
    out48, out24 = tmp_path / "o48.wav", tmp_path / "o24.wav"
    assert _run(str(inp), str(out48), "--model", tiny_ckpt) == 0
    assert (sf.info(str(out48)).samplerate, sf.info(str(out48)).frames) == (
        SR,
        24000,
    )
    assert (_run(str(inp), str(out24), "--model", tiny_ckpt, "--output-sr",
                 "0") == 0)
    assert (sf.info(str(out24)).samplerate, sf.info(str(out24)).frames) == (
        24000,
        12000,
    )


def test_directory_mode_keeps_basenames_and_reports_bad_files(
        tmp_path, tiny_ckpt, capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "sub").mkdir(parents=True)
    audio_lib.save_audio(str(src / "a.wav"), _sine(0.2), SR, subtype="FLOAT")
    sf.write(str(src / "sub" / "b.flac"), _sine(0.2).numpy(), SR)
    (src / "broken.wav").write_bytes(b"not audio at all")
    rc = _run(str(src), str(out), "--model", tiny_ckpt)
    # one bad file must not stop the batch, but the exit code says so
    assert rc == 1
    assert (out / "a.wav").exists() and (out / "sub" / "b.wav").exists()
    assert not (out / "broken.wav").exists()
    log = capsys.readouterr().out
    assert f"FAIL {src / 'broken.wav'}" in log and "DONE 2/3" in log


# each name also matches the decoy beside it as a glob pattern
@pytest.mark.parametrize("name, decoy", [("Takes [2024]", "Takes 2"),
                                         ("take*", "take two"),
                                         ("take?", "take2")])
def test_directory_input_names_are_literal(tmp_path, tiny_ckpt, capsys, name,
                                           decoy):
    src, out = tmp_path / name, tmp_path / "out"
    for folder, clip in ((src, "a.wav"), (tmp_path / decoy, "b.wav")):
        folder.mkdir()
        audio_lib.save_audio(str(folder / clip),
                             _sine(0.1),
                             SR,
                             subtype="FLOAT")

    result = _run(str(src), str(out), "--model", tiny_ckpt)

    assert result == 0
    assert "1 file(s), 1 to do" in capsys.readouterr().out
    assert _files(out) == ["a.wav", "a_background.wav"]


def test_directory_input_needs_an_output_directory(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out.wav"
    src.mkdir()
    (src / "a.wav").write_bytes(b"x")
    out.write_bytes(b"an unrelated file")

    with pytest.raises(SystemExit, match="needs an output directory"):
        _run(str(src), str(out), "--model", "not-a-model")

    assert out.read_bytes() == b"an unrelated file"


def test_colliding_or_self_overwriting_outputs_are_refused_before_any_work(
    tmp_path,):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for name in ("same.wav", "same.flac", "other.wav"):
        (src / name).write_bytes(b"x")
    with pytest.raises(SystemExit, match="would both be written to"):
        cli._outputs(str(src), str(out), 0, 1)
    # even across shards
    with pytest.raises(SystemExit, match="would both be written to"):
        cli._outputs(str(src), str(out), 1, 2)
    (src / "same.flac").unlink()
    with pytest.raises(SystemExit, match="outside the input directory"):
        cli._outputs(str(src), str(src), 0, 1)
    with pytest.raises(SystemExit, match="overwritten by an output"):
        cli._outputs(str(src / "other.wav"), str(src / "other.wav"), 0, 1)
    (src / "same.flac").write_bytes(b"x")
    # before any model loads
    with pytest.raises(SystemExit, match="would both be written to"):
        cli.main([str(src), str(out), "--model", "no-such-checkpoint"])


@pytest.mark.parametrize("output_rate", [0, 32000, 48000])
def test_offline_pair_preserves_mono_input_sum_and_levels(
        tmp_path, output_rate):
    sample_rate = 24000
    source = _sine(0.1, sr=sample_rate, amp=1.5)
    stereo = torch.stack([source, source * 0.8], dim=1).numpy()
    inp, out = tmp_path / "input.wav", tmp_path / "voice.wav"
    sf.write(inp, stereo, sample_rate, subtype="FLOAT")
    network = mock.create_autospec(cli.isolator.VoiceIsolator,
                                   instance=True,
                                   spec_set=True)

    def isolate(audio, sr, output_sample_rate, **_):
        return audio_lib.resample(audio, sr, output_sample_rate) * 1.25

    network.isolate.side_effect = isolate

    with mock.patch.object(cli.isolator.VoiceIsolator,
                           "load",
                           autospec=True,
                           return_value=network):
        result = _run(str(inp), str(out), "--output-sr", str(output_rate))

    assert result == 0
    background = tmp_path / "voice_background.wav"
    voice, sr = audio_lib.load_audio(str(out))
    noise, noise_sr = audio_lib.load_audio(str(background))
    mono, _ = audio_lib.load_audio(str(inp))
    expected = audio_lib.resample(mono, sample_rate, sr)
    assert sr == noise_sr == (output_rate or sample_rate)
    assert sf.info(out).subtype == sf.info(background).subtype == "FLOAT"
    assert float(voice.abs().max()) > 1.0
    assert torch.allclose(voice + noise, expected, atol=2e-7, rtol=1e-6)
    assert torch.allclose(noise, -expected * 0.25, atol=2e-7, rtol=1e-6)


def test_explicit_background_path_is_created(tmp_path, tiny_ckpt):
    inp, out = tmp_path / "input.wav", tmp_path / "voice.wav"
    background = tmp_path / "residuals" / "noise.wav"
    audio_lib.save_audio(str(inp), _sine(0.1), SR, subtype="FLOAT")

    result = _run(str(inp), str(out), "--model", tiny_ckpt, "--background",
                  str(background))

    assert result == 0
    assert sf.info(background).subtype == "FLOAT"
    assert not (tmp_path / "voice_background.wav").exists()


@pytest.mark.parametrize("extra, outputs", [
    pytest.param([], ["out.wav", "out_background.wav"], id="pair"),
    pytest.param(["--no-background"], ["out.wav"], id="voice-only"),
])
def test_outputs_get_the_permissions_of_any_new_file(tmp_path, tiny_ckpt, umask,
                                                     extra, outputs):
    inp = tmp_path / "in.wav"
    audio_lib.save_audio(str(inp), _sine(0.1), SR, subtype="FLOAT")

    result = _run(str(inp), str(tmp_path / "out.wav"), "--model", tiny_ckpt,
                  *extra)

    assert result == 0
    assert [
        oct(stat.S_IMODE((tmp_path / name).stat().st_mode)) for name in outputs
    ] == [oct(0o666 & ~umask)] * len(outputs)


@pytest.mark.parametrize("missing", ["voice", "background"])
def test_directory_resume_rebuilds_an_incomplete_pair(tmp_path, tiny_ckpt,
                                                      missing):
    src, out = tmp_path / "input", tmp_path / "output"
    src.mkdir()
    out.mkdir()
    x = _sine(0.1)
    audio_lib.save_audio(str(src / "a.wav"), x, SR, subtype="FLOAT")
    voice, background = out / "a.wav", out / "a_background.wav"
    (background if missing == "voice" else voice).write_bytes(b"old output")

    result = _run(str(src), str(out), "--model", tiny_ckpt)

    assert result == 0
    y, _ = audio_lib.load_audio(str(voice))
    residual, _ = audio_lib.load_audio(str(background))
    assert torch.allclose(y + residual, x, atol=1e-7)


def test_failed_background_commit_leaves_pair_resumable(tmp_path, tiny_ckpt):
    src, out = tmp_path / "input", tmp_path / "output"
    src.mkdir()
    out.mkdir()
    audio_lib.save_audio(str(src / "a.wav"), _sine(0.1), SR, subtype="FLOAT")
    background = out / "a_background.wav"
    background.write_bytes(b"old background without a voice")
    replace = os.replace

    def interrupted(source, destination):
        if os.fspath(destination) == str(background):
            raise OSError("Interrupted before background commit")
        return replace(source, destination)

    with mock.patch.object(os,
                           "replace",
                           autospec=True,
                           side_effect=interrupted):
        result = _run(str(src), str(out), "--model", tiny_ckpt)

    assert result == 1
    assert (out / "a.wav").is_file()
    assert not background.exists()
    assert not list(out.glob(".pair-*"))
    assert _run(str(src), str(out), "--model", tiny_ckpt) == 0
    assert sf.info(background).subtype == "FLOAT"


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "directory"])
def test_source_path_alias_is_refused_before_loading(tmp_path, alias):
    inp = tmp_path / "input.wav"
    inp.write_bytes(b"original recording")
    out = tmp_path / "output.wav"
    if alias == "symlink":
        out.symlink_to(inp)
    elif alias == "hardlink":
        os.link(inp, out)
    else:
        directory = tmp_path / "alias"
        directory.symlink_to(tmp_path, target_is_directory=True)
        out = directory / inp.name

    with pytest.raises(SystemExit, match="overwritten by an output"):
        _run(str(inp), str(out), "--model", "not-a-model")

    assert inp.read_bytes() == b"original recording"


def test_background_collision_is_checked_across_shards(tmp_path):
    src, out = tmp_path / "input", tmp_path / "output"
    src.mkdir()
    (src / "a.wav").write_bytes(b"first input")
    (src / "a_background.wav").write_bytes(b"second input")

    with pytest.raises(SystemExit, match="would both be written to"):
        _run(str(src), str(out), "--shard", "1", "--num-shards", "2", "--model",
             "not-a-model")

    assert not out.exists()


def test_explicit_background_cannot_alias_source_or_voice(tmp_path):
    inp, out = tmp_path / "input.wav", tmp_path / "voice.wav"
    inp.write_bytes(b"original")

    with pytest.raises(SystemExit, match="overwritten by an output"):
        _run(str(inp), str(out), "--background", str(inp))
    with pytest.raises(SystemExit, match="would both be written to"):
        _run(str(inp), str(out), "--background", str(out))

    assert inp.read_bytes() == b"original"


@pytest.mark.parametrize("inputs, argv, outputs, refusal", [
    pytest.param(["song.wav"],
                 ["song.wav", "clean.wav", "--background", "CLEAN.wav"],
                 ["CLEAN.wav", "clean.wav"],
                 "would both be written",
                 id="background-is-the-voice"),
    pytest.param(["in/a.wav", "in/A.flac"], ["in", "out"], [
        "out/A.wav", "out/A_background.wav", "out/a.wav", "out/a_background.wav"
    ],
                 "would both be written",
                 id="two-voices"),
    pytest.param(["in/x.wav", "in/x_Background.wav"], ["in", "out"], [
        "out/x.wav", "out/x_Background.wav", "out/x_Background_background.wav",
        "out/x_background.wav"
    ],
                 "would both be written",
                 id="a-voice-is-a-background"),
    pytest.param(["data/a.wav"], ["data", "DATA/out"],
                 ["DATA/out/a.wav", "DATA/out/a_background.wav"],
                 "outside the input",
                 id="output-inside-the-input"),
])
def test_names_differing_in_case_are_one_file_only_where_case_is_ignored(
        tmp_path, monkeypatch, tiny_ckpt, ignores_case, inputs, argv, outputs,
        refusal):
    monkeypatch.chdir(tmp_path)
    for name in inputs:
        os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
        sf.write(name, _sine(0.1).numpy(), SR)

    try:
        outcome = _run(*argv, "--model", tiny_ckpt)
    except SystemExit as refused:
        outcome = refused.code

    if ignores_case:  # refused before anything is written
        assert refusal in str(outcome)
        assert _files(tmp_path) == sorted(inputs)
    else:
        assert outcome == 0
        assert _files(tmp_path) == sorted(inputs + outputs)


@pytest.mark.parametrize(
    "extra, message",
    [(["--mode", "streaming", "--background", "noise.wav"], "only in offline"),
     (["--output-sr", "-1"], "must be positive")])
def test_invalid_output_options_fail_before_loading(tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        _run(str(tmp_path / "input.wav"), str(tmp_path / "out.wav"), *extra)


def test_paired_outputs_require_wav_files(tmp_path):
    with pytest.raises(SystemExit, match="must be WAV files"):
        _run(str(tmp_path / "input.wav"), str(tmp_path / "out.flac"))


@pytest.mark.parametrize("shard,num_shards", [(0, 0), (2, 2), (-1, 2)])
def test_shard_arguments_outside_their_range_are_refused(
        tmp_path, shard, num_shards):
    with pytest.raises(SystemExit, match="--shard must be in"):
        cli.main([
            str(tmp_path),
            str(tmp_path / "out"),
            "--shard",
            str(shard),
            "--num-shards",
            str(num_shards),
        ])


@pytest.mark.parametrize(
    "platform,device,env,cores,expected",
    [
        ("darwin", "cpu", {}, 12, 2),
        ("darwin", "cpu", {}, 1, None),  # never more threads than cores
        ("darwin", "cpu", {
            "OMP_NUM_THREADS": "8"
        }, 12, None),
        ("darwin", "mps", {}, 12, None),
        ("linux", "cpu", {}, 12, None),
    ])
def test_cpu_threads_are_capped_on_macos_only(monkeypatch, platform, device,
                                              env, cores, expected):
    calls = []
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(torch, "get_num_threads", lambda: cores)
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    cli._limit_macos_cpu_threads(torch.device(device))
    assert calls == ([] if expected is None else [expected])


@pytest.mark.parametrize("suffix", ["/", "/sub"])
def test_an_output_directory_that_cannot_be_made_is_one_line_of_error(
        tmp_path, suffix):
    src, out = tmp_path / "in", tmp_path / "out.wav"
    src.mkdir()
    (src / "a.wav").write_bytes(b"x")
    out.write_bytes(b"an unrelated file")

    with pytest.raises(SystemExit, match="Cannot create the output directory"):
        _run(str(src), str(out) + suffix, "--model", "not-a-model")

    assert out.read_bytes() == b"an unrelated file"


@pytest.mark.parametrize("inputs, argv", [
    (["song.wav"], ["song.wav", "clean.wav", "--background", "CLEAN.wav"]),
    (["in/a.wav", "in/A.flac"], ["in", "out"]),
    (["in/x.wav", "in/x_Background.wav"], ["in", "out"]),
])
def test_names_differing_in_case_are_refused_where_a_folder_ignores_case(
        tmp_path, monkeypatch, inputs, argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_ignores_case", lambda folder: True)
    for name in inputs:
        os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
        sf.write(name, _sine(0.1).numpy(), SR)

    with pytest.raises(SystemExit, match="would both be written"):
        _run(*argv, "--model", "not-a-model")

    assert _files(tmp_path) == sorted(inputs)


def test_the_case_probe_agrees_with_the_file_system(tmp_path, ignores_case):
    assert cli._ignores_case(str(tmp_path)) == ignores_case


def test_an_empty_folder_is_probed_with_a_file_of_its_own(
        tmp_path, monkeypatch, ignores_case):
    folder = tmp_path / "empty"
    folder.mkdir()
    monkeypatch.setattr(cli, "_probe_paths", lambda path: iter(()))

    found = cli._ignores_case(str(folder))

    assert found == ignores_case
    assert not any(folder.iterdir())


def test_a_folder_that_cannot_be_probed_counts_as_ignoring_case(
        tmp_path, monkeypatch):

    def read_only(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(cli, "_probe_paths", lambda path: iter(()))
    monkeypatch.setattr(cli.tempfile, "mkstemp", read_only)

    assert cli._ignores_case(str(tmp_path))


def test_a_hard_link_in_another_case_is_not_taken_for_ignored_case(
        tmp_path, ignores_case):
    if ignores_case:
        pytest.skip("needs a file system that tells case apart")
    folder = tmp_path / "links"
    folder.mkdir()
    (folder / "readme").write_bytes(b"x")
    os.link(folder / "readme", folder / "README")

    assert not cli._ignores_case(str(folder))
