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
"""VoiceIsolator on CPU: rates, channels, modes, checkpoints and precision."""

# pylint: disable=redefined-outer-name
import dataclasses
import json
import os

import numpy as np
import pytest
import torch

from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import isolator as isolator_lib
from inworld.voice_isolation import registry


@pytest.fixture
def isolator(dual_model):
    return isolator_lib.VoiceIsolator(dual_model, device="cpu")


def _training_checkpoint(m, path, prefix="isolator.", extra=None):
    """Writes a checkpoint the way training does: config, prefixed state."""
    state = {prefix + k: v for k, v in m.state_dict().items()}
    state.update(extra or {})
    torch.save(
        {
            "config": {
                "modeling": {
                    "compact_family": m.config.to_dict()
                }
            },
            "model": state,
        }, path)
    return str(path)


@pytest.mark.parametrize("sr", [16000, 24000, 44100, 48000])
def test_sample_rate_round_trip(isolator, sr):
    torch.manual_seed(0)
    x = torch.randn(int(0.13 * sr)) * 0.3
    y = isolator.isolate(x, sr)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_numpy_in_numpy_out_and_output_rate(isolator):
    x = (np.random.RandomState(0).randn(2400) * 0.3).astype(np.float32)
    y = isolator.isolate(x, 24000, output_sample_rate=48000)
    assert isinstance(y, np.ndarray)
    assert y.shape == (4800,)


def test_identity_model_passes_48k_audio_through(identity_model):
    vi = isolator_lib.VoiceIsolator(identity_model, device="cpu")
    torch.manual_seed(0)
    x = torch.randn(4800) * 0.3
    assert torch.equal(vi.isolate(x, 48000), x)
    vi.mode = "streaming"
    assert torch.equal(vi.isolate(x, 48000), x)


def test_the_level_is_left_alone(identity_model):
    vi = isolator_lib.VoiceIsolator(identity_model, device="cpu")
    x = torch.randn(4800) * 0.001
    assert torch.equal(vi.isolate(x, 48000), x)


def test_stereo_and_transposed_inputs_are_downmixed(isolator):
    torch.manual_seed(0)
    mono = torch.randn(4800) * 0.3
    ref = isolator.isolate(mono, 48000)
    assert torch.allclose(isolator.isolate(torch.stack([mono, mono]), 48000),
                          ref,
                          atol=1e-5)
    assert torch.allclose(isolator.isolate(torch.stack([mono, mono], 1), 48000),
                          ref,
                          atol=1e-5)


def test_modes_and_lookahead(isolator):
    assert isolator.mode == "offline" and isolator.lookahead_choices == (0, 2)
    assert isolator.frame_seconds == pytest.approx(16 / 48000)
    torch.manual_seed(0)
    x = torch.randn(4800) * 0.3
    offline = isolator.isolate(x, 48000)
    isolator.mode = "streaming"
    assert isolator.model.lookahead_frames == 0
    k0 = isolator.isolate(x, 48000)
    isolator.lookahead_frames = 2
    assert isolator.model.lookahead_frames == 2
    k2 = isolator.isolate(x, 48000)
    assert not torch.allclose(offline, k0, atol=1e-4)
    assert not torch.allclose(k0, k2, atol=1e-4)
    # the block sets throughput, not latency
    assert isolator.latency_samples() == 64 + 16 * 2
    assert isolator.latency_samples(block_frames=1) == 64 + 16 * 2
    with pytest.raises(ValueError, match="lookahead_frames must be one of"):
        isolator.lookahead_frames = 1
    with pytest.raises(ValueError, match="mode must be one of"):
        isolator.mode = "live"
    isolator.mode = "offline"
    assert torch.equal(isolator.isolate(x, 48000), offline)


def test_an_offline_only_checkpoint_has_no_streaming_mode(offline_model):
    vi = isolator_lib.VoiceIsolator(offline_model, device="cpu")
    assert vi.lookahead_choices == ()
    with pytest.raises(ValueError, match="no streaming mode"):
        vi.mode = "streaming"
    with pytest.raises(ValueError, match="no streaming mode"):
        isolator_lib.VoiceIsolator(offline_model,
                                   device="cpu",
                                   mode="streaming")


def test_the_stream_needs_streaming_mode(isolator):
    with pytest.raises(RuntimeError, match="mode='streaming'"):
        isolator.stream()
    isolator.mode = "streaming"
    stream = isolator.stream(block_frames=2)
    x = torch.randn(2000) * 0.3
    out = torch.cat([stream.push(x), stream.flush()])
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "factory", [isolator_lib.VoiceIsolator, isolator_lib.VoiceIsolator.load])
def test_unsupported_restoration_is_rejected_before_loading(factory):
    with pytest.raises(ValueError,
                       match="Stage-2 restoration is not supported"):
        factory("no-such-checkpoint", stage2="no-such-restorer", device="cpu")


def test_load_accepts_default_stage2_for_frozen_helpers(dual_model, tmp_path):
    path = _training_checkpoint(dual_model, tmp_path / "weights.pt")
    x = torch.randn(4800, generator=torch.Generator().manual_seed(4)) * 0.3
    expected = isolator_lib.VoiceIsolator(dual_model,
                                          device="cpu").isolate(x, 48000)

    loaded = isolator_lib.VoiceIsolator.load(path, stage2=None, device="cpu")
    actual = loaded.isolate(x, 48000)

    assert torch.equal(actual, expected)


def test_the_default_model_is_the_repository_id_resolved_by_mode(
        dual_model, tmp_path, monkeypatch):
    path = _training_checkpoint(dual_model, tmp_path / "weights.pt")
    calls = []

    def resolve(model, mode="offline"):
        calls.append((model, mode))
        return path

    monkeypatch.setattr(registry, "resolve", resolve)

    offline = isolator_lib.VoiceIsolator(device="cpu")
    live = isolator_lib.VoiceIsolator.load(device="cpu", mode="streaming")

    assert calls == [(registry.REPO_ID, "offline"),
                     (registry.REPO_ID, "streaming")]
    assert (offline.mode, live.mode) == ("offline", "streaming")


def test_load_model_reads_safetensors_and_training_checkpoints(
        dual_model, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import safetensors.torch

    st = {k: v.contiguous() for k, v in dual_model.state_dict().items()}
    safetensors.torch.save_file(
        st,
        str(tmp_path / "m.safetensors"),
        metadata={"config": json.dumps(dual_model.config.to_dict())},
    )
    m1 = isolator_lib.load_model(str(tmp_path / "m.safetensors"))
    m2 = isolator_lib.load_model(
        _training_checkpoint(
            dual_model,
            tmp_path / "ema.pt",
            extra={"isolator.enhance_decoder.heads.0.0.weight": torch.ones(3)}))
    x = torch.randn(3, 2400)
    with torch.inference_mode():
        assert torch.equal(m1(x), dual_model(x))
        assert torch.equal(m2(x), dual_model(x))
    assert m1.config == dual_model.config


def test_a_checkpoint_without_its_architecture_is_refused(dual_model, tmp_path):
    torch.save({"model": dual_model.state_dict()}, tmp_path / "bare.pt")
    with pytest.raises(ValueError, match="no embedded architecture"):
        isolator_lib.load_model(str(tmp_path / "bare.pt"))
    torch.save(dual_model.state_dict(), tmp_path / "state.pt")
    with pytest.raises(ValueError, match="no embedded architecture"):
        isolator_lib.load_model(str(tmp_path / "state.pt"))
    # pylint: disable-next=import-outside-toplevel
    import safetensors.torch

    safetensors.torch.save_file(
        {
            k: v.contiguous() for k, v in dual_model.state_dict().items()
        }, str(tmp_path / "nometa.safetensors"))
    with pytest.raises(ValueError, match="lacks the 'config'"):
        isolator_lib.load_model(str(tmp_path / "nometa.safetensors"))


def test_unknown_config_field_is_rejected(dual_model, tmp_path):
    path = _training_checkpoint(dual_model, tmp_path / "ema.pt")
    raw = torch.load(path, weights_only=True)
    raw["config"]["modeling"]["compact_family"]["future_knob"] = 7
    torch.save(raw, path)
    with pytest.raises(ValueError, match="does not implement"):
        isolator_lib.load_model(path)


def test_save_and_load_audio(tmp_path):
    x = torch.sin(torch.linspace(0, 100, 24000)) * 0.5
    audio_lib.save_audio(str(tmp_path / "a.wav"), x, 24000, subtype="FLOAT")
    y, sr = audio_lib.load_audio(str(tmp_path / "a.wav"))
    assert sr == 24000
    assert torch.allclose(x, y, atol=1e-6)
    assert not os.path.exists(tmp_path / "b.wav")


def test_missing_learned_weights_are_rejected(dual_model, tmp_path):
    """A checkpoint lacking a learned tensor must not run on random init."""
    path = _training_checkpoint(dual_model, tmp_path / "ema.pt")
    raw = torch.load(path, weights_only=True)
    raw["model"].pop("isolator.band_embedding")
    torch.save(raw, path)
    with pytest.raises(RuntimeError, match="missing"):
        isolator_lib.load_model(path)


def test_unexpected_model_weights_are_rejected(dual_model, tmp_path):
    path = _training_checkpoint(
        dual_model,
        tmp_path / "ema.pt",
        extra={"isolator.not_in_this_architecture.weight": torch.zeros(3)})
    with pytest.raises(RuntimeError, match="unexpected"):
        isolator_lib.load_model(path)


class _Payload:
    """Stand-in for an arbitrary pickled object (module scope for pickling)."""


def test_pickled_objects_in_checkpoints_are_refused(tmp_path):
    p = tmp_path / "evil.pt"
    torch.save({"config": {"modeling": {}}, "model": {}, "obj": _Payload()}, p)
    with pytest.raises(ValueError, match="plain tensors"):
        isolator_lib.load_model(str(p))


def _clips(sr, seed=0):
    g = torch.Generator().manual_seed(seed)
    loud = [torch.randn(int(0.05 * sr), generator=g) * 0.2 for _ in range(3)]
    quiet = torch.randn(int(0.03 * sr), generator=g) * 0.05
    return loud + [quiet]


def test_isolate_many_matches_isolate_across_lengths(isolator):
    clips = _clips(isolator.sample_rate)
    single = [isolator.isolate(c, isolator.sample_rate) for c in clips]
    many = isolator.isolate_many(clips, isolator.sample_rate)
    assert len(many) == len(clips)
    for s, m in zip(single, many):
        assert s.shape == m.shape
        assert torch.allclose(s, m, atol=1e-5, rtol=1e-4)


def test_isolate_many_keeps_order_types_and_rates(isolator):
    in_sr = 16000
    clips = [c.numpy() for c in _clips(in_sr, seed=1)]
    out = isolator.isolate_many(clips, in_sr, output_sample_rate=48000)
    assert all(isinstance(o, np.ndarray) for o in out)
    for c, o in zip(clips, out):
        ref = isolator.isolate(c, in_sr, output_sample_rate=48000)
        assert len(o) == len(ref) and np.allclose(o, ref, atol=1e-5, rtol=1e-4)
    same_rate = isolator.isolate_many(clips, in_sr)
    assert [len(o) for o in same_rate] == [len(c) for c in clips]


def test_precision_defaults_to_the_training_precision_on_a_gpu_only(dual_model):
    # auto: bf16 autocast on a CUDA device, float32 anywhere else
    assert isolator_lib.DEFAULT_PRECISION == "auto"
    vi = isolator_lib.VoiceIsolator(dual_model, device="cpu")
    assert vi.precision == "fp32"
    assert isolator_lib.VoiceIsolator(dual_model,
                                      device="cpu",
                                      precision="auto").precision == "fp32"
    assert isolator_lib.VoiceIsolator(dual_model,
                                      device="cpu",
                                      precision="bf16").precision == "bf16"
    if torch.cuda.is_available():
        on_gpu = isolator_lib.VoiceIsolator(dual_model, device="cuda")
        assert on_gpu.precision == "bf16"
    with pytest.raises(ValueError, match="precision"):
        isolator_lib.VoiceIsolator(dual_model, device="cpu", precision="fp8")


def test_reduced_precision_stays_close_to_fp32(tiny):
    if not torch.cuda.is_available():
        pytest.skip("autocast parity is checked on a GPU")
    clip = torch.randn(4800, generator=torch.Generator().manual_seed(2)) * 0.2
    ref = isolator_lib.VoiceIsolator(tiny.dual(),
                                     device="cuda").isolate(clip, 48000)
    for precision in ("fp16", "bf16"):
        out = isolator_lib.VoiceIsolator(tiny.dual(),
                                         device="cuda",
                                         precision=precision).isolate(
                                             clip, 48000)
        err = float(
            (out - ref).pow(2).mean()) / max(float(ref.pow(2).mean()), 1e-12)
        assert err < 1e-4


@pytest.mark.parametrize(
    "kw",
    [
        {
            "chunk_seconds": -1.0
        },
        {
            "overlap_seconds": -0.5
        },
        {
            "chunk_seconds": float("nan")
        },
        {
            "chunk_seconds": 0.0001
        },
    ],
)
def test_invalid_windows_fail_before_any_audio_is_processed(
        isolator, monkeypatch, kw):

    def untouched(*args, **kwargs):
        raise AssertionError("audio was processed")

    monkeypatch.setattr(isolator, "_prepare", untouched)
    with pytest.raises(ValueError):
        isolator.isolate(np.zeros(4800, dtype=np.float32), 48000, **kw)
    with pytest.raises(ValueError):
        isolator.isolate_many([np.zeros(4800, dtype=np.float32)], 48000, **kw)


@pytest.mark.parametrize(
    "audio,message",
    [
        (np.zeros(0, dtype=np.float32), "empty"),
        (np.array([0.1, np.inf, 0.0], dtype=np.float32), "NaN or infinite"),
        (np.array([0.1, np.nan, 0.0], dtype=np.float32), "NaN or infinite"),
    ],
)
def test_empty_or_non_finite_audio_is_refused(isolator, audio, message):
    with pytest.raises(ValueError, match=message):
        isolator.isolate(audio, 48000)


def test_streaming_mode_renders_through_the_stream(isolator):
    torch.manual_seed(0)
    x = torch.randn(2000) * 0.3
    isolator.mode = "streaming"
    isolator.lookahead_frames = 2
    rendered = isolator.isolate(x, 48000)
    session = isolator.stream()
    manual = torch.cat(
        [session.push(x[:900]),
         session.push(x[900:]),
         session.flush()])
    assert torch.allclose(rendered, manual, atol=1e-6)
    assert isolator.block_frames == 4 and isolator.latency_samples(
    ) == 64 + 16 * 2


def test_isolate_many_in_streaming_mode_matches_isolate(isolator):
    isolator.mode = "streaming"
    clips = _clips(isolator.sample_rate)
    single = [isolator.isolate(c, isolator.sample_rate) for c in clips]
    many = isolator.isolate_many(clips, isolator.sample_rate)
    for s, m in zip(single, many):
        assert s.shape == m.shape and torch.allclose(s, m, atol=1e-5)


def test_block_frames_are_validated(dual_model):
    with pytest.raises(ValueError, match="block_frames"):
        isolator_lib.VoiceIsolator(dual_model, device="cpu", block_frames=0)
    vi = isolator_lib.VoiceIsolator(dual_model,
                                    device="cpu",
                                    block_frames=2,
                                    mode="streaming")
    assert vi.latency_samples() == 64 and vi.stream().block_frames == 2


def test_the_streams_memory_is_the_training_window_unless_set(dual_model):
    vi = isolator_lib.VoiceIsolator(dual_model,
                                    device="cpu",
                                    mode="streaming",
                                    lookahead_frames=0)
    assert vi.cache_frames is None
    default = vi.stream()
    longer = isolator_lib.VoiceIsolator(dual_model,
                                        device="cpu",
                                        mode="streaming",
                                        lookahead_frames=0,
                                        cache_frames=2 * default.cache_frames)
    assert longer.stream().cache_frames == 2 * default.cache_frames
    assert longer.stream(cache_frames=7).cache_frames == 7
    with pytest.raises(ValueError, match="cache_frames"):
        isolator_lib.VoiceIsolator(dual_model, device="cpu", cache_frames=0)


def test_the_streams_windows_default_on_and_zero_turns_them_off(dual_model):
    vi = isolator_lib.VoiceIsolator(dual_model,
                                    device="cpu",
                                    mode="streaming",
                                    lookahead_frames=0)
    assert (vi.window_frames,
            vi.crossfade_frames) == (isolator_lib.STREAM_WINDOW_FRAMES,
                                     isolator_lib.STREAM_CROSSFADE_FRAMES)
    session = vi.stream()
    assert (session.window_frames, session.crossfade_frames) == (600, 200)
    assert vi.stream(window_frames=0).window_frames is None
    off = isolator_lib.VoiceIsolator(dual_model,
                                     device="cpu",
                                     mode="streaming",
                                     lookahead_frames=0,
                                     window_frames=0)
    assert off.window_frames is None and off.stream().window_frames is None
    assert off.stream(window_frames=400).crossfade_frames == 200
    custom = isolator_lib.VoiceIsolator(dual_model,
                                        device="cpu",
                                        mode="streaming",
                                        lookahead_frames=0,
                                        window_frames=40,
                                        crossfade_frames=10)
    session = custom.stream()
    assert (session.window_frames, session.crossfade_frames) == (40, 10)
    assert custom.stream(window_frames=80).crossfade_frames == 10
    x = torch.randn(48000, generator=torch.Generator().manual_seed(3)) * 0.3
    y = custom.isolate(x, 48000)
    assert y.shape == x.shape and torch.isfinite(y).all()
    with pytest.raises(ValueError, match="crossfade_frames"):
        isolator_lib.VoiceIsolator(dual_model,
                                   device="cpu",
                                   window_frames=40,
                                   crossfade_frames=40)


def test_offline_passes_run_the_model_over_its_own_output(tiny):
    net = tiny.offline()
    x = torch.randn(4800, generator=torch.Generator().manual_seed(3)) * 0.3
    once = isolator_lib.VoiceIsolator(net, device="cpu", precision="fp32")
    twice = isolator_lib.VoiceIsolator(net,
                                       device="cpu",
                                       precision="fp32",
                                       offline_passes=2)
    assert once.offline_passes == 1 and twice.offline_passes == 2
    with torch.inference_mode():
        expected = net.enhance(net.enhance(x))
    assert torch.allclose(twice.isolate(x, 48000), expected, atol=1e-6)
    assert not torch.allclose(twice.isolate(x, 48000), once.isolate(x, 48000))
    # batched clips take the same passes
    many = twice.isolate_many([x, x[:2400]], 48000)
    assert torch.allclose(many[0], twice.isolate(x, 48000), atol=1e-5)


def test_the_checkpoint_carries_its_passes_and_a_request_overrides_them(tiny):
    net = tiny.dual()
    net.config = dataclasses.replace(net.config, offline_passes=2)
    assert isolator_lib.VoiceIsolator(net, device="cpu").offline_passes == 2
    assert isolator_lib.VoiceIsolator(net, device="cpu",
                                      offline_passes=1).offline_passes == 1
    x = torch.randn(4800, generator=torch.Generator().manual_seed(4)) * 0.3
    # streaming ignores the file's count: the stream is one pass
    live = isolator_lib.VoiceIsolator(net, device="cpu", mode="streaming")
    plain = tiny.dual()
    single = isolator_lib.VoiceIsolator(plain, device="cpu", mode="streaming")
    assert torch.allclose(live.isolate(x, 48000),
                          single.isolate(x, 48000),
                          atol=1e-6)
    # a request above one is refused where it cannot apply
    with pytest.raises(ValueError, match="offline mode"):
        isolator_lib.VoiceIsolator(plain,
                                   device="cpu",
                                   mode="streaming",
                                   offline_passes=2)
    with pytest.raises(ValueError, match="offline mode"):
        isolator_lib.VoiceIsolator(tiny.windowed(),
                                   device="cpu",
                                   mode="streaming",
                                   offline_passes=2)
    with pytest.raises(ValueError, match="whole number"):
        isolator_lib.VoiceIsolator(plain, device="cpu", offline_passes=0)


@pytest.mark.parametrize(
    "cuda,mps,version,fft_ok,expected",
    [
        (True, True, "2.6.0", True, "cuda"),
        (False, True, "2.6.0", True, "mps"),
        (False, True, "2.15.0", True, "mps"),
        (False, False, "2.6.0", True, "cpu"),
        # the supported torch>=2.1 whose MPS backend cannot run the models
        (False, True, "2.1.0", True, "cpu"),
        (False, True, "2.5.1", True, "cpu"),
        # a macOS whose Metal backend lacks a working FFT
        (False, True, "2.6.0", False, "cpu"),
    ])
def test_default_device_prefers_cuda_then_a_working_mps_then_cpu(
        monkeypatch, cuda, mps, version, fft_ok, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch, "__version__",
                        torch.torch_version.TorchVersion(version))
    monkeypatch.setattr(isolator_lib, "_mps_fft_matches_cpu", lambda: fft_ok)
    assert isolator_lib.default_device() == torch.device(expected)


def test_the_mps_fft_check_reports_a_missing_op_as_unusable(monkeypatch):
    # pylint: disable=protected-access

    def missing(*_):
        raise NotImplementedError("aten::_fft_r2c is not implemented for MPS")

    monkeypatch.setattr(torch.Tensor, "to", lambda self, *_: self)
    monkeypatch.setattr(torch.fft, "rfft", missing)
    isolator_lib._mps_fft_matches_cpu.cache_clear()
    try:
        assert not isolator_lib._mps_fft_matches_cpu()
    finally:
        isolator_lib._mps_fft_matches_cpu.cache_clear()


def test_an_explicit_device_overrides_the_default(dual_model, monkeypatch):
    monkeypatch.setattr(isolator_lib, "default_device",
                        lambda: pytest.fail("default_device was consulted"))
    assert isolator_lib.VoiceIsolator(
        dual_model, device="cpu").device == torch.device("cpu")


@pytest.mark.parametrize("mode", isolator_lib.MODES)
def test_mps_matches_cpu(tiny, mode):
    if isolator_lib.default_device().type != "mps":
        pytest.skip("needs a Metal backend the default would choose")
    clip = torch.randn(9600, generator=torch.Generator().manual_seed(3)) * 0.2
    model = tiny.dual()
    outs = [
        np.asarray(
            isolator_lib.VoiceIsolator(model, device=device,
                                       mode=mode).isolate(clip, 48000))
        for device in ("cpu", "mps")
    ]
    assert outs[1].dtype == outs[0].dtype and outs[1].shape == outs[0].shape
    assert np.allclose(outs[1], outs[0], atol=1e-4)
