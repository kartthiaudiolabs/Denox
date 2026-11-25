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
"""The isolator network: config, bands, modes, causality, windows."""

import dataclasses

import pytest
import torch

from inworld.voice_isolation import model

SR = 48000


@pytest.mark.parametrize("position", ["offline", 0, 2])
def test_identity_at_init(identity_model, position):
    """The untrained model is an exact identity: its heads start at zero."""
    if position == "offline":
        identity_model.set_offline()
    else:
        identity_model.set_lookahead(position)
    torch.manual_seed(0)
    x = torch.randn(2, SR // 10) * 0.3
    with torch.inference_mode():
        y = identity_model(x)
    assert torch.equal(y, x)
    assert torch.equal(identity_model.enhance(x[0]), x[0])


def test_band_edges_cover_the_spectrum_once(tiny):
    edges = model.band_edges(model.VoiceIsolatorConfig())
    assert len(edges) == 46
    assert edges[0][0] == 0 and edges[-1][1] == 1025
    assert all(a[1] == b[0] for a, b in zip(edges, edges[1:]))
    tiny = model.band_edges(tiny.config())
    assert tiny == [(0, 8), (8, 16), (16, 24), (24, 32), (32, 33)]


def test_the_released_architectures_have_their_parameter_counts():
    offline = model.VoiceIsolatorNetwork(model.VoiceIsolatorConfig(width=512))
    assert offline.num_params == 64_047_112
    assert offline.default_window() == (12.0, 2.0)
    assert len(offline.blocks) == 7 and len(offline.edges) == 46
    realtime = model.VoiceIsolatorNetwork(
        model.VoiceIsolatorConfig(width=512,
                                  causal=True,
                                  chunk_seconds=30.0,
                                  lookahead_frames=2,
                                  lookahead_frames_choices=(0, 2, 4),
                                  attention_window_frames=1200))
    assert realtime.num_params == 58_178_056
    assert not realtime.offline and realtime.lookahead_frames == 2


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(width=15), "multiple of twice"),
        (dict(block_types=("grid",)), "block_types"),
        (dict(n_fft=65), "n_fft must be even"),
        (dict(hop_length=40), "n_fft must be even"),
        (dict(correction_bound=1.0), "correction_bound"),
        (dict(overlap_seconds=0.6), "overlap_seconds"),
        (dict(exclusive_axes="none"), "exclusive_axes"),
        (dict(lookahead_frames_choices=(2, 0)), "ascending"),
        (dict(lookahead_frames=1), "causal or dual-mode"),
        (dict(dual_mode=True), "list the streaming positions"),
        (dict(dual_mode=True, lookahead_frames_choices=(0,),
              causal=True), "leave causal off"),
        (dict(mode_film=True), "mode_film"),
        (dict(causal=True, lookahead_frames=3, lookahead_frames_choices=(0, 2)),
         "one of lookahead_frames_choices"),
        (dict(attention_window_frames=8), "needs a causal model"),
        (dict(causal=True, attention_window_frames=-1), "negative"),
    ],
)
def test_configs_that_cannot_be_built_are_refused(tiny, overrides, match):
    with pytest.raises(ValueError, match=match):
        tiny.config(**overrides)


def test_config_from_a_checkpoint_dictionary():
    raw = dict(
        width=16,
        block_types=["transformer", "mamba"],
        num_heads=2,
        n_fft=64,
        hop_length=16,
        chunk_seconds=0.05,
        overlap_seconds=0.01,
        band_plan=[[6000, 24000]],
        attention_backend="fa2",
        checkpoint_every=1,
        offline_mode_prob=0.5,
        dual_distill=True,
        grid_width=48,
        representation="bands",
        hint_checkpoint=None,
        dual_mode=True,
        lookahead_frames=0,
        lookahead_frames_choices=[0, 2],
    )
    cfg = model.VoiceIsolatorConfig.from_dict(raw)
    assert cfg.block_types == ("transformer", "mamba")
    assert cfg.lookahead_frames_choices == (0, 2)
    assert cfg.band_plan == ((6000, 24000),)
    assert model.VoiceIsolatorConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="does not implement"):
        model.VoiceIsolatorConfig.from_dict(dict(raw, future_knob=7))
    with pytest.raises(ValueError, match="representation"):
        model.VoiceIsolatorConfig.from_dict(
            dict(raw, representation="full_grid"))
    with pytest.raises(ValueError, match="decoder_type"):
        model.VoiceIsolatorConfig.from_dict(
            dict(raw, decoder_type="positive_real"))
    with pytest.raises(ValueError, match="hint_checkpoint"):
        model.VoiceIsolatorConfig.from_dict(dict(raw, hint_checkpoint="h.pt"))
    assert dataclasses.asdict(cfg) == cfg.to_dict()
    windowed = model.VoiceIsolatorConfig.from_dict(
        dict(raw,
             dual_mode=False,
             causal=True,
             lookahead_frames_choices=[0, 2, 4],
             attention_window_frames=1200))
    assert windowed.attention_window_frames == 1200
    assert model.VoiceIsolatorConfig.from_dict(windowed.to_dict()) == windowed


def test_an_attention_window_bounds_the_causal_forward(tiny):
    """A frame sees itself and the ``window - 1`` frames before it: the
    windowed model equals the same weights without a window on a clip that
    fits the window and differs once the clip is longer."""
    mask = model.window_mask(6, 3, torch.device("cpu"))
    assert mask.tolist() == [
        [j <= i and i - j < 3 for j in range(6)] for i in range(6)
    ]
    windowed, plain = tiny.windowed(), tiny.windowed(attention_window_frames=0)
    assert all(
        torch.equal(a, b)
        for a, b in zip(windowed.parameters(), plain.parameters()))
    hop = windowed.config.hop_length
    window = windowed.config.attention_window_frames
    g = torch.Generator().manual_seed(3)
    short = torch.randn(1, hop * (window - 2), generator=g) * 0.3
    long = torch.randn(1, hop * 6 * window, generator=g) * 0.3
    with torch.inference_mode():
        assert torch.allclose(windowed(short), plain(short), atol=1e-6)
        assert not torch.allclose(windowed(long), plain(long), atol=1e-4)


def test_the_mode_knob(dual_model, offline_model):
    assert dual_model.offline and dual_model.lookahead_frames == 0
    assert dual_model.set_lookahead(2) == 2
    assert not dual_model.offline and dual_model.lookahead_frames == 2
    dual_model.set_offline()
    assert dual_model.offline
    with pytest.raises(ValueError, match="not among the trained positions"):
        dual_model.set_lookahead(1)
    assert offline_model.offline
    with pytest.raises(ValueError, match="no streaming positions"):
        offline_model.set_lookahead(0)
    with pytest.raises(ValueError, match="Only a dual-mode"):
        offline_model.set_offline()


def test_the_modes_give_different_outputs(dual_model):
    torch.manual_seed(0)
    x = torch.randn(1, SR // 10) * 0.3
    outputs = []
    for position in ("offline", 0, 2):
        if position == "offline":
            dual_model.set_offline()
        else:
            dual_model.set_lookahead(position)
        with torch.inference_mode():
            outputs.append(dual_model(x))
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-4)
    assert not torch.allclose(outputs[1], outputs[2], atol=1e-4)


@pytest.mark.parametrize("lookahead", [0, 2])
def test_a_streaming_position_reads_exactly_its_lookahead_ahead(
        dual_model, lookahead):
    """Changing the input from sample j on leaves earlier output alone.

    Up to one STFT window plus the look-ahead frames before j; offline, the
    change reaches the start of the clip.
    """
    cfg = dual_model.config
    torch.manual_seed(0)
    x = torch.randn(1, SR // 10) * 0.3
    j = SR // 20
    x2 = x.clone()
    x2[:, j:] = torch.randn(x2.shape[-1] - j) * 0.3
    dual_model.set_lookahead(lookahead)
    with torch.inference_mode():
        y1, y2 = dual_model(x), dual_model(x2)
    safe = j - cfg.n_fft - cfg.hop_length * (lookahead + 1)
    assert torch.allclose(y1[:, :safe], y2[:, :safe], atol=1e-6)
    assert not torch.allclose(y1, y2, atol=1e-4)
    dual_model.set_offline()
    with torch.inference_mode():
        y1, y2 = dual_model(x), dual_model(x2)
    assert (y1[:, :cfg.n_fft] - y2[:, :cfg.n_fft]).abs().max() > 1e-6


def test_batched_forward_matches_single_clips(dual_model):
    torch.manual_seed(0)
    x = torch.randn(3, SR // 10) * 0.3
    with torch.inference_mode():
        batched = dual_model(x)
        single = torch.cat([dual_model(x[i:i + 1]) for i in range(3)])
    assert torch.allclose(batched, single, atol=1e-5)


def test_windowed_render_matches_a_single_pass_within_one_window(dual_model):
    torch.manual_seed(0)
    x = torch.randn(SR // 20) * 0.3
    with torch.inference_mode():
        single = dual_model(x[None])[0]
    assert torch.equal(dual_model.enhance(x), single)
    assert dual_model.default_window() == (dual_model.config.chunk_seconds,
                                           dual_model.config.overlap_seconds)


def test_chunked_render_keeps_the_length_and_is_deterministic(dual_model):
    torch.manual_seed(0)
    x = torch.randn(2, int(0.137 * SR)) * 0.3
    y = dual_model.enhance(x, chunk_seconds=0.03, overlap_seconds=0.01)
    assert y.shape == x.shape
    assert torch.equal(
        y, dual_model.enhance(x, chunk_seconds=0.03, overlap_seconds=0.01))
    assert torch.isfinite(y).all()


def test_chunking_an_identity_model_stays_the_identity(identity_model):
    """The crossfade weights sum to one everywhere, with and without overlap."""
    torch.manual_seed(0)
    x = torch.randn(int(0.137 * SR)) * 0.3
    for overlap in (0.01, 0.0, 0.02):
        y = identity_model.enhance(x,
                                   chunk_seconds=0.03,
                                   overlap_seconds=overlap)
        assert torch.allclose(y, x, atol=1e-5)


@pytest.mark.parametrize("extra", [1, 7, 100, 959])
def test_every_window_of_a_long_clip_is_a_full_window(dual_model, extra):
    """A clip just past a step boundary no longer ends in a stub window."""
    torch.manual_seed(0)
    chunk, overlap = model.window_samples(0.03, 0.01, SR)
    x = torch.randn(2 * (chunk - overlap) + chunk + extra) * 0.3
    lengths = []
    hook = dual_model.register_forward_pre_hook(
        lambda _, args: lengths.append(args[0].shape[-1]))

    try:
        y = dual_model.enhance(x, chunk_seconds=0.03, overlap_seconds=0.01)
    finally:
        hook.remove()

    assert y.shape == x.shape
    assert set(lengths) == {chunk}


def test_the_tail_of_a_long_clip_comes_from_the_last_full_window(dual_model):
    torch.manual_seed(0)
    chunk, overlap = model.window_samples(0.03, 0.01, SR)
    step = chunk - overlap
    n = 2 * step + chunk + 100
    x = torch.randn(n) * 0.3
    with torch.inference_mode():
        last = dual_model(x[None, n - chunk:])[0]

    y = dual_model.enhance(x, chunk_seconds=0.03, overlap_seconds=0.01)

    previous_end = 2 * step + chunk
    assert torch.allclose(y[previous_end:],
                          last[previous_end - (n - chunk):],
                          atol=1e-6)


@pytest.mark.parametrize("samples", [1, 16, 32, 63, 64, 65])
def test_inputs_shorter_than_one_window(dual_model, samples):
    torch.manual_seed(0)
    x = torch.randn(samples) * 0.3
    y = dual_model.enhance(x)
    assert y.shape == (samples,)
    assert torch.isfinite(y).all()


def test_bad_inputs_are_refused(dual_model):
    with pytest.raises(ValueError, match="mono audio"):
        dual_model(torch.zeros(10))
    with pytest.raises(ValueError, match="mono audio"):
        dual_model(torch.zeros(1, 0))
    with pytest.raises(ValueError, match="mono audio"):
        dual_model(torch.zeros(1, 10, dtype=torch.int32))


@pytest.mark.parametrize(
    "chunk,overlap,match",
    [
        (-8.0, 2.0, "shorter than"),
        (0.0, 0.0, "shorter than"),
        (float("nan"), 1.0, "must be finite"),
        (float("inf"), 1.0, "must be finite"),
        (8.0, -1.0, "must not be negative"),
        (8.0, float("nan"), "must be finite"),
        ("eight", 1.0, "number of seconds"),
    ],
)
def test_windows_that_would_drop_or_blank_audio_are_refused(
        dual_model, chunk, overlap, match):
    x = torch.randn(4800)
    with pytest.raises(ValueError, match=match):
        dual_model.enhance(x, chunk_seconds=chunk, overlap_seconds=overlap)


def test_window_samples_caps_the_crossfade_at_half_the_window():
    assert model.window_samples(1.0, 0.25, 48000) == (48000, 12000)
    assert model.window_samples(1.0, 0.9, 48000) == (48000, 24000)
    assert model.window_samples(1.0, 0.0, 48000) == (48000, 0)


def test_read_ahead_takes_later_frames_and_holds_the_last():
    raw = torch.arange(6.0).view(1, 1, 6, 1)
    assert model.read_ahead(raw, 2).flatten().tolist() == [2, 3, 4, 5, 5, 5]
    assert torch.equal(model.read_ahead(raw, 0), raw)


def test_rope_rotation_is_relative():
    """Attention logits depend on the offset between positions only."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    cos, sin = model.rope_tables(20, 8, "cpu", torch.float32)
    logits = []
    for base in (0, 7):
        rq = model.apply_rope(q, cos[base + 3:base + 4], sin[base + 3:base + 4])
        rk = model.apply_rope(k, cos[base:base + 1], sin[base:base + 1])
        logits.append((rq * rk).sum())
    assert torch.allclose(logits[0], logits[1], atol=1e-5)


def test_exclusive_attention_removes_the_own_value_component():
    torch.manual_seed(0)
    value = torch.randn(2, 3, 5, 4)
    output = model.exclusive(value * 2.0, value)
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    orthogonal = model.exclusive(torch.randn(2, 3, 5, 4), value)
    assert torch.allclose((orthogonal * value).sum(-1),
                          torch.zeros(2, 3, 5),
                          atol=1e-5)


def test_bounded_correction_never_exceeds_the_bound():
    torch.manual_seed(0)
    reference = torch.randn(3, 7, dtype=torch.complex64)
    correction = model.bounded_correction(
        torch.randn(3, 7, 2) * 10, reference, 0.5)
    magnitude = torch.linalg.vector_norm(correction, dim=-1)
    assert (magnitude <= 0.5 * reference.abs() + 1e-6).all()


def test_offline_passes_are_a_count_a_causal_model_cannot_take(tiny):
    assert tiny.config().offline_passes == 1
    two = model.VoiceIsolatorConfig.from_dict(
        dict(tiny.config().to_dict(), offline_passes=2))
    assert two.offline_passes == 2
    assert model.VoiceIsolatorConfig.from_dict(two.to_dict()) == two
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="whole number"):
            tiny.config(offline_passes=bad)
    # a causal model streams in one pass; a dual-mode one runs passes offline
    with pytest.raises(ValueError, match="causal model streams in one pass"):
        tiny.config(causal=True, offline_passes=2)
    assert tiny.config(dual_mode=True,
                       lookahead_frames_choices=(0, 2),
                       offline_passes=2).offline_passes == 2
