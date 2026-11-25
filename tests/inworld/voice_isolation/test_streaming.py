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
"""The stream against the whole-clip forward at the same position."""

import numpy as np
import pytest
import torch

from inworld.voice_isolation import streaming

SR = 48000


def _clip(seconds=0.05, seed=0):
    """A clip that starts with one window of silence.

    Zero padding (the stream) and reflect padding (a clip) then agree at the
    start; the end differs by construction and is left out of comparisons.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.cat(
        [torch.zeros(64),
         torch.randn(int(seconds * SR), generator=g) * 0.3])


def _run(stream, clip, sizes):
    pieces, pos, index = [], 0, 0
    while pos < clip.shape[-1]:
        step = sizes[index % len(sizes)]
        pieces.append(stream.push(clip[..., pos:pos + step]))
        pos += step
        index += 1
    pieces.append(stream.flush())
    return torch.cat(pieces, dim=-1)


@pytest.mark.parametrize("lookahead,block", [(0, 1), (0, 3), (2, 4), (2, 7)])
def test_the_stream_equals_the_whole_clip_at_its_position(
        dual_model, lookahead, block):
    dual_model.set_lookahead(lookahead)
    clip = _clip()
    with torch.inference_mode():
        whole = dual_model(clip[None])[0]
    out = _run(streaming.Stream(dual_model, block_frames=block), clip,
               (37, 101, 5))
    assert out.shape == clip.shape
    tail = 2 * dual_model.config.n_fft
    assert torch.allclose(out[:-tail], whole[:-tail], atol=1e-5)


def test_piece_sizes_do_not_change_the_output(dual_model):
    """Within the attention cache, how the audio arrives does not matter."""
    dual_model.set_lookahead(0)
    clip = _clip(0.04)
    a = _run(streaming.Stream(dual_model, block_frames=4), clip, (1000,))
    b = _run(streaming.Stream(dual_model, block_frames=4), clip, (1, 7, 300))
    assert torch.allclose(a, b, atol=1e-6)


def test_samples_are_released_once_final_and_nothing_is_lost(dual_model):
    dual_model.set_lookahead(2)
    stream = streaming.Stream(dual_model, block_frames=4)
    clip = _clip()
    released = 0
    for pos in range(0, len(clip), 500):
        out = stream.push(clip[pos:pos + 500])
        released += len(out)
        fed = min(pos + 500, len(clip))
        assert released <= fed
        assert released >= fed - stream.latency_samples - stream.hop
    released += len(stream.flush())
    assert released == len(clip)
    assert stream.latency_samples == 64 + 16 * 2  # the block adds none
    assert stream.latency_seconds == pytest.approx(stream.latency_samples / SR)


def test_the_stream_needs_a_streaming_position(dual_model, offline_model):
    dual_model.set_offline()
    with pytest.raises(ValueError, match="streaming position"):
        streaming.Stream(dual_model)
    with pytest.raises(ValueError, match="streaming position"):
        streaming.Stream(offline_model)
    dual_model.set_lookahead(0)
    with pytest.raises(ValueError, match="at least one frame"):
        streaming.Stream(dual_model, block_frames=0)


def test_flush_ends_the_stream(dual_model):
    dual_model.set_lookahead(0)
    stream = streaming.Stream(dual_model)
    clip = _clip(0.02)
    out = torch.cat([stream.push(clip), stream.flush()])
    assert out.shape == clip.shape
    assert stream.flush().numel() == 0
    with pytest.raises(RuntimeError, match="flushed"):
        stream.push(clip)


def test_numpy_in_numpy_out_and_bad_pieces_refused(dual_model):
    dual_model.set_lookahead(0)
    stream = streaming.Stream(dual_model)
    out = stream.push(_clip(0.02).numpy())
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert isinstance(stream.flush(as_numpy=True), np.ndarray)
    stream = streaming.Stream(dual_model)
    with pytest.raises(ValueError, match="mono float"):
        stream.push(torch.zeros(1, 2, 10))
    with pytest.raises(ValueError, match="mono float"):
        stream.push(torch.zeros(10, dtype=torch.int16))


def test_a_short_cache_bounds_the_attention_past(dual_model):
    dual_model.set_lookahead(0)
    clip = _clip(0.1)
    unbounded = _run(streaming.Stream(dual_model, block_frames=4), clip, (640,))
    bounded = _run(streaming.Stream(dual_model, block_frames=4, cache_frames=8),
                   clip, (640,))
    assert bounded.shape == unbounded.shape
    assert not torch.allclose(bounded, unbounded, atol=1e-4)


def test_clips_stream_in_lockstep_as_a_batch(dual_model):
    dual_model.set_lookahead(2)
    clips = torch.stack([_clip(seed=1), _clip(seed=2)])
    alone = [_run(streaming.Stream(dual_model), c, (500,)) for c in clips]
    together = _run(streaming.Stream(dual_model), clips, (500,))
    assert together.shape == clips.shape
    for j in range(2):
        assert torch.allclose(together[j], alone[j], atol=1e-5)
    assert isinstance(
        streaming.Stream(dual_model).push(clips[:, :300].numpy()), np.ndarray)
    with pytest.raises(ValueError, match="same rank and batch"):
        stream = streaming.Stream(dual_model)
        stream.push(clips[:, :100])
        stream.push(clips[0, :100])


def test_silence_appended_to_a_clip_changes_nothing_before_its_last_window(
        dual_model):
    dual_model.set_lookahead(0)
    clip = _clip()
    alone = _run(streaming.Stream(dual_model), clip, (640,))
    padded = _run(streaming.Stream(dual_model),
                  torch.cat([clip, torch.zeros(3000)]), (640,))
    n_fft = dual_model.config.n_fft
    assert torch.allclose(padded[:len(clip) - n_fft], alone[:-n_fft], atol=1e-6)


@pytest.mark.parametrize("lookahead", [0, 2])
def test_padded_clips_with_their_lengths_end_exactly_as_alone(
        dual_model, lookahead):
    dual_model.set_lookahead(lookahead)
    short, long = _clip(0.03, seed=3), _clip(0.04, seed=4)
    alone = [
        _run(streaming.Stream(dual_model), c, (700,)) for c in (short, long)
    ]
    batch = torch.stack(
        [torch.cat([short, torch.zeros(len(long) - len(short))]), long])
    together = _run(
        streaming.Stream(dual_model, lengths=[len(short), len(long)]), batch,
        (700,))
    assert torch.allclose(together[0, :len(short)], alone[0], atol=1e-5)
    assert torch.allclose(together[1], alone[1], atol=1e-5)
    with pytest.raises(ValueError, match="lengths names every clip"):
        streaming.Stream(dual_model, lengths=[1]).push(batch[:, :100])


def _windowed(model, clip, window, crossfade, cache=None):
    stream = streaming.Stream(model,
                              block_frames=4,
                              cache_frames=cache,
                              window_frames=window,
                              crossfade_frames=crossfade)
    return _run(stream, clip, (301,)), stream


def test_windowed_attention_equals_the_plain_stream_before_the_first_crossfade(
        dual_model):
    dual_model.set_lookahead(0)
    clip = _clip(seconds=0.5, seed=7)
    plain, _ = _windowed(dual_model, clip, None, None)
    windowed, stream = _windowed(dual_model, clip, 40, 10)
    assert windowed.shape == plain.shape == clip.shape
    assert torch.isfinite(windowed).all()
    # Frames before the first crossfade (30 frames) are lane A alone: the
    # samples agree until the synthesis window that first overlaps a blended
    # frame.
    same = 30 * stream.hop - stream.n_fft
    assert torch.equal(windowed[:same], plain[:same])
    assert not torch.allclose(windowed, plain)


def test_windowed_attention_swaps_lanes_at_every_boundary_and_bounds_each_lane(
        dual_model):
    dual_model.set_lookahead(0)
    clip = _clip(seconds=0.5, seed=8)
    out, stream = _windowed(dual_model, clip, 40, 10, cache=1000)
    assert out.shape == clip.shape and torch.isfinite(out).all()
    frames = clip.shape[-1] // stream.hop
    expected = [40 * k for k in range(1, frames // 40 + 1)]
    assert stream.boundaries == expected[:len(stream.boundaries)]
    assert len(stream.boundaries) >= 30
    for state in stream.states:
        if isinstance(state, streaming._AttentionState):  # pylint: disable=protected-access
            assert state.end <= 40 + 10


def test_windowed_attention_flushes_inside_a_crossfade(dual_model):
    dual_model.set_lookahead(0)
    g = torch.Generator().manual_seed(9)
    # ends five frames into the first crossfade, which starts at frame 30
    clip = torch.randn(35 * 16 + 8, generator=g) * 0.3
    out, stream = _windowed(dual_model, clip, 40, 10)
    assert out.shape == clip.shape and torch.isfinite(out).all()
    assert stream.states_b is not None and not stream.boundaries


@pytest.mark.parametrize("window,crossfade", [(40, None), (None, 10), (40, 40),
                                              (40, 0)])
def test_windowed_attention_arguments_go_together(dual_model, window,
                                                  crossfade):
    dual_model.set_lookahead(0)
    with pytest.raises(ValueError):
        streaming.Stream(dual_model,
                         window_frames=window,
                         crossfade_frames=crossfade)


def test_windowed_clips_stream_in_lockstep_as_a_batch(dual_model):
    dual_model.set_lookahead(2)
    clips = torch.stack(
        [_clip(seconds=0.3, seed=1),
         _clip(seconds=0.3, seed=2)])
    alone = [_windowed(dual_model, c, 40, 10)[0] for c in clips]
    together, _ = _windowed(dual_model, clips, 40, 10)
    assert together.shape == clips.shape
    for j in range(2):
        assert torch.allclose(together[j], alone[j], atol=1e-5)


def _lane_values(model, clip, window, crossfade, block, sizes, monkeypatch):
    """The blended decoder output per frame when each lane emits its own
    constant: 0 for the first lane, 1 for the next, and so on."""
    lanes, alive = {}, []  # every lane is kept alive so no id is reused
    clone = streaming._clone_states  # pylint: disable=protected-access

    def tagged_clone(states):
        fresh = clone(states)
        alive.append(fresh)
        lanes[id(fresh)] = len(alive)
        return fresh

    monkeypatch.setattr(streaming, "_clone_states", tagged_clone)

    def backbone(self, features, states, film):  # pylint: disable=unused-argument
        return torch.cat([
            torch.full_like(features, float(lanes.get(id(lane), 0)))
            for lane in states
        ])

    monkeypatch.setattr(streaming.Stream, "_backbone", backbone)
    decode = streaming._PackedBands.decode  # pylint: disable=protected-access

    def constant(self, features):
        """The decoder's output shape, each batch row (lanes run stacked)
        filled with its lane's constant."""
        out = decode(self, features)
        return features.flatten(1)[:, 0].view(-1, 1, 1, 1).to(
            out.dtype).expand_as(out).clone()

    monkeypatch.setattr(streaming._PackedBands, "decode", constant)  # pylint: disable=protected-access
    blocks, values = [], []
    run = streaming.Stream._run_block  # pylint: disable=protected-access

    def spy(self, spectra):
        start = self._frames  # pylint: disable=protected-access
        raw = run(self, spectra)
        blocks.append((start, spectra.shape[0]))
        values.extend(float(v) for v in raw[:, 0, 0, 0])
        return raw

    monkeypatch.setattr(streaming.Stream, "_run_block", spy)
    stream = streaming.Stream(model,
                              block_frames=block,
                              window_frames=window,
                              crossfade_frames=crossfade)
    _run(stream, clip, sizes)
    return blocks, values, stream


@pytest.mark.parametrize("window,crossfade,block,sizes", [
    (40, 10, 4, (301,)),
    (40, 10, 4, (37, 101, 5)),
    (37, 9, 4, (480,)),
    (30, 11, 7, (16 * 13 + 3,)),
    (25, 5, 3, (97, 211)),
])
def test_the_new_lane_owns_every_frame_from_its_boundary(
        dual_model, window, crossfade, block, sizes, monkeypatch):
    dual_model.set_lookahead(0)
    clip = _clip(seconds=0.5, seed=11)
    blocks, values, stream = _lane_values(dual_model, clip, window, crossfade,
                                          block, sizes, monkeypatch)
    start = window - crossfade
    for first, count in blocks:
        last = first + count - 1
        assert first // window == last // window, (first, count)
        assert not first % window < start <= last % window, (first, count)
    for f, value in enumerate(values):
        k, offset = divmod(f, window)
        ramp = min(max((offset - start + 1) / crossfade, 0.0), 1.0)
        assert value == pytest.approx(k + ramp, abs=1e-5), (f, value)
    assert stream.boundaries == [
        window * k for k in range(1,
                                  len(values) // window + 1)
    ]


@pytest.mark.parametrize("sizes", [(301,), (37, 101, 5), (16 * 40 - 8,), (1,)])
def test_windowed_output_does_not_depend_on_the_pushes(dual_model, sizes):
    dual_model.set_lookahead(2)
    clip = _clip(seconds=0.3, seed=12)
    reference, _ = _windowed(dual_model, clip, 40, 10)
    out = _run(
        streaming.Stream(dual_model,
                         block_frames=4,
                         window_frames=40,
                         crossfade_frames=10), clip, sizes)
    assert torch.allclose(out, reference, atol=1e-5)


@pytest.mark.parametrize("sizes", [(37, 101, 5), (16 * 40 - 8,), (1,)])
def test_the_plain_stream_does_not_depend_on_the_pushes_within_its_cache(
        dual_model, sizes):
    dual_model.set_lookahead(2)
    clip = _clip(seconds=0.3, seed=12)
    frames = clip.shape[-1] // dual_model.config.hop_length + 1
    reference = _run(
        streaming.Stream(dual_model, block_frames=4, cache_frames=frames), clip,
        (301,))
    out = _run(
        streaming.Stream(dual_model, block_frames=4, cache_frames=frames), clip,
        sizes)
    assert torch.allclose(out, reference, atol=1e-5)


@pytest.mark.parametrize("lookahead", [0, 2, 4])
@pytest.mark.parametrize("block", [1, 4, 7])
def test_a_windowed_checkpoint_streams_as_its_forward_beyond_the_cache(
        windowed_model, lookahead, block):
    """The stream keeps the training bound, a frame seeing itself and the
    window - 1 frames before it, across every block and cache eviction."""
    windowed_model.set_lookahead(lookahead)
    clip = _clip(0.1)
    config = windowed_model.config
    assert clip.shape[
        -1] // config.hop_length > 4 * config.attention_window_frames
    with torch.inference_mode():
        whole = windowed_model(clip[None])[0]
    stream = streaming.Stream(windowed_model, block_frames=block)
    assert stream.cache_frames == config.attention_window_frames
    out = _run(stream, clip, (37, 101, 5))
    tail = 2 * config.n_fft
    assert torch.allclose(out[:-tail], whole[:-tail], atol=1e-5)


@pytest.mark.parametrize("sizes", [(1,), (37, 101, 5), (640,)])
def test_a_windowed_checkpoint_does_not_depend_on_the_pushes(
        windowed_model, sizes):
    windowed_model.set_lookahead(2)
    clip = _clip(0.1, seed=5)
    reference = _run(streaming.Stream(windowed_model, block_frames=4), clip,
                     (301,))
    out = _run(streaming.Stream(windowed_model, block_frames=4), clip, sizes)
    assert torch.allclose(out, reference, atol=1e-5)


def test_a_cache_of_the_window_less_one_is_exact_and_a_shorter_one_bounds(
        windowed_model):
    windowed_model.set_lookahead(0)
    clip = _clip(0.1)
    window = windowed_model.config.attention_window_frames
    with torch.inference_mode():
        whole = windowed_model(clip[None])[0]
    exact = _run(
        streaming.Stream(windowed_model,
                         block_frames=4,
                         cache_frames=window - 1), clip, (640,))
    short = _run(
        streaming.Stream(windowed_model,
                         block_frames=4,
                         cache_frames=window // 2), clip, (640,))
    tail = 2 * windowed_model.config.n_fft
    assert torch.allclose(exact[:-tail], whole[:-tail], atol=1e-5)
    assert not torch.allclose(short[:-tail], whole[:-tail], atol=1e-4)


def test_windowed_lanes_run_on_a_windowed_checkpoint(windowed_model):
    windowed_model.set_lookahead(2)
    clip = _clip(0.1, seed=6)
    out = _run(
        streaming.Stream(windowed_model,
                         block_frames=4,
                         window_frames=40,
                         crossfade_frames=10), clip, (37, 101, 5))
    assert out.shape == clip.shape and torch.isfinite(out).all()


@pytest.mark.parametrize("sizes", [(301,), (37, 101, 5), (1,)])
def test_compacting_the_attention_buffer_changes_nothing_but_rounding(
        dual_model, sizes):
    dual_model.set_lookahead(2)
    clip = _clip(seconds=0.4, seed=11)
    compacting = streaming.Stream(dual_model, block_frames=4, cache_frames=8)
    roomy = streaming.Stream(dual_model, block_frames=4, cache_frames=8)
    config = dual_model.config
    roomy._attention = streaming._AttentionCache(  # pylint: disable=protected-access
        8, 4096, config.width // config.num_heads,
        config.attention_window_frames, clip.device)

    out = _run(compacting, clip, sizes)

    reference = _run(roomy, clip, sizes)
    assert compacting._attention.capacity < clip.shape[-1] // config.hop_length  # pylint: disable=protected-access
    assert torch.allclose(out, reference, atol=1e-5)


@pytest.mark.parametrize("batch,frames", [(1, 1), (3, 4)])
def test_the_packed_bands_equal_the_model_encoders_and_decoder_heads(
        dual_model, batch, frames):
    dual_model.set_lookahead(2)
    packed = streaming._PackedBands(dual_model)  # pylint: disable=protected-access
    g = torch.Generator().manual_seed(12)
    bins = dual_model.config.n_fft // 2 + 1
    spectrum = torch.complex(torch.randn(batch, bins, frames, generator=g),
                             torch.randn(batch, bins, frames, generator=g))
    features = torch.randn(batch,
                           len(dual_model.edges),
                           frames,
                           dual_model.config.width,
                           generator=g)

    with torch.inference_mode():
        encoded = packed.encode(dual_model, spectrum)
        decoded = packed.decode(features)
        expected_encoded = dual_model.encode(spectrum)
        expected_decoded = dual_model.denoise_decoder(features)

    assert encoded.shape == expected_encoded.shape
    assert torch.allclose(encoded, expected_encoded, atol=1e-5)
    assert decoded.dtype == torch.float32
    assert decoded.shape == expected_decoded.shape
    assert torch.allclose(decoded, expected_decoded, atol=1e-5)


def test_streams_of_one_model_share_its_packed_band_weights(dual_model):
    dual_model.set_lookahead(2)

    first = streaming.Stream(dual_model)
    second = streaming.Stream(dual_model, block_frames=7)

    # pylint: disable-next=protected-access
    assert first._bands is second._bands


def test_a_new_stream_follows_band_weights_changed_in_place(dual_model):
    dual_model.set_lookahead(2)
    clip = _clip()
    before = _run(streaming.Stream(dual_model), clip, (480,))
    changed = {
        name:
            tensor * 1.5 if name.startswith(("encoders.0.", "denoise_decoder."
                                             "heads.0.")) else tensor
        for name, tensor in dual_model.state_dict().items()
    }
    dual_model.load_state_dict(changed)
    with torch.inference_mode():
        whole = dual_model(clip[None])[0]

    after = _run(streaming.Stream(dual_model), clip, (480,))

    tail = 2 * dual_model.config.n_fft
    assert not torch.allclose(after[:-tail], before[:-tail], atol=1e-4)
    assert torch.allclose(after[:-tail], whole[:-tail], atol=1e-5)


def test_a_model_built_in_inference_mode_streams(tiny):
    with torch.inference_mode():
        model = tiny.dual()
        model.set_lookahead(2)
    clip = _clip()
    with torch.inference_mode():
        whole = model(clip[None])[0]

    out = _run(streaming.Stream(model), clip, (480,))

    tail = 2 * model.config.n_fft
    assert torch.allclose(out[:-tail], whole[:-tail], atol=1e-5)
