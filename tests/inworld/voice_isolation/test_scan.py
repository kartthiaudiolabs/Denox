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
"""The selective scan: chunks against the frame-by-frame recurrence."""

# pylint: disable=protected-access
import pytest
import torch

from inworld.voice_isolation import model
from inworld.voice_isolation import scan


def _inputs(batch, frames, channels, state, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    return dict(
        u=rand(batch, frames, channels),
        delta=torch.nn.functional.softplus(rand(batch, frames, channels) * 2),
        a=-torch.exp(rand(channels, state)),
        b=rand(batch, frames, state),
        c=rand(batch, frames, state),
        d=rand(channels),
        z=rand(batch, frames, channels),
    )


@pytest.mark.parametrize("frames", [1, 2, 7, 64, 203])
def test_the_scan_matches_the_recurrence(frames, monkeypatch):
    monkeypatch.setattr(scan, "BLOCK", 5)
    kw = _inputs(3, frames, 5, 4)
    y, h = scan.selective_scan(**kw)
    ref, href = scan.selective_scan_reference(**kw)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h, href, atol=1e-5, rtol=1e-5)
    assert y.dtype == torch.float32 and y.shape == kw["u"].shape


def test_a_carried_state_splits_the_scan_anywhere(monkeypatch):
    monkeypatch.setattr(scan, "BLOCK", 3)
    kw = _inputs(2, 40, 6, 3, seed=1)
    whole, final = scan.selective_scan(**kw)
    parts, state = [], None
    for start, stop in ((0, 1), (1, 17), (17, 18), (18, 40)):
        piece = {
            k: (v[:, start:stop] if v.ndim == 3 else v) for k, v in kw.items()
        }
        y, state = scan.selective_scan(**piece, state=state)
        parts.append(y)
    assert torch.allclose(torch.cat(parts, 1), whole, atol=1e-5)
    assert torch.allclose(state, final, atol=1e-5)


def _metal_or_skip():
    if not torch.backends.mps.is_available():
        pytest.skip("the Metal scan runs on Apple's GPU")
    if scan.metal_kernel() is None:
        pytest.skip("this PyTorch cannot compile Metal kernels")


@pytest.mark.parametrize("frames", [1, 2, 46, 203])
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_the_metal_scan_matches_the_recurrence(frames, carried, gated):
    _metal_or_skip()
    kw = _inputs(3, frames, 37, 16, seed=frames)
    if not gated:
        kw["z"] = None
    state = torch.randn(3, 37, 16) if carried else None

    y, h = scan.selective_scan(**{
        k: None if v is None else v.to("mps") for k, v in kw.items()
    },
                               state=None if state is None else state.to("mps"))

    ref, href = scan.selective_scan_reference(**kw, state=state)
    assert y.device.type == "mps" and y.shape == kw["u"].shape
    assert torch.allclose(y.cpu(), ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h.cpu(), href, atol=1e-5, rtol=1e-5)


def test_more_states_than_the_metal_scan_holds_take_the_torch_scan():
    _metal_or_skip()
    kw = _inputs(2, 9, 5, scan.METAL_STATES + 1, seed=5)

    y, _ = scan.selective_scan(**{k: v.to("mps") for k, v in kw.items()})

    ref, _ = scan.selective_scan_reference(**kw)
    assert torch.allclose(y.cpu(), ref, atol=1e-5, rtol=1e-5)


def test_fast_decays_do_not_overflow():
    kw = _inputs(1, 50, 4, 2)
    kw["delta"] = kw["delta"] * 100
    y, _ = scan.selective_scan(**kw)
    assert torch.isfinite(y).all()
    ref, _ = scan.selective_scan_reference(**kw)
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)


def test_the_mamba_module_carries_its_convolution_and_state():
    torch.manual_seed(0)
    mamba = model.Mamba(8, 4, 3, 2).eval()
    with torch.no_grad():
        for p in mamba.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(2, 25, 8)
    with torch.inference_mode():
        whole, (tail, state) = mamba(x)
        first, carried = mamba(x[:, :1])
        assert carried[0].shape == (2, 16, 2)
        second, carried = mamba(x[:, 1:9], carried)
        third, (tail2, state2) = mamba(x[:, 9:], carried)
    assert torch.allclose(torch.cat([first, second, third], 1),
                          whole,
                          atol=1e-5)
    assert torch.allclose(tail2, tail, atol=1e-5) and torch.allclose(
        state2, state, atol=1e-5)
    assert tail.shape == (2, 16, 2)


def test_the_two_sided_block_is_causal_only_when_streaming():
    torch.manual_seed(0)
    block = model.BidirectionalMamba(8, 4, 3, 2, causal=False).eval()
    with torch.no_grad():
        for p in block.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(1, 30, 8)
    x2 = x.clone()
    x2[:, 20:] += 1.0
    with torch.inference_mode():
        assert torch.allclose(block(x, causal=True)[:, :20],
                              block(x2, causal=True)[:, :20],
                              atol=1e-6)
        assert not torch.allclose(
            block(x)[:, :20], block(x2)[:, :20], atol=1e-4)


def test_the_kernel_path_matches_the_recurrence_when_installed():
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("the fused kernels run on CUDA")
    assert scan.kernel() is not None
    kw = {k: v.cuda() for k, v in _inputs(3, 150, 8, 4, seed=2).items()}
    y, h = scan.selective_scan(**kw)
    ref, href = scan.selective_scan_reference(**kw)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h, href, atol=1e-5, rtol=1e-5)


def test_the_fused_kernels_agree_when_installed():
    mamba_ssm = pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("the fused kernels run on CUDA")
    torch.manual_seed(0)
    ours = model.Mamba(32, 16, 4, 2).cuda().eval()
    theirs = mamba_ssm.Mamba(d_model=32, d_state=16, d_conv=4,
                             expand=2).cuda().eval()
    theirs.load_state_dict(ours.state_dict())
    x = torch.randn(3, 100, 32, device="cuda")
    with torch.inference_mode():
        y, _ = ours(x)
        ref = theirs(x)
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)
