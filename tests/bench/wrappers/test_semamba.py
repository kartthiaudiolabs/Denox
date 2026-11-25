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
"""The SEMamba family's runner, with the release's packages faked.

The packages bound by path, the STFT scaled to the input rate, mostly-silent
frames blanked, the length matched to the input.
"""

# pylint: disable=redefined-outer-name

import sys

import numpy as np
import pytest

import semamba

STFTS = """
import torch
CALLS = []
def mag_phase_stft(x, n_fft, hop_size, win_size, compress_factor, center,
                   addeps):
    CALLS.append((n_fft, hop_size, win_size))
    frames = x.shape[-1] // hop_size + 1
    mag = torch.ones(1, n_fft // 2 + 1, frames)
    return mag, torch.zeros_like(mag), None
def mag_phase_istft(amp, pha, n_fft, hop, win, compress):
    return amp.sum(1).repeat_interleave(hop, dim=-1)
"""
UTIL = """
import torch
def pad_or_trim_to_match(x, y, pad_value):
    return torch.nn.functional.pad(
        y, (0, max(0, x.shape[-1] - y.shape[-1])),
        value=pad_value)[..., :x.shape[-1]]
"""
CFG = {
    "stft_cfg": {
        "sampling_rate": 8000,
        "n_fft": 320,
        "hop_size": 40,
        "win_size": 320,
    },
    "model_cfg": {
        "compress_factor": 0.3
    },
}


@pytest.fixture
def release(tmp_path):
    for package, module, text in (
        ("models", "stfts", STFTS),
        ("utils", "util", UTIL),
    ):
        (tmp_path / package).mkdir()
        (tmp_path / package / f"{module}.py").write_text(text)
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k.split(".")[0] in ("models", "utils")
    }
    yield semamba.bind(str(tmp_path))
    for k in [k for k in sys.modules if k.split(".")[0] in ("models", "utils")]:
        del sys.modules[k]
    sys.modules.update(saved)


def test_sizes_scale_with_the_input_rate_and_stay_even():
    for base, rate, expected in (
        (320, 48000, 1920),
        (40, 48000, 240),
        (320, 44100, 1764),
        (160, 22050, 442),
        (320, 8000, 320),
    ):
        assert semamba.make_even(base * rate / 8000) == expected
    assert semamba.make_even(3) == 4 and semamba.make_even(4) == 4


def test_release_runs_at_the_input_rate_blanks_silent_frames_and_keeps_length(
    release,):
    util, stfts = release

    def model(mag, pha):
        amp = mag.clone()
        # frame 1: more than half of its bins silent
        amp[:, :amp.shape[1] // 2 + 2, 1] = 0
        return amp, pha, None

    enhance = semamba.enhancer(model, util, stfts, CFG, "cpu")
    x = np.random.default_rng(0).standard_normal(48000).astype("float32")
    y, rate = enhance("a.wav", x, 48000)
    assert stfts.CALLS[-1] == (1920, 240, 1920) and rate == 48000
    assert y.dtype == np.float32 and y.shape == x.shape
    # frame 1 blanked whole, frame 2 kept
    assert not y[240:480].any() and y[480:720].all()
    enhance("b.wav", x[:16000], 16000)
    assert stfts.CALLS[-1] == (640, 80, 640)
