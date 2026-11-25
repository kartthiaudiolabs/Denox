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
"""Input conversion and resampling contracts of the public API."""

# pylint: disable=redefined-outer-name

import os
import stat

import numpy as np
import pytest
import soundfile as sf
import torch

from inworld.voice_isolation import audio as audio_lib


@pytest.fixture
def umask():
    """Runs the test under umask 027, then restores the process's own."""
    previous = os.umask(0o027)
    yield 0o027
    os.umask(previous)


def test_int16_pcm_is_scaled_to_unit_range():
    x = np.array([-32768, 0, 16384, 32767], dtype=np.int16)
    y = audio_lib.to_mono(x)
    assert y.dtype == torch.float32
    assert torch.allclose(y, torch.tensor([-1.0, 0.0, 0.5, 32767 / 32768]))


def test_uint8_pcm_is_centred():
    y = audio_lib.to_mono(np.array([0, 128, 255], dtype=np.uint8))
    assert torch.allclose(y, torch.tensor([-1.0, 0.0, 127 / 128]))


def test_float_input_passes_through_and_stereo_is_downmixed():
    # [C, T]
    stereo = np.stack([np.ones(100, np.float32), -np.ones(100, np.float32)])
    assert torch.allclose(audio_lib.to_mono(stereo), torch.zeros(100))
    # [T, C]
    assert torch.allclose(audio_lib.to_mono(stereo.T), torch.zeros(100))
    t = torch.linspace(-0.5, 0.5, 50)
    assert torch.equal(audio_lib.to_mono(t), t)


def test_to_mono_rejects_3d():
    with pytest.raises(ValueError):
        audio_lib.to_mono(np.zeros((2, 3, 4), np.float32))


def test_resample_identity_and_length():
    x = torch.randn(48000)
    assert audio_lib.resample(x, 48000, 48000) is x
    y = audio_lib.resample(x, 48000, 16000)
    assert abs(len(y) - 16000) <= 1


def test_save_audio_scales_pcm_instead_of_clipping(tmp_path):
    y = torch.zeros(1000)
    y[100], y[200] = 1.5, -0.75
    audio_lib.save_audio(str(tmp_path / "pcm.wav"), y, 48000)
    z, _ = audio_lib.load_audio(str(tmp_path / "pcm.wav"))
    assert (abs(float(z.abs().max()) - 1.0) < 1e-3 and
            abs(float(z[200]) + 0.5) < 1e-3)
    audio_lib.save_audio(str(tmp_path / "f32.wav"), y, 48000, subtype="FLOAT")
    z32, _ = audio_lib.load_audio(str(tmp_path / "f32.wav"))
    assert float(z32[100]) == pytest.approx(1.5)


def test_int64_and_more_than_eight_channels_are_refused():
    with pytest.raises(ValueError, match="int64"):
        audio_lib.to_mono(np.zeros(100, np.int64))
    with pytest.raises(ValueError, match="at most eight channels"):
        audio_lib.to_mono(np.zeros((16, 48000), np.float32))
    assert audio_lib.to_mono(np.zeros((8, 48000), np.float32)).shape == (48000,)


def test_save_audio_leaves_nothing_behind_when_a_write_fails(
        tmp_path, monkeypatch):
    write = sf.write

    def full_disk(path, data, *args, **kwargs):
        write(path, data[:10], *args, **kwargs)
        raise OSError("No space left on device")

    monkeypatch.setattr(sf, "write", full_disk)
    with pytest.raises(OSError):
        audio_lib.save_audio(str(tmp_path / "out.wav"), torch.zeros(1000),
                             48000)
    assert not list(tmp_path.iterdir())


def test_saved_audio_gets_the_permissions_of_any_new_file(tmp_path, umask):
    path = tmp_path / "out.wav"

    audio_lib.save_audio(str(path), torch.zeros(100), 48000)

    assert oct(stat.S_IMODE(path.stat().st_mode)) == oct(0o666 & ~umask)
