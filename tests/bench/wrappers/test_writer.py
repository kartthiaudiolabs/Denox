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
"""The shared render loop of the wrappers.

One process's inputs, outputs fitted to the input's length and written whole or
not at all, failures reported and counted in the exit status.
"""

import os

import numpy as np
import pytest
import soundfile as sf

import writer


def test_inputs_are_sorted_sharded_and_limited(tiny_set):
    in_dir, _ = tiny_set
    names = [os.path.basename(p) for p in writer.inputs(str(in_dir))]
    assert names == ["item_0.wav", "item_1.wav", "item_2.wav"]
    assert [os.path.basename(p) for p in writer.inputs(str(in_dir), "1/2")
           ] == ["item_1.wav"]
    assert len(writer.inputs(str(in_dir), limit=2)) == 2
    with pytest.raises(ValueError):
        writer.inputs(str(in_dir), "2/2")


def test_fit_pads_or_trims_frame_rounding_and_refuses_a_cut_off_output():
    x = np.ones(1000, dtype=np.float64)
    assert writer.fit(x, 1010, 48000).dtype == np.float32
    assert np.array_equal(
        writer.fit(x, 1010, 48000)[1000:], np.zeros(10, dtype=np.float32))
    assert len(writer.fit(x, 990, 48000)) == 990
    with pytest.raises(ValueError, match="cut-off"):
        writer.fit(x, 1000 + int(0.06 * 48000), 48000)


def test_save_writes_float32_whole_or_not_at_all(tmp_path):
    path = tmp_path / "a.wav"
    writer.save(str(path), np.full(480, 0.123456789), 48000)
    info = sf.info(str(path))
    assert info.subtype == "FLOAT" and info.frames == 480
    assert sf.read(str(path), dtype="float32")[0][0] == np.float32(0.123456789)
    with pytest.raises(ValueError, match="NaN"):
        writer.save(str(tmp_path / "b.wav"), np.array([0.0, np.nan]), 48000)
    # no output, no partial file
    assert sorted(os.listdir(tmp_path)) == ["a.wav"]


def test_read_averages_channels(tmp_path):
    sf.write(
        tmp_path / "s.wav",
        np.stack([np.ones(10), np.zeros(10)], 1),
        48000,
        subtype="FLOAT",
    )
    samples, rate = writer.read(str(tmp_path / "s.wav"))
    assert (rate == 48000 and samples.shape == (10,) and
            samples[0] == np.float32(0.5))


def test_render_resumes_reports_failures_and_exits_nonzero_unless_all_present(
        tiny_set, capsys):
    in_dir, out_dir = tiny_set
    files = writer.inputs(str(in_dir))
    os.makedirs(out_dir)
    sf.write(
        out_dir / "item_0.wav",
        np.ones(24000, dtype="float32"),
        48000,
        subtype="FLOAT",
    )
    seen = []

    def enhance(path, samples, rate):
        seen.append(os.path.basename(path))
        if path.endswith("item_2.wav"):
            # a cut-off output fails, never padded
            return samples[:len(samples) // 2], rate
        return np.concatenate([samples, np.zeros(37, dtype="float32")]), rate

    assert writer.render(files, str(out_dir), enhance) == 1
    # the existing output is kept, not re-run
    assert seen == ["item_1.wav", "item_2.wav"]
    assert sf.read(out_dir / "item_0.wav")[0].min() == 1.0
    assert sf.info(str(out_dir / "item_1.wav")).frames == 36000
    assert not (out_dir / "item_2.wav").exists()
    out = capsys.readouterr().out
    assert "FAIL item_2.wav" in out and "DONE 2/3" in out
    assert writer.render(files, str(out_dir), lambda p, s, r: (s, r)) == 0
    assert "DONE 3/3" in capsys.readouterr().out


def test_render_refuses_another_input_rate(tiny_set, capsys):
    in_dir, out_dir = tiny_set
    assert (writer.render(
        writer.inputs(str(in_dir)),
        str(out_dir),
        lambda p, s, r: (s, r),
        expect_rate=16000,
    ) == 1)
    assert "expected 16000" in capsys.readouterr().out and not list(
        out_dir.glob("*.wav"))
