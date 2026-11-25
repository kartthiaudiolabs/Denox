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
"""The FastEnhancer wrapper runs the export frame by frame, on the CPU only.

It flushes the export's state and removes the algorithmic delay.
"""

# pylint: disable=unused-argument

import sys

import numpy as np
import pytest

import runfastenhancer


class _Port:

    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _DelayLine:
    """A streaming export that returns its input N_FFT - HOP samples late.

    Through its cache, as the export's recurrent state does.
    """

    def get_inputs(self):
        return [
            _Port("wav_in", [1, runfastenhancer.HOP]),
            _Port("cache", [1, runfastenhancer.N_FFT - runfastenhancer.HOP]),
        ]

    def get_outputs(self):
        return [
            _Port("wav_out", [1, runfastenhancer.HOP]),
            _Port("cache_out",
                  [1, runfastenhancer.N_FFT - runfastenhancer.HOP]),
        ]

    def run(self, names, feeds):
        line = np.concatenate(
            [feeds["cache"].reshape(-1), feeds["wav_in"].reshape(-1)])
        return [
            line[:runfastenhancer.HOP].reshape(1, -1),
            line[runfastenhancer.HOP:].reshape(1, -1)
        ]


def test_the_output_is_aligned_with_the_input_and_covers_its_whole_length():
    n = 48000 * 2 + 37
    x = np.random.default_rng(0).standard_normal(n).astype("float32")
    y = runfastenhancer.stream(_DelayLine(), x)
    assert y.shape == x.shape and y.dtype == np.float32
    # delay removed, tail flushed: nothing lost, nothing shifted
    assert np.array_equal(y, x)


def test_every_input_is_rendered_and_only_the_cpu_is_accepted(
        tiny_set, run_wrapper, monkeypatch):
    in_dir, out_dir = tiny_set
    calls = run_wrapper(runfastenhancer, in_dir, out_dir, "--threads", "3")
    assert (calls["exit"] == 0 and calls["load"] == (3,) and
            calls["begin"][5] == "cpu")
    monkeypatch.setattr(
        sys,
        "argv",
        ["w", "--in-dir", "a", "--out-dir", "b", "--device", "cuda"],
    )
    with pytest.raises(SystemExit):
        runfastenhancer.main()
