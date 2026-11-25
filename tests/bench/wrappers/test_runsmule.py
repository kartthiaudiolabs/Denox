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
"""The Smule Renaissance wrapper renders 48 kHz inputs only.

It fills the release's short tail.
"""

# pylint: disable=unused-argument

import numpy as np
import soundfile as sf

import runsmule


def test_a_short_output_is_filled_and_another_rate_fails(
        tiny_set, run_wrapper, capsys):
    in_dir, out_dir = tiny_set
    sf.write(
        in_dir / "item_3.wav",
        np.zeros(16000, dtype="float32"),
        16000,
        subtype="FLOAT",
    )

    # the release's inverse STFT ends up to 2047 samples short
    def enhance(path, samples, rate):
        return samples[:len(samples) - 2047] * 0.5, rate

    calls = run_wrapper(runsmule, in_dir, out_dir, enhance=enhance)
    assert calls["exit"] == 1 and calls["begin"][0] == "smule"
    y, _ = sf.read(out_dir / "item_0.wav", dtype="float32")
    assert len(y) == 24000 and not y[-2047:].any() and y[:100].any()
    assert "FAIL item_3.wav" in capsys.readouterr().out
