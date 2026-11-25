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
"""The output-gain convention, read one raw measurement at a time."""

import math

import numpy as np
import pytest

import levels


def test_the_gain_factor_undoes_a_level_given_in_db():
    assert levels.gain(0.0) == 1.0
    assert levels.gain(20.0) == pytest.approx(0.1)
    assert levels.gain(-6.0) == pytest.approx(10**(6 / 20))


def test_suppression_reads_the_residual_with_the_gain_undone():
    r = np.array([1.0, -0.5, 0.25, 0.75])
    m = r + np.array([0.3, 0.1, -0.2, 0.05])
    louder = 2.0 * r
    energies = [
        np.mean(louder * louder),
        np.mean(louder * r),
        np.mean(r * r),
        np.mean((m - r)**2),
    ]
    # the reference itself, 6 dB louder: nothing is left once the gain is
    # undone, and the reference's own energy is left when it is not
    assert levels.suppression_db(energies, 20 * math.log10(2.0)) < -60.0
    raw = 10 * math.log10(np.mean(r * r) / np.mean((m - r)**2))
    assert levels.suppression_db(energies, 0.0) == pytest.approx(raw, abs=1e-6)


def test_energy_ratios_and_the_high_bands_read_the_gain_undone():
    assert levels.energy_db(-3.0, 6.0) == -9.0
    # the three high bands 6, 4 and 8 dB over the reference, under a 6 dB gain
    bands = [0.0] * 4 + [6.0, 4.0, 8.0]
    assert levels.hf_error_db(bands, 6.0) == pytest.approx(4 / 3)
    assert math.isnan(levels.hf_error_db([0.0] * 4 + [math.nan] * 3, 0.0))


def test_the_level_change_is_the_gain_itself_in_absolute_db():
    assert levels.level_error_db(-2.5) == 2.5
    assert levels.level_error_db(3.0) == 3.0


def test_lost_blocks_and_deleted_segments_are_read_with_the_gain_undone():
    kept = [-5.0, -5.0, -12.0]
    assert levels.lost(kept, 0.0).tolist() == [False, False, True]
    # the same levels read off an output 6 dB louder than its input: every
    # block sits 11 dB or more under the recording once the gain is undone
    assert levels.lost(kept, 6.0).tolist() == [True, True, True]
    assert not levels.lost([], 0.0).size
    assert levels.deleted([-5.0, -12.0], 0.0) == [0.0, 1.0]
    assert levels.deleted([-5.0, -12.0], 6.0) == [1.0, 1.0]
