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
"""The output-gain convention: one gain per system.

The gain is undone wherever level is not the question.

A system's gain is measured once, on clean speech (``level_cal_db``: the median
output-over-input level of the ``clean`` items, bounded at
``common.MAX_LEVEL_CAL_DB``), and scored once, as the level-change row of the
fidelity axis. Everywhere else the output is read as if that gain were undone:
the kept-energy rows, the lost blocks, suppression of a competing voice and of
non-voiced sound, the high-band error and deleted speech. These functions read
one item's raw measurements at a given calibration; the scorer, its sidecars
and the paired comparison all read through them, so one rule holds everywhere.

Per-item errors are absolute before they are averaged: a mean absolute error
whose signs could cancel would hide a dull copy behind a bright one.
"""

import math

import numpy as np

# a speech segment whose energy drops further than this is deleted
DEL_DB = -10.0
# a block kept further than this under the recording is lost: the floor of the
# preservation scale
LOST_DB = -10.0
HF_BANDS = slice(4, 7)  # 8-12, 12-16 and 16-24 kHz of common.BANDS_HZ


def gain(level_cal_db: float) -> float:
    """Returns the linear factor that undoes a system's output gain."""
    return 10.0**(-level_cal_db / 20.0)


def suppression_db(energies: list, level_cal_db: float) -> float:
    """Returns residual interference over input interference, in dB.

    Args:
        energies: Mean squares over the item, ``[|o|^2, <o, r>, |r|^2, |m -
            r|^2]`` for output o, reference r and mixture m.
        level_cal_db: The system's output gain, undone: |g o - r|^2 is
            g^2 |o|^2 - 2 g <o, r> + |r|^2.
    """
    oo, orr, rr, mr = (float(v) for v in energies)
    g = gain(level_cal_db)
    residual = max(g * g * oo - 2.0 * g * orr + rr, 0.0)
    return 10.0 * math.log10((residual + 1e-9) / (mr + 1e-9))


def energy_db(raw_db: float, level_cal_db: float) -> float:
    """Returns an output-over-input energy ratio with the output gain undone."""
    return float(raw_db) - level_cal_db


def hf_error_db(bands: list, level_cal_db: float) -> float:
    """Returns one item's mean absolute level error over the high bands.

    Args:
        bands: The item's output-over-reference level per band (dB).
        level_cal_db: The system's output gain, undone first.
    """
    high = np.asarray(bands[HF_BANDS], dtype=float)
    if not high.size or not np.isfinite(high).all():
        return float("nan")
    return float(np.mean(np.abs(high - level_cal_db)))


def level_error_db(out_in_rms_db: float) -> float:
    """Returns one item's level change, in absolute dB: the gain itself."""
    return abs(float(out_in_rms_db))


def lost(kept_db: list, level_cal_db: float) -> np.ndarray:
    """Returns, block by block, whether the output lost it.

    Args:
        kept_db: Each block's kept level: the output's energy coherent with the
            recording over the recording's, in dB.
        level_cal_db: The system's output gain, undone first.
    """
    return np.asarray(kept_db, dtype=float) - level_cal_db < LOST_DB


def deleted(drops: list, level_cal_db: float) -> list:
    """Returns 1.0 for each speech segment the output deleted, 0.0 otherwise."""
    return [1.0 if d - level_cal_db < DEL_DB else 0.0 for d in drops]
