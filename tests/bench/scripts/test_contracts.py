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
"""Every condition has one contract.

The composite scores what the contract states and no more.
"""

import os

import pytest

import contracts
import ranking
import validateset


def test_every_family_has_exactly_one_profile_and_unknown_ones_have_none():
    families = [f for group in contracts.FAMILIES.values() for f in group]
    assert len(families) == len(set(families)) and set(
        contracts.FAMILIES) == set(contracts.PROFILES)
    assert (contracts.profile("babble@-5") == "denoise" and
            contracts.profile("clean") == "denoise")
    assert (contracts.profile("singer@-2") == "voice_policy" and
            contracts.profile("reverb_real") == "restoration")
    assert (contracts.profile("nonverbal@+0") == "denoise" and
            contracts.profile("nonverbal_clean") == "denoise")
    assert contracts.profile("_deletion") == "denoise"
    with pytest.raises(KeyError):
        # a new family needs a stated contract, never a default
        contracts.profile("underwater@+0")


def test_every_condition_the_validator_expects_has_a_contract():
    for condition in validateset.EXPECTED_INVENTORY:
        assert contracts.profile(condition) in contracts.PROFILES


def test_composite_scores_denoising_and_the_room_axis_in_every_view_and_tier():
    for view in ranking.AXES_VIEWS.values():
        for axes in view.values():
            contracts.check_axes(axes)
    contracts.check_axes({"room": [("reverb_real", "spk_sim", 1)]})
    # a laugh is the voice
    contracts.check_axes(
        {"preservation": [("nonverbal@+0", "nv_lost_pct", -1)]})
    with pytest.raises(ValueError, match="restoration"):
        contracts.check_axes({"fidelity": [("reverb_real", "si_sdr", 1)]})
    with pytest.raises(ValueError, match="voice_policy"):
        contracts.check_axes(
            {"preservation": [("singer@-5", "kept_coh_p10", 1)]})
    # the room axis is rooms only
    with pytest.raises(ValueError, match="scores restoration"):
        contracts.check_axes(
            {"room": [("reverb_real", "spk_sim", 1), ("clean", "spk_sim", 1)]})
    with pytest.raises(ValueError, match="denoise only"):
        contracts.check_axes({"room": [("reverb_real", "spk_sim", 1)]},
                             {"denoise": None})
    assert (contracts.axes_statement() ==
            "the axes score denoise rows and restoration rows in the room "
            "axis alone")


def test_conventions_quote_scorers_own_bounds_and_the_readme_states_contracts():
    text = " ".join(" ".join(row) for row in contracts.CONVENTIONS)
    assert "100 ms" in text and "12 dB" in text and "polarity" in text
    grouped = contracts.by_profile(
        ["babble@+4", "singer@-5", "reverb_synth", "clean", "babble@+4"])
    assert grouped == {
        "denoise": ["babble@+4", "clean"],
        "voice_policy": ["singer@-5"],
        "restoration": ["reverb_synth"],
    }
    path = os.path.join(os.path.dirname(contracts.__file__), "..", "README.md")
    with open(path, encoding="utf-8") as fh:
        readme = fh.read()
    section = readme.split("### Task contracts", 1)[1].split("\n## ", 1)[0]
    families = contracts.FAMILIES
    for family in families["voice_policy"] + families["restoration"]:
        assert f"`{family}" in section, family
