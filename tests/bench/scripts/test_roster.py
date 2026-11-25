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
"""The board's roster: every system in one tier, under a label of its own."""

import roster


def test_every_system_has_a_known_tier_and_a_label_of_its_own():
    tiers = {tier for tier, _ in roster.TIERS.values()}
    labels = [label for _, label in roster.TIERS.values()]
    assert tiers <= {"full", "nb", "ref"}
    assert len(set(labels)) == len(labels) and all(labels)
    # the unprocessed input and the clean reference are shown, never ranked
    assert roster.TIERS["INPUT"][0] == roster.TIERS["REF"][0] == "ref"


def test_a_tiers_file_adds_systems_and_relabels_registered_ones(tmp_path):
    path = tmp_path / "tiers.json"
    path.write_text(
        '{"candidate": ["full", "Candidate (1M)"], "dfn3": ["ref", "DFN3"]}',
        encoding="utf-8")
    tiers = roster.tiers(str(path))
    assert tiers["candidate"] == ("full", "Candidate (1M)")
    assert tiers["dfn3"] == ("ref", "DFN3")
    assert roster.tiers() == roster.TIERS
    assert roster.TIERS["dfn3"] == ("full", "DeepFilterNet3")


def test_display_name_drops_the_parenthesised_details():
    assert roster.display_name("voice_isolator_v0") == "Inworld Voice Isolator"
    assert roster.display_name("unregistered") == "unregistered"
