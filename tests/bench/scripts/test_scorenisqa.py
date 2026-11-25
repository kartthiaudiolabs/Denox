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
"""The scorenisqa sidecar imports cleanly.

It applies the render-identity rule.
"""

import pytest

import scorenisqa


def test_module_applies_the_identity_rule_of_scorebench():
    # one rule decides which renders a sidecar may read: scorebench's
    assert not hasattr(scorenisqa, "render_problem")
    assert "render_problem" in scorenisqa.main.__code__.co_names


def test_a_familys_eligible_seconds_sum_the_items_that_carry_them():
    """An item without primary segments revives to NaN; the family's eligible
    duration is the sum over the others."""
    rows = [{"id": str(i), "condition": "real_x"} for i in range(3)]
    records = [{
        "id": "0",
        "metrics": {
            "nisqa_mos": 3.0,
            "nisqa_mos_primary": 3.1,
            "nisqa_primary_eligible_s": 4.0,
            "nisqa_primary_coverage": 0.9
        }
    }, {
        "id": "1",
        "metrics": {
            "nisqa_mos": 3.5,
            "nisqa_mos_primary": 3.6,
            "nisqa_primary_eligible_s": 6.0,
            "nisqa_primary_coverage": 0.8
        }
    }, {
        "id": "2",
        "metrics": {
            "nisqa_mos": 2.5
        }
    }]

    cell = scorenisqa.aggregate(records, rows, ["real_x"], 0)["real_x"]

    assert cell["nisqa_primary_items"] == 2
    assert cell["nisqa_primary_eligible_s"] == pytest.approx(10.0)
