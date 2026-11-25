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
"""Synthetic board rows placed by the anchor policy.

INPUT, REF and systems a, b and c sit at 0, 1, 0.7, 0.4 and 0.9 of every
declared gap (INPUT and REF on the endpoint each verifies) and at the same
fractions of a fixed scale where a metric declares none; b deletes one of the
deletion cohort's 20 segments.
"""

import anchors
import scorefiles


def scores() -> dict[str, dict]:
    """Returns the score dicts of INPUT, REF and systems a, b and c."""
    policy = anchors.definitions()
    systems = {}
    for name, value in (
        ("INPUT", 0),
        ("REF", 1),
        ("a", 0.7),
        ("b", 0.4),
        ("c", 0.9),
    ):
        data = {
            "_set_id": "set",
            "_subset": "all",
            **anchors.stamp(policy),
            "_deletion": {
                "items": {
                    "r0": [0, 8],
                    "r1": [int(name == "b"), 12]
                }
            },
        }
        for rule in policy["rules"]:
            for condition in rule["conditions"]:
                d = data.setdefault(condition, {"n": 24})
                for metric in rule["metrics"]:
                    low = rule["floor"].get("value")
                    high = rule["ceiling"].get("value")
                    # INPUT on the floor, REF on the ceiling, and each on
                    # the endpoint it verifies
                    if low is not None and high is not None:
                        v = low + value * (high - low)
                        for side in ("floor", "ceiling"):
                            verify = rule[side].get("verify") or {}
                            if verify.get("source") == name:
                                v = rule[side]["value"]
                    elif "spk_sim" in metric:
                        v = 0.5 + value * 0.5
                    elif "nisqa" in metric:
                        v = 2 + value * 2
                    elif "kept" in metric:
                        v = -10 + value * 10
                    elif metric == "_hf_err":
                        v = 1 - value
                    else:
                        v = value * 20 * rule["direction"]
                    d[metric] = v
        # the band-limited rows as a denoiser's, which the registry's rule
        # ranks on the denoising board: under its threshold, with the per-band
        # levels the row derives from; the input stays on its anchor
        for condition, first in scorefiles.BAND_LIMITED_ROWS:
            d = data.get(condition)
            if d is None or "band_added_db" not in d:
                continue
            if name != "INPUT":
                low = anchors.rule_for(policy, condition,
                                       "band_added_db")["floor"]["value"]
                d["band_added_db"] = low + value * (8.0 - low)
            d["band_kept_db"] = [0.0] * first + [d["band_added_db"]
                                                ] * (7 - first)
        systems[name] = data
    return systems
