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
"""The board as markdown tables."""

import ranking
import tables


def test_markdown_table_lists_full_tier_in_order_with_axes_and_headline_metrics(
):
    scores = {
        "x": {
            "babble@+4": {
                "babble_supp_db": -12.04,
                "babble_resid_db": -12.04,
                "spk_sim": 0.9876,
                "nisqa_mos": 4.0,
            },
            "nonverbal@+0": {
                "nv_lost_pct": 4.63
            },
            "_real_cleanup": {
                "pause_level_db": -15.24
            },
            "srsweep_8k": {
                "band_kept_db": [0, 0, 0, -24.0, -47.0, -47.0, -47.0]
            },
            "_deletion": {
                "pct": 0.2
            },
        },
        "y": {
            "babble@+4": {
                "babble_supp_db": -10.1,
                "babble_resid_db": -10.1,
                "spk_sim": 0.98,
                "nisqa_mos": 4.5,
            },
            "_real_cleanup": {
                "pause_level_db": -8.0
            },
            "srsweep_8k": {
                "band_kept_db": [0, 0, 0, -24.0, -47.0, -47.0, -47.0]
            },
            "_deletion": {
                "pct": 0.44
            },
        },
    }
    axis_scores = {
        "x": {
            a: 1.0 for a in ranking.AXES_FULL
        },
        "y": {
            a: 0.5 for a in ranking.AXES_FULL
        }
    }
    md = tables.markdown_table(
        scores,
        axis_scores,
        {
            "x": 0.7777,
            "y": 0.5
        },
        {"full": ["x", "y"]},
        {
            "x": ("full", "X label"),
            "y": ("full", "Y label")
        },
    )
    lines = md.splitlines()
    assert (lines[0].startswith(
        "| # | system | precision | denoising | removes | keeps | sounds |"
        " supp | music | cleanup | events | voices | pres | fidel | timbre |"
        " qual | band |") and len(lines) == 4)
    assert lines[0].endswith(
        "| supp dB | pauses dB | del % | laughs lost % | NISQA |")
    assert lines[2].startswith(
        "| 1 | X label | – | **1.000** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |")
    assert lines[2].endswith("| -12.0 | -15.2 | 0.2 | 4.6 | 4.00 |")
    assert lines[3].startswith("| 2 | Y label | – | **0.500** | 0.50 |")
    assert len({len(line.split("|")) for line in lines}) == 1
    stream = {"x": {"verdict": "yes"}, "y": {"verdict": "nonsense"}}
    md = tables.markdown_table(
        scores,
        axis_scores,
        {
            "x": 0.7777,
            "y": 0.5
        },
        {"full": ["x", "y"]},
        {
            "x": ("full", "X label"),
            "y": ("full", "Y label")
        },
        stream,
    )
    # the mark after a name: the streaming probe says yes, with the note
    assert "| 1 | X label\\* | – | **1.000** |" in md
    assert "| 2 | Y label | – | **0.500** |" in md
    assert md.endswith(tables.STREAMING_NOTE)


def test_the_16k_tier_gets_its_own_table_only_when_it_has_an_entrant():
    scores = {
        "z": {
            "babble@+4": {
                "babble_supp_db_nb": -9.04,
                "babble_resid_db_nb": -9.04,
                "spk_sim_nb": 0.9512,
                "nisqa_mos": 3.9,
            },
            "_real_cleanup": {
                "pause_level_db": -7.5
            },
            "srsweep_8k": {
                "band_kept_db": [0, 0, 0, -24.0, -47.0, -47.0, -47.0]
            },
            "_deletion": {
                "pct": 1.2
            },
        }
    }
    axis_scores = {"z": {a: 0.4 for a in ranking.AXES_NB}}
    tiers = {"z": ("nb", "Z label (16k)")}
    assert (tables.markdown_table_nb(scores, axis_scores, {"z": 0.4}, {
        "full": [],
        "nb": []
    }, tiers) == "")
    md = tables.markdown_table_nb(scores, axis_scores, {"z": 0.4}, {
        "full": [],
        "nb": ["z"]
    }, tiers)
    lines = md.splitlines()
    assert lines[0].startswith(
        "| # | system | precision | denoising | removes | keeps | sounds |")
    assert "supp dB (0–8 kHz)" in lines[0]
    assert lines[2].startswith("| 1 | Z label (16k) | – | **0.400** | 0.40 |")
    assert lines[2].endswith("| -9.0 | -7.5 | 1.2 | – | 3.90 |")
    assert len(lines[0].split("|")) == len(lines[2].split("|"))


def test_outside_tables_can_add_an_axis_in_its_own_order(monkeypatch):
    monkeypatch.setattr(ranking, "OUTSIDE", [])
    axis_scores = {
        "x": {
            "room": 0.2,
            "music": 0.9
        },
        "y": {
            "room": 0.8,
            "music": 0.1
        }
    }
    tiers = {"x": ("full", "X"), "y": ("full", "Y")}
    assert tables.markdown_outside(axis_scores, {"full": ["x", "y"]},
                                   tiers) == ""
    md = tables.markdown_outside(axis_scores, {"full": ["x", "y"]}, tiers,
                                 ["room"])
    assert md.splitlines()[0] == "| # | system | room |"
    assert (md.splitlines()[2] == "| 1 | Y | 0.80 |" and
            md.splitlines()[3] == "| 2 | X | 0.20 |")


def test_each_board_lists_its_own_systems_and_the_mark_means_streaming():
    scores = {
        "z": {
            "babble@+4": {
                "babble_resid_db": -9.0,
                "nisqa_mos": 3.9
            },
            "_real_cleanup": {
                "pause_level_db": -7.5
            },
            "_deletion": {
                "pct": 1.2
            },
        }
    }
    # a system that restores above the input's band
    scores["r"] = dict(scores["z"],
                       srsweep_8k={"band_kept_db": [0, 0, 0, 40, 30, 20, 10]})
    axis_scores = {
        "z": {
            a: 0.4 for a in ranking.AXES_FULL
        },
        "r": {
            a: 0.6 for a in ranking.AXES_FULL
        },
    }
    tiers = {"z": ("full", "Z"), "r": ("full", "R")}
    order = {"full": ["z"], "nb": [], "restoration": ["r"]}
    md = tables.markdown_table(scores, axis_scores, {
        "z": 0.4,
        "r": 0.6
    }, order, tiers)
    # no mark without the probe, no note
    assert "| 1 | Z |" in md and "| R |" not in md
    assert not md.endswith(tables.STREAMING_NOTE)
    md = tables.markdown_table(scores, axis_scores, {
        "z": 0.4,
        "r": 0.6
    }, order, tiers, {"z": {
        "verdict": "yes"
    }})
    assert "| 1 | Z\\* |" in md and md.endswith(tables.STREAMING_NOTE)
    both = tables.markdown_boards(scores, axis_scores, {
        "z": 0.4,
        "r": 0.6
    }, order, tiers)
    denoising, restoration = both.split("**Restoration**")
    assert denoising.startswith("**Denoising**") and "| 1 | Z |" in denoising
    assert "| R |" not in denoising and "| 1 | R |" in restoration
    assert "more than 10 dB on the band-limited rows" in restoration
    assert "| precision | restoration | restores | room | band |" in restoration
