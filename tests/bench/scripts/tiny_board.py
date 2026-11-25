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
"""Small synthetic boards for the board's tests.

Each board is a set of score files and a frozen calibration.
"""

import json
import os

import anchors
import ranking

FULL = tuple(
    sorted({
        c for rows in ranking.AXES_FULL.values() for c, _, _ in rows
        if not c.startswith("_")
    }))  # every condition the composite axes read


def metrics(v: float, n: int = 24) -> dict:
    """Returns complete synthetic metrics at scale ``v`` with ``n`` items."""
    d = {
        c: {
            "n": n,
            "babble_supp_db": -v,
            "babble_resid_db": -v,
            "babble_resid_db_nb": -v,
            "si_sdr": v,
            "kept_p10": v / 10,
            "kept_mean": v / 10,
            "kept_coh_p10": v / 10,
            "kept_coh_nb_p10": v / 10,
            "spk_sim": 0.9 + v / 1000,
            "nisqa_mos": 3 + v / 100,
            "nv_lost_pct": 20 - v,
            "mag_snr": 10 + v,
            "band_kept_db": [0, 0, 0, 0, v / 10, v / 10, v / 10],
            # what scorefiles.load derives from the bands above the input's
            "band_added_db": v / 10,
            "hf_err_db": abs(v) / 10,
            "out_in_rms_db": -v / 10,
            "level_err_db": abs(v) / 10,
            # the same-room block's rows (scoreevents): noise left, in dB
            "event_resid_db": -v,
            "floor_resid_db": -v,
        } for c in FULL
    }
    # the block pooled: the voice's level while the noise goes, at most zero
    d["_coroom"] = {"n": n, "voice_level_db": -abs(v) / 100}
    # the pooled rows of the real recordings (scoreresidual, scorefiles)
    d["_real_cleanup"] = {
        "n": n,
        "pause_resid_db": -v,
        "pause_resid_db_nb": -v,
        # the pauses against the output's own voice: the unprocessed recordings
        # sit at -20.7 dB, the cap at -45
        "pause_level_db": -20.7 - v,
    }
    d["_real_quality"] = {"n": n, "nisqa_mos": 3 + v / 100}
    d["_level_cal_db"] = 0.0
    d["_set_id"] = "tiny"
    d.update(anchors.stamp())
    return d


def write(results: str | os.PathLike[str],
          name: str,
          scores: dict,
          deletion_pct: float | None = None,
          items: dict[str, list[int]] | None = None) -> None:
    """Writes the score files of ``name``, with an optional deletion cohort.

    ``items`` is {id: [n_deleted, n_segments]}; ``deletion_pct`` builds a
    100-item cohort at that rate.
    """
    scores.setdefault("_set_id", "tiny")
    scores.update(anchors.stamp())
    json.dump(
        scores,
        open(os.path.join(results, f"{name}.metrics.json"),
             "w",
             encoding="utf-8"))
    if items is None and deletion_pct is not None:
        items = {
            f"r{i:03d}": [1 if i < deletion_pct else 0, 1] for i in range(100)
        }
    if items is not None:
        json.dump(
            {
                "_items_primary": items,
                "_set_id": scores["_set_id"],
                "_level_cal_db": scores.get("_level_cal_db"),
                **anchors.stamp(),
            },
            open(os.path.join(results, f"{name}.realx.json"),
                 "w",
                 encoding="utf-8"),
        )


def calibration(scores: dict, cohort: dict[str, int] | None = None) -> dict:
    """Resolves synthetic anchor endpoints and the supplied deletion cohort."""
    if cohort is None:
        items = next(
            (v["_deletion"]["items"]
             for v in scores.values()
             if v.get("_deletion", {}).get("items")),
            {},
        )
        cohort = {uid: values[1] for uid, values in items.items()}
    policy, refs = anchors.definitions(), {}
    for name, side in (("INPUT", "floor"), ("REF", "ceiling")):
        d = metrics(0)
        # the two anchor rows sit on the declared endpoints, so the policy's
        # verification against the set passes
        for rule in policy["rules"]:
            for cond in rule["conditions"]:
                at = next(
                    (
                        # a row sits on the endpoint it verifies
                        e
                        for e in ("floor", "ceiling")
                        if (rule[e].get("verify") or {}).get("source") == name),
                    side,
                )
                if cond in d and "value" in rule[at]:
                    d[cond].update({
                        metric: rule[at]["value"] for metric in rule["metrics"]
                    })
        d["_deletion"] = {
            "items": {
                uid: [0, count] for uid, count in cohort.items()
            }
        }
        refs[name] = d
    return anchors.resolve(policy, refs, ranking.AXES_BY_TIER)


def rank(scores: dict, tiers: dict) -> tuple[dict, dict, dict]:
    """Ranks synthetic scores with their resolved calibration."""
    for data in scores.values():
        data.setdefault("_set_id", "tiny")
        data.update(anchors.stamp())
    return ranking.rank(scores, tiers, calibration(scores))
