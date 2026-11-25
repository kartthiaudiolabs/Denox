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
"""Bench-script tests import the scripts as plain modules."""

import os
import sys

import pytest

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "bench", "scripts"))
# the wrappers' receipt module, which the scripts read render receipts with
sys.path.insert(0, os.path.join(REPO, "bench", "wrappers"))


def suppression_energies(value_db, scale=1e6):
    """Returns per-item energies whose suppression is ``value_db``.

    |o|^2, <o, r>, |r|^2 and |m - r|^2 of an output that is the voice plus the
    chatter scaled to ``value_db`` (orthogonal, of equal energy), with no
    output gain: both the suppression row (levels.suppression_db) and the
    residual row (scoreresidual.babble_metrics) read ``value_db`` from them,
    and the unchanged mixture reads zero. Large enough that the epsilons do
    not show.
    """
    return [scale * (1.0 + 10**(value_db / 10)), scale, scale, scale]


@pytest.fixture
def composite_case():
    """Small complete score/observation bundles.

    Their checkpoints differ in known ways.
    """

    def create(root, set_id="public", sealed=False):
        # pylint: disable-next=import-outside-toplevel
        import anchors
        # pylint: disable-next=import-outside-toplevel
        import observations
        # pylint: disable-next=import-outside-toplevel
        import ranking
        # pylint: disable-next=import-outside-toplevel
        import scorefiles
        # pylint: disable-next=import-outside-toplevel
        import scoreresidual

        policy = anchors.definitions()
        scores, systems = {}, {}
        for name, fraction in (
            ("INPUT", 0.0),
            ("REF", 1.0),
            ("candidate", 0.7),
            ("baseline", 0.5),
        ):
            score = {
                "_set_id": set_id,
                "_subset": "all",
                **anchors.stamp(policy),
            }
            for rule in policy["rules"]:
                for condition in rule["conditions"]:
                    if condition.startswith("_"):
                        continue
                    cell = score.setdefault(condition, {
                        "n": 24,
                        "n_missing": 0
                    })
                    for metric in rule["metrics"]:
                        low = rule["floor"].get("value")
                        high = rule["ceiling"].get("value")
                        # declared endpoints: INPUT sits on the floor, REF on
                        # the ceiling, and each on the endpoint it verifies
                        if low is not None and high is not None:
                            value = low + fraction * (high - low)
                            for side in ("floor", "ceiling"):
                                verify = rule[side].get("verify") or {}
                                if verify.get("source") == name:
                                    value = rule[side]["value"]
                        elif "spk_sim" in metric:
                            value = 0.5 + fraction * 0.5
                        elif "nisqa" in metric:
                            value = 2 + fraction * 2
                        elif "kept" in metric:
                            value = -10 + fraction * 10
                        elif metric == "_hf_err":
                            value = 1 - fraction
                        elif condition == "clean":
                            value = 10 + fraction * 30
                        elif rule["direction"] == -1:
                            value = -20 * fraction
                        else:
                            value = -5 + 25 * fraction
                        cell[metric] = value
            # the band-limited rows: denoisers, as the registry's assignment
            # reads them (under its threshold), with the per-band levels the
            # row derives from; the input stays on its anchor
            for condition, first in scorefiles.BAND_LIMITED_ROWS:
                cell = score.get(condition)
                if cell is None or "band_added_db" not in cell:
                    continue
                if name != "INPUT":
                    low = anchors.rule_for(policy, condition,
                                           "band_added_db")["floor"]["value"]
                    cell["band_added_db"] = low + fraction * (8.0 - low)
                cell["band_kept_db"] = ([0.0] * first +
                                        [cell["band_added_db"]] * (7 - first))
            # the real recordings' pooled rows: every family between the
            # pooled row's endpoints, so the pooled value sits there too
            quality = anchors.rule_for(policy, "_real_quality", "nisqa_mos")
            cleanup = anchors.rule_for(policy, "_real_cleanup",
                                       "pause_resid_db")
            level = anchors.rule_for(policy, "_real_cleanup", "pause_level_db")
            for family in ("real_dirty", "real_multi", "real_noisy",
                           "real_wild"):
                cell = score.setdefault(family, {"n": 24, "n_missing": 0})
                low, high = (quality["floor"]["value"],
                             quality["ceiling"]["value"])
                cell["nisqa_mos"] = low + fraction * (high - low)
                low, high = (cleanup["floor"]["value"],
                             cleanup["ceiling"]["value"])
                cell["pause_resid_db"] = low + fraction * (high - low)
                cell["pause_resid_db_nb"] = cell["pause_resid_db"]
                low, high = level["floor"]["value"], level["ceiling"]["value"]
                cell["pause_level_db"] = low + fraction * (high - low)
            score["_real_quality"] = {
                "n": 24,
                "nisqa_mos": score["real_noisy"]["nisqa_mos"],
            }
            score["_real_cleanup"] = {
                "n": 96,
                "pause_resid_db": score["real_noisy"]["pause_resid_db"],
                "pause_resid_db_nb": score["real_noisy"]["pause_resid_db"],
                "pause_level_db": score["real_noisy"]["pause_level_db"],
            }
            # the same-room block pooled (scoreevents), once the policy
            # declares it: between its endpoints, the anchor rows on theirs
            try:
                voice = anchors.rule_for(policy, "_coroom", "voice_level_db")
            except ValueError:
                voice = None
            if voice is not None:
                low, high = voice["floor"]["value"], voice["ceiling"]["value"]
                value = low + fraction * (high - low)
                for side in ("floor", "ceiling"):
                    if (voice[side].get("verify") or {}).get("source") == name:
                        value = voice[side]["value"]
                score["_coroom"] = {
                    "n": 96,
                    "n_missing": 0,
                    "voice_level_db": value,
                }
                # every item of the block carries the row it pools
                for condition in voice.get(
                        "pooled_from", ()) or ("coroom@pause", "coroom@speech",
                                               "coroom@floor", "coroom@both"):
                    if condition in score:
                        score[condition]["voice_level_db"] = value
            rows, cohort = [], {}
            for condition, cell in sorted(score.items()):
                if condition.startswith("_"):
                    continue
                for i in range(24):
                    uid = f"{condition}_{i}.wav"
                    metrics = {
                        k: v
                        for k, v in cell.items()
                        if k not in ("n", "n_missing")
                    }
                    if condition == "clean":
                        hf_err = cell["_hf_err"]
                        metrics["band_kept_db"] = [0] * 4 + [hf_err] * 3
                        metrics["out_in_rms_db"] = -cell["_level_err"]
                    if "band_added_db" in cell:
                        # the per-band levels the band row derives from: the
                        # bands above the input hold the cell's value, those
                        # inside it nothing
                        first = dict(scorefiles.BAND_LIMITED_ROWS)[condition]
                        metrics["band_kept_db"] = ([0] * first +
                                                   [cell["band_added_db"]] *
                                                   (7 - first))
                    # pooled on the board: counts per item
                    if "nv_lost_pct" in metrics:
                        metrics.update(
                            nv_blocks=100,
                            nv_lost=int(round(metrics["nv_lost_pct"])),
                        )
                    # read from energies: the suppression row by the energies
                    # that give its value, the residual row by what those
                    # energies read as (scoreresidual.babble_metrics)
                    for key in ("babble_supp_db", "babble_supp_db_nb"):
                        if key in metrics:
                            energy = suppression_energies(metrics[key])
                            metrics[key.replace("supp_db", "energy")] = energy
                            resid = key.replace("supp_db", "resid_db")
                            metrics[resid] = scoreresidual.babble_metrics(
                                energy)
                            cell[resid] = metrics[resid]
                    if condition == "real_noisy":
                        deleted = int(name == "baseline" and i == 0)
                        cohort[uid] = [deleted, 1]
                        metrics.update(
                            primary_drops=[-30.0 if deleted else 0.0],
                            primary_segments=1,
                        )
                    rows.append({
                        "id": uid,
                        "condition": condition,
                        "language": "en",
                        "speech_source": f"{set_id}-speech-{i}",
                        "bed_source": (None if condition == "clean" else
                                       f"{set_id}-bed-{i % 6}"),
                        "groups_verified": True,
                        "output_status": "ok",
                        # what the record measured, as every scorer stamps it
                        "identity": {
                            "output": f"{name}-{uid}",
                            "mix": f"mix-{uid}",
                            "ref": (None if condition.startswith("real") else
                                    f"ref-{uid}"),
                            "semantics": {
                                "condition": condition
                            },
                        },
                        "metrics": metrics,
                    })
            score["_deletion"] = {"items": cohort, "primary_anchored": True}
            score["_level_cal_db"] = 0.0
            scores[name] = score
            metadata = {
                "system": name,
                "set_id": set_id,
                "subset": "all",
                "sealed": sealed,
                "anchors_version": policy["version"],
                "anchors_sha256": anchors.digest(policy),
                "scorer_sha256": "fixture",
                "level_cal_db": 0,
                "n_expected": len(rows),
                "render_identity": {
                    "weights_sha256": anchors.digest(name),
                    "precision": "fp32",
                },
            }
            systems[name] = metadata, rows
            observations.write(root / f"{name}.items.jsonl", rows, metadata)
        calibration = anchors.resolve(policy, scores, ranking.AXES_BY_TIER,
                                      sealed)
        anchors.atomic_json(root / anchors.RESOLVED, calibration)
        return scores, calibration, systems

    return create


@pytest.fixture
def saturate():
    """Rewrites a system's per-item header with a saturated level calibration.

    Takes the folder ``composite_case`` wrote, the system and the calibration in
    dB (12 by default).
    """

    def rewrite(root, system, level_cal_db=12.0):
        # pylint: disable-next=import-outside-toplevel
        import observations

        path = root / f"{system}.items.jsonl"
        metadata, rows = observations.read(path)
        metadata = {
            k: v
            for k, v in metadata.items()
            if k not in ("type", "schema_version")
        }
        metadata["level_cal_db"] = level_cal_db
        observations.write(path, rows, metadata)

    return rewrite
