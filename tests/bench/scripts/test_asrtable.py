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
"""The downstream-ASR table from synthetic sidecars."""

import json
import math
import os

import pytest

import asrtable


def _block(wer, n=10):
    return {"wer_pct": wer, "ci95": [wer - 1.0, wer + 1.0], "n": n, "items": {}}


def _sidecar(results, system, tag, conds, set_id="s1", **extra):
    doc = {
        "_system": system,
        "_set_id": set_id,
        "_scorer_sha256": "scorer",
        "_selection": {
            "conds": ",".join(conds),
            "limit": 0
        },
        "_outputs": {
            f"{c}.wav": f"{system}-{c}" for c in conds
        },
        "asr": {
            tag: {
                "identity": {
                    "model": tag
                },
                "conditions": {
                    c: _block(w) for c, w in conds.items()
                },
            }
        },
    }
    doc.update(extra)
    json.dump(
        doc,
        open(os.path.join(results, f"{system}.asr.json"), "w",
             encoding="utf-8"))


def test_load_collects_one_recogniser_and_refuses_mixed_builds(tmp_path):
    _sidecar(tmp_path, "INPUT", "large-v3", {"clean": 0.0, "babble@-10": 40.0})
    _sidecar(
        tmp_path,
        "voice_isolator_v0",
        "large-v3",
        {
            "clean": 0.5,
            "babble@-10": 20.0
        },
    )
    # another recogniser: not in this view
    _sidecar(tmp_path, "other", "small", {"clean": 1.0})
    scores = asrtable.load(str(tmp_path), "large-v3")
    assert set(scores) == {"INPUT", "voice_isolator_v0"}
    _sidecar(tmp_path, "stale", "large-v3", {"clean": 9.0}, set_id="s0")
    with pytest.raises(SystemExit):
        asrtable.load(str(tmp_path), "large-v3")


def test_table_marks_the_best_and_the_relative_change():
    scores = {
        "INPUT": {
            "clean": _block(0.0),
            "babble@-10": _block(40.0)
        },
        "voice_isolator_v0": {
            "clean": _block(0.5),
            "babble@-10": _block(20.0)
        },
        "reuse": {
            "babble@-10": _block(25.0)
        },
    }
    md = asrtable.table(
        scores,
        ["voice_isolator_v0", "reuse"],
        "voice_isolator_v0",
        "large-v3",
        label=lambda n: n,
    )
    lines = md.splitlines()
    assert lines[2].startswith(
        "| condition | unprocessed input | voice_isolator_v0 | reuse |"
        " voice_isolator_v0 vs input |")
    row = [l for l in lines if l.startswith("| `babble@-10`")][0]
    assert ("**20.0 ± 1.0**" in row and "40.0 ± 1.0" in row and
            row.rstrip().endswith("| -50 % |"))
    clean = [l for l in lines if l.startswith("| `clean`")][0]
    # no change against zero
    assert "**0.0 ± 1.0**" in clean and clean.rstrip().endswith("| - |")
    assert "| - |" in clean or "| - " in clean  # reuse has no clean row: a dash
    assert math.isnan(asrtable.relative_change(None, _block(1.0)))
    assert asrtable.relative_change(_block(30.0),
                                    _block(40.0)) == pytest.approx(-25.0)


def test_cell_without_interval_shows_the_rate_alone():
    b = {"wer_pct": 12.34, "ci95": [float("nan"), float("nan")]}
    assert asrtable.cell(b) == "12.3" and asrtable.cell(None) == "-"


def test_reference_phrase_says_what_the_transcripts_were_scored_against():
    golden = {
        "clean": {
            "reference_tiers": {
                "corpus-verified": 6,
                "recognisers-agree": 3
            }
        }
    }
    mixed = {
        "clean": {
            "reference_tiers": {
                "corpus-verified": 6
            }
        },
        "wild@-5": {
            "reference_tiers": {
                "recognisers-agree": 2,
                asrtable.FALLBACK_TIER: 2
            }
        },
    }
    fallback = {"clean": {"reference_tiers": {asrtable.FALLBACK_TIER: 4}}}
    assert asrtable.reference_phrase(golden) == "the set's golden transcripts"
    assert "8 of 10 items" in asrtable.reference_phrase(
        mixed) and "other 2" in asrtable.reference_phrase(mixed)
    # no golden text anywhere: say so
    for conditions in (fallback, {"clean": {}}, {}):
        assert (asrtable.reference_phrase(conditions) ==
                "the same recogniser's transcript of the clean reference")
    scored = {c: _block(1.0) | tiers for c, tiers in mixed.items()}
    md = asrtable.table({"INPUT": scored}, [], "x", "large-v3")
    assert "golden transcripts (8 of 10 items" in md.splitlines()[0]


def test_slices_keep_the_two_contracts_apart_and_pair_by_source():
    rows, golden = [], {}
    for cond in ("babble@+4", "babble@-5", "conversation@0"):
        for i in range(6):
            rid = f"{cond.replace('@', 'AT')}_{i:03d}.wav"
            rows.append({
                "id": rid,
                "condition": cond,
                "language": "en",
                "clean_path": "x",
                "provenance": {
                    "speech": f"s{i % 3}",
                    "corruptor": f"b{i % 2}",
                },
            })
    scores = {}
    for name, factor in (("INPUT", 4), ("voice_isolator_v0", 1), ("other", 2)):
        scores[name] = {}
        for cond in ("babble@+4", "babble@-5", "conversation@0"):
            items = {
                r["id"]: [factor * (1 + i % 2), 20]
                for i, r in enumerate(rows)
                if r["condition"] == cond
            }
            scores[name][cond] = {"items": items}
    md = asrtable.slices(
        {"large-v3": scores},
        rows,
        golden,
        ["voice_isolator_v0", "other"],
        "voice_isolator_v0",
        min_items=6,
    )
    assert "| a conversation" not in asrtable.slices(
        {"large-v3": scores},
        rows,
        golden,
        ["other"],
        "other",
        # six items: under the floor, not
        # shown
        min_items=7,
    )
    lines = [
        l for l in md.splitlines()
        if l.startswith("| background") or l.startswith("| a conversation")
    ]
    assert len(lines) == 2  # one recogniser, English only: words
    chatter, conversation = lines
    # the babble ladder's items, the conversation's
    assert "| 12 |" in chatter and "| 6 |" in conversation
    assert "**7.5**" in chatter  # the isolator: 1.5 edits per 20 tokens
    # paired change against the input: 30.0 down to 7.5 points
    assert "-22.5 [" in chatter
    assert "conversation@0" not in chatter


def test_load_refuses_partial_unbound_or_mixed_scorer_sidecars(tmp_path):
    _sidecar(tmp_path, "INPUT", "large-v3", {"clean": 0.0})
    asrtable.load(str(tmp_path), "large-v3")
    _sidecar(
        tmp_path,
        "sys",
        "large-v3",
        {"clean": 1.0},
        _selection={
            "conds": "clean",
            "limit": 5
        },
    )
    with pytest.raises(SystemExit, match="--limit"):
        asrtable.load(str(tmp_path), "large-v3")
    _sidecar(tmp_path, "sys", "large-v3", {"clean": 1.0}, _outputs=None)
    with pytest.raises(SystemExit, match="no audio identity"):
        asrtable.load(str(tmp_path), "large-v3")
    _sidecar(tmp_path,
             "sys",
             "large-v3", {"clean": 1.0},
             _scorer_sha256="another")
    with pytest.raises(SystemExit, match="different implementations"):
        asrtable.load(str(tmp_path), "large-v3")


def test_row_missing_items_or_on_other_items_reads_incomplete_and_never_best():
    full = dict(_block(40.0), items={"a.wav": [4, 10], "b.wav": [4, 10]})
    scores = {
        "INPUT": {
            "babble@-10": full
        },
        "voice_isolator_v0": {
            "babble@-10": dict(full, wer_pct=30.0)
        },
        "reuse": {
            "babble@-10": dict(full, wer_pct=10.0, items={"a.wav": [1, 10]})
        },
        "dfn3": {
            "babble@-10": dict(full, wer_pct=5.0, n_missing=1)
        },
    }
    md = asrtable.table(
        scores,
        ["voice_isolator_v0", "reuse", "dfn3"],
        "voice_isolator_v0",
        "large-v3",
    )
    row = next(l for l in md.splitlines() if l.startswith("| `babble@-10`"))
    # the best of the complete rows
    assert row.count("incomplete") == 2 and "**30.0" in row
