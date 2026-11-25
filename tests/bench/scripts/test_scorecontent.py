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
"""Word alignment, word classes and the verified panel.

Rates are pooled, and changes are paired.
"""

# pylint: disable=unused-argument

import json
import math
import os
import zlib

import numpy as np
import pytest
import soundfile as sf

import common
import observations
import scoreasr
import scorecontent


class _Norm:

    def tokens(self, text, language):
        clean = "".join(
            ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
        return clean.split()


class _Asr:
    """Transcribes by reading ``<audio>.txt``."""

    tag = "fake"
    identity = {"engine": "fake", "model": "fake"}

    def transcribe(self, path, language=None):
        return open(path + ".txt", encoding="utf-8").read()


def test_alignment_counts_kept_substituted_deleted_and_inserted():
    ref = "the cat sat on the mat".split()
    assert scorecontent.align(ref, ref) == (["kept"] * 6, 0)
    ops, ins = scorecontent.align("the cat on the mat".split(), ref)
    assert ops == ["kept", "kept", "del", "kept", "kept", "kept"] and ins == 0
    ops, ins = scorecontent.align("the cat sat on a big mat".split(), ref)
    assert ops == ["kept", "kept", "kept", "kept", "sub", "kept"] and ins == 1
    assert scorecontent.align([], ref) == (["del"] * 6, 0)
    assert scorecontent.align("x y".split(), []) == ([], 2)


def test_word_classes():
    assert [
        scorecontent.word_class(t)
        for t in ("1984", "3rd", "at", "the", "cat", "walk", "idea", "yes")
    ] == [
        "number",
        "number",
        "short",
        "short",
        "short",
        "consonant-final",
        "other",
        "short",
    ]
    assert (scorecontent.word_class("morning") == "consonant-final" and
            scorecontent.word_class("sofa") == "other")


def test_the_panel_is_verified_english_paired_items_of_the_named_conditions():
    rows = [
        {
            "id": "a.wav",
            "condition": "clean",
            "language": "en",
            "clean_path": "r",
        },
        {
            "id": "b.wav",
            "condition": "clean",
            "language": "en",
            "clean_path": "r",
        },
        {
            "id": "c.wav",
            "condition": "clean",
            "language": "de",
            "clean_path": "r",
        },
        {
            "id": "d.wav",
            "condition": "wild@-5",
            "language": "en",
            "clean_path": "r",
        },
        {
            "id": "e.wav",
            "condition": "clean",
            "language": "en"
        },
    ]
    golden = {
        "a.wav": {
            "tier": "corpus-verified"
        },
        "b.wav": {
            "tier": "recognisers-agree"
        },  # machine agreement only: not verified
        "c.wav": {
            "tier": "corpus-verified"
        },
        "d.wav": {
            "tier": "corpus-verified"
        },
        "e.wav": {
            "tier": "gemini-accepted"
        },
    }

    def panel_ids(conditions):
        return [r["id"] for r in scorecontent.panel(rows, golden, conditions)]

    assert panel_ids(("clean",)) == ["a.wav"]
    golden["b.wav"]["tier"] = "human-verified"
    assert panel_ids(("clean", "wild@-5")) == [
        "a.wav",
        "b.wav",
        "d.wav",
    ]


def _set(tmp_path, monkeypatch):
    root, out = tmp_path / "set", tmp_path / "out"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    texts = [
        "the cat sat on the mat at 10",
        "we walked home past the shop",
        "call me at 7 tonight",
    ]
    rows, golden = [], {}
    for i, text in enumerate(texts):
        rid = f"clean_{i:03d}.wav"
        x = (0.1 * np.sin(np.arange(12000) * (i + 1) / 50)).astype(np.float32)
        for d in ("mix", "ref"):
            sf.write(str(root / d / rid), x, 48000, subtype="FLOAT")
            path = str(root / d / rid) + ".txt"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        rows.append({
            "id": rid,
            "condition": "clean",
            "language": "en",
            "clean_path": str(root / "ref" / rid),
            "mix_path": str(root / "mix" / rid),
        })
        golden[rid] = {
            "id": rid,
            "language": "en",
            "text": text,
            "tier": "corpus-verified",
        }
    json.dump(
        {
            "set_id": "tiny",
            "n_items": len(rows)
        },
        open(root / "checksums.json", "w", encoding="utf-8"),
    )
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(out))
    return rows, golden, out


def _render(rows, out, name, fn):
    (out / name).mkdir(parents=True, exist_ok=True)
    for r in rows:
        x, sr = sf.read(r["clean_path"], dtype="float32")
        # perturbed by a stable hash of the transcript: outputs that transcribe
        # differently never share bytes (the cache keys on bytes), and the same
        # text always gives the same bytes
        sf.write(
            str(out / name / r["id"]),
            x * (1 + 1e-6 * (zlib.crc32(fn(r).encode()) % 997 + 1)),
            sr,
            subtype="FLOAT",
        )
        path = str(out / name / r["id"]) + ".txt"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(fn(r))


def test_rates_are_pooled_per_class_and_paired_against_the_input(
        tmp_path, monkeypatch):
    rows, golden, out = _set(tmp_path, monkeypatch)
    asr = _Asr()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    norm = _Norm()
    _render(rows, out, "perfect", lambda r: golden[r["id"]]["text"])
    _render(
        rows,
        out,
        "dropsshort",
        lambda r: " ".join(t for t in golden[r["id"]]["text"].split()
                           if len(t) > 3 or t.isdigit()),
    )

    # numbers gone; a spurious word inserted early, away from them, so it
    # reads as an insertion, not a substitution
    def without_numbers(r):
        kept = [t for t in golden[r["id"]]["text"].split() if not t.isdigit()]
        return " ".join(kept[:1] + ["extra"] + kept[1:])

    _render(rows, out, "dropsnumbers", without_numbers)
    inp = scorecontent.score_system("INPUT", rows, asr, cache, norm, golden)
    perfect = scorecontent.score_system("perfect", rows, asr, cache, norm,
                                        golden)
    short = scorecontent.score_system("dropsshort", rows, asr, cache, norm,
                                      golden)
    numbers = scorecontent.score_system("dropsnumbers", rows, asr, cache, norm,
                                        golden)
    assert (inp["rates"]["all"]["del_pct"] == 0.0 and
            perfect["rates"]["all"]["del_pct"] == 0.0)
    assert (short["rates"]["short"]["del_pct"] == 100.0 and
            short["rates"]["number"]["del_pct"] == 0.0)
    assert (short["rates"]["consonant-final"]["del_pct"] == 0.0 and
            short["rates"]["all"]["sub_pct"] == 0.0)
    assert (numbers["rates"]["number"]["del_pct"] == 100.0 and
            numbers["rates"]["short"]["del_pct"] == 0.0)
    # 19 reference words
    assert numbers["rates"]["all"]["ins_pct"] == pytest.approx(100.0 * 3 / 19)
    lo, hi = short["rates"]["all"]["del_ci95"]
    assert lo <= short["rates"]["all"]["del_pct"] <= hi
    d = scorecontent.paired_delta(short, inp)
    assert d["n_items"] == 3 and d["delta"] > 0 and d["ci95"][0] > 0
    assert scorecontent.paired_delta(perfect, inp)["delta"] == 0.0
    md = scorecontent.markdown({
        "INPUT": inp,
        "dropsshort": short,
        "perfect": perfect
    })
    assert "| INPUT | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |" in md
    assert "| dropsshort | 3 |" in md and "| perfect | 3 | 0.0 |" in md
    # controls go in their own table, read against the untouched reference they
    # were made from
    md = scorecontent.markdown({
        "INPUT": inp,
        "ctl_reference": perfect,
        "ctl_delete80ms": short
    })
    tables = md.split("deletions vs reference")
    assert len(tables) == 2 and "| ctl_reference | 3 | 0.0 |" in tables[1]
    assert "| ctl_delete80ms | 3 |" in tables[1] and "ctl_" not in tables[0]


def test_controls_are_rendered_from_the_panel_references_and_stamped(
        tmp_path, monkeypatch):
    rows, _, out = _set(tmp_path, monkeypatch)
    monkeypatch.setattr(common, "RESULTS", str(tmp_path / "results"))
    (tmp_path / "results").mkdir()
    systems = scorecontent.render_controls(rows, ("reference", "delete80ms"))
    assert systems == ["ctl_reference", "ctl_delete80ms"]
    for name in systems:
        assert sorted(p.name for p in (out / name).iterdir()) == sorted(
            r["id"] for r in rows)
        meta = json.load(
            open(tmp_path / "results" / f"{name}.meta.json", encoding="utf-8"))
        assert meta["set_id"] == "tiny" and meta["n_files"] == len(rows)
        assert len(meta["controls_sha256"]) == 64
    ref, _ = sf.read(rows[0]["clean_path"], dtype="float32")
    kept, _ = sf.read(out / "ctl_reference" / rows[0]["id"], dtype="float32")
    cut, _ = sf.read(out / "ctl_delete80ms" / rows[0]["id"], dtype="float32")
    assert np.array_equal(kept, ref) and (cut == 0).sum() >= int(0.076 * 48000)


def test_control_renders_are_reused_only_while_sources_and_code_are_unchanged(
        tmp_path, monkeypatch):
    rows, _, out = _set(tmp_path, monkeypatch)
    monkeypatch.setattr(common, "RESULTS", str(tmp_path / "results"))
    (tmp_path / "results").mkdir()
    scorecontent.render_controls(rows, ("reference",))
    first = {
        r["id"]: os.stat(out / "ctl_reference" / r["id"]).st_mtime_ns
        for r in rows
    }
    # nothing changed: every file is kept as it is
    scorecontent.render_controls(rows, ("reference",))
    assert {
        r["id"]: os.stat(out / "ctl_reference" / r["id"]).st_mtime_ns
        for r in rows
    } == first
    # the reference of one item changes: that item alone is rendered again
    x, sr = sf.read(rows[1]["clean_path"], dtype="float32")
    sf.write(rows[1]["clean_path"], x * 0.5, sr, subtype="FLOAT")
    scorecontent.render_controls(rows, ("reference",))
    again = {
        r["id"]: os.stat(out / "ctl_reference" / r["id"]).st_mtime_ns
        for r in rows
    }
    assert again[rows[1]["id"]] != first[rows[1]["id"]]
    assert all(again[r["id"]] == first[r["id"]] for r in (rows[0], rows[2]))
    y, _ = sf.read(out / "ctl_reference" / rows[1]["id"], dtype="float32")
    assert np.array_equal(y, x * 0.5)
    # the controls' implementation changes: everything is rendered again
    monkeypatch.setattr(
        observations,
        "implementation",
        lambda *paths: "another implementation",
    )
    scorecontent.render_controls(rows, ("reference",))
    third = {
        r["id"]: os.stat(out / "ctl_reference" / r["id"]).st_mtime_ns
        for r in rows
    }
    assert all(third[r["id"]] != again[r["id"]] for r in rows)
    path = tmp_path / "results" / "ctl_reference.meta.json"
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["controls_sha256"] == "another implementation"


def _counts(n, dels, subs=0, cls="other"):
    cells = {
        c: {
            "n": 0,
            "kept": 0,
            "sub": 0,
            "del": 0
        } for c in scorecontent.CLASSES
    }
    cells[cls] = {"n": n, "kept": n - dels - subs, "sub": subs, "del": dels}
    return cells


def _panel_rows():
    """Six items from two recordings, each at three levels of one bed.

    The set repeats its recordings the same way.
    """
    rows = []
    for k, speech in enumerate(("recording-a", "recording-b")):
        for level in ("+4", "0", "-5"):
            rows.append({
                "id": f"babble{level}_{k}.wav",
                "condition": f"babble@{level}",
                "language": "en",
                "clean_path": "r",
                "mix_path": "m",
                "provenance": {
                    "speech": speech,
                    "corruptor": "bed-x@12.0s",
                },
            })
    return rows


def test_groups_are_opaque_labels_of_the_recording_and_the_bed():
    rows = _panel_rows()
    records = {
        r["id"]: {
            "condition": r["condition"],
            "classes": _counts(10, 1),
            "ins": 0,
        } for r in rows
    }
    scorecontent.label_groups(records, rows)
    speech = {records[r["id"]]["speech_group"] for r in rows}
    # two recordings, however many levels; nothing names a corpus
    assert speech == {"s0", "s1"}
    # one bed behind every item
    assert {records[r["id"]]["bed_group"] for r in rows} == {"b0"}
    first = records[rows[0]["id"]]["speech_group"]
    assert all(records[r["id"]]["speech_group"] == first for r in rows[:3])
    groups = scorecontent.groups_of(records)
    assert len(groups) == 2 and len(groups[0]) == 6


def test_intervals_resample_recordings_and_change_is_pooled_rate_difference():
    rows = _panel_rows()
    # recording a loses a word per item, recording b loses none: the deletion
    # rate depends on which recordings are drawn, so its interval spans from 0
    # to about a fifth
    left = {"items": {}, "missing": []}
    right = {"items": {}, "missing": []}
    for r in rows:
        dels = 2 if r["provenance"]["speech"] == "recording-a" else 0
        left["items"][r["id"]] = {
            "condition": r["condition"],
            "classes": _counts(10, dels),
            "ins": 0,
        }
        right["items"][r["id"]] = {
            "condition": r["condition"],
            "classes": _counts(10, 0),
            "ins": 0,
        }
    scorecontent.label_groups(left["items"], rows)
    scorecontent.label_groups(right["items"], rows)
    left["rates"] = scorecontent.pooled(left["items"])
    right["rates"] = scorecontent.pooled(right["items"])
    assert (left["rates"]["all"]["del_pct"] == 10.0 and
            right["rates"]["all"]["del_pct"] == 0.0)
    assert (
        left["rates"]["all"]["n_items"],
        left["rates"]["all"]["n_recordings"],
        left["rates"]["all"]["n_beds"],
    ) == (6, 2, 1)
    assert ("rest on the 2 recordings and 1 beds behind the 6 items"
            in scorecontent.markdown({
                "INPUT": right,
                "sys": left
            }))
    lo, hi = left["rates"]["all"]["del_ci95"]
    # only recording b drawn ... only recording a drawn
    assert lo == 0.0 and hi == 20.0
    d = scorecontent.paired_delta(left, right)
    assert (d["delta"] == 10.0 and d["ci95"] == [0.0, 20.0] and
            not d["incomplete"])
    assert d["n_items"] == 6
    # an item-level bootstrap would have read the six items as independent and
    # given a tighter interval; the clustered one cannot exclude zero from two
    # recordings
    items = np.array([20.0, 20.0, 20.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    boots = [items[rng.integers(0, 6, 6)].mean() for _ in range(1000)]
    assert np.percentile(boots, 2.5) > 0.0


def test_incomplete_cohorts_are_marked_never_intersected():
    rows = _panel_rows()
    full = {"items": {}, "missing": []}
    part = {"items": {}, "missing": [rows[-1]["id"]]}
    for r in rows:
        full["items"][r["id"]] = {
            "condition": r["condition"],
            "classes": _counts(10, 0),
            "ins": 0,
        }
        if r["id"] != rows[-1]["id"]:
            part["items"][r["id"]] = {
                "condition": r["condition"],
                "classes": _counts(10, 1),
                "ins": 0,
            }
    scorecontent.label_groups(full["items"], rows)
    scorecontent.label_groups(part["items"], rows[:-1])
    full["rates"] = scorecontent.pooled(full["items"])
    part["rates"] = scorecontent.pooled(part["items"])
    d = scorecontent.paired_delta(part, full)
    assert (d["incomplete"] and math.isnan(d["delta"]) and
            d["missing"] == [rows[-1]["id"]])
    md = scorecontent.markdown({"INPUT": full, "partial": part})
    assert ("| partial | 5 of 6 (incomplete) |" in md and
            "incomplete (1 items differ)" in md)
    assert "| INPUT | 6 |" in md


def test_missing_outputs_are_listed_not_skipped(tmp_path, monkeypatch):
    rows, golden, out = _set(tmp_path, monkeypatch)
    asr = _Asr()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    norm = _Norm()
    _render(rows[:2], out, "partial", lambda r: golden[r["id"]]["text"])
    res = scorecontent.score_system("partial", rows, asr, cache, norm, golden)
    assert res["missing"] == [rows[2]["id"]] and len(res["items"]) == 2
    assert math.isfinite(res["rates"]["all"]["del_pct"])


def test_incompatible_recognisers_or_reference_tokens_are_refused():
    rows = _panel_rows()
    left = {
        "items": {},
        "missing": [],
        "asr_identity": {
            "engine": "whisper",
            "model": "large-v3"
        },
    }
    right = {
        "items": {},
        "missing": [],
        "asr_identity": {
            "engine": "whisper",
            "model": "small"
        },
    }
    for r in rows:
        left["items"][r["id"]] = {
            "condition": r["condition"],
            "classes": _counts(10, 1),
            "ins": 0,
        }
        right["items"][r["id"]] = {
            "condition": r["condition"],
            "classes": _counts(10, 1),
            "ins": 0,
        }
    scorecontent.label_groups(left["items"], rows)
    scorecontent.label_groups(right["items"], rows)
    left["rates"] = scorecontent.pooled(left["items"])
    right["rates"] = scorecontent.pooled(right["items"])
    d = scorecontent.paired_delta(left, right)
    assert d["incompatible"] == "recognisers differ" and math.isnan(d["delta"])
    right["asr_identity"] = dict(left["asr_identity"])
    assert scorecontent.paired_delta(left, right)["delta"] == 0.0
    # equal pooled rates over different reference token counts: no difference
    # may be reported
    # 10 % as well, on four times the words
    right["items"][rows[0]["id"]]["classes"] = _counts(40, 4)
    right["rates"] = scorecontent.pooled(right["items"])
    d = scorecontent.paired_delta(left, right)
    assert d["incompatible"] == "reference tokens differ" and math.isnan(
        d["delta"])
    md = scorecontent.markdown({"INPUT": right, "sys": left})
    assert "incompatible (reference tokens differ)" in md
    # a loaded sidecar carries the identity under its underscore key
    sidecar = {"_asr_identity": {"engine": "x"}}
    assert scorecontent.asr_identity(sidecar) == {"engine": "x"}


def test_a_system_without_a_render_is_skipped_before_scoring(
        tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUT_ROOT", str(tmp_path / "out"))
    assert "no outputs" in scorecontent.missing_render("ghost")
    assert scorecontent.missing_render("INPUT") == ""
    os.makedirs(tmp_path / "out" / "ghost")
    assert scorecontent.missing_render("ghost") == ""
