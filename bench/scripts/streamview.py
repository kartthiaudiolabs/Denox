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
"""The streaming probe on the board: its sidecars, its cell and its table.

The probe is scored by scorestream.py; a sidecar is shown only while its set,
rule, scorer and the render it compared with are the board's. The scorer
records every item's look-ahead, drift and dropped seconds under its rule,
which fixes how renders are read; the board judges those measurements with
the limits declared here, so a limit can move without reading a render again
and every system is judged alike.
"""

import json
import os
import sys

import scorefiles
import scorestream

STREAM_EXT = ".stream.json"
# the board's verdict policy over the recorded per-item measurements
LOOK_AHEAD_MAX_MS = scorestream.LOOK_AHEAD_MAX_MS
DRIFT_MEAN_DB = scorestream.DRIFT_MEAN_DB
DRIFT_MIN_DB = -5.0  # no item under this (the scorer records under -3)


def rule_sentence() -> str:
    """One sentence saying what the streaming probe column reads."""
    return (
        "the streaming probe is yes when the output depends on at most"
        f" {LOOK_AHEAD_MAX_MS:.0f} ms of future input and"
        f" {scorestream.TAIL_SECONDS:.0f} s of a 45 s item score within"
        f" {-DRIFT_MEAN_DB:.0f} dB SI-SDR (no item under"
        f" {-DRIFT_MIN_DB:.0f} dB, and no second of speech lost that a fresh"
        " start keeps) of the same"
        f" stretch processed after a {scorestream.LEAD_IN:.1f} s lead-in")


def judged(side: dict) -> dict:
    """Returns the sidecar with the board's verdict over its recorded items.

    A sidecar without per-item measurements keeps the scorer's verdict.
    """
    items = [
        it for it in side.get("items", [])
        if "drift_db" in it and "look_ahead_ms" in it
    ]
    if not items:
        return side
    ahead = max(float(it["look_ahead_ms"]) for it in items)
    drifts = [float(it["drift_db"]) for it in items]
    mean, low = sum(drifts) / len(drifts), min(drifts)
    lost = sum(it.get("dropped_seconds", 0) for it in items)
    speech = sum(it.get("speech_seconds", 0) for it in items)
    dropped = sum(1 for it in items if it.get("dropped_seconds", 0))
    capped = any(
        it.get("look_ahead_capped")
        for it in items
        if float(it["look_ahead_ms"]) == ahead)
    causal = ahead <= LOOK_AHEAD_MAX_MS
    holds = mean >= DRIFT_MEAN_DB and low >= DRIFT_MIN_DB and lost == 0
    reasons = []
    if not causal:
        reasons.append(
            f"the output depends on {'at least ' if capped else ''}{ahead:.0f}"
            f" ms of future input (limit {LOOK_AHEAD_MAX_MS:.0f} ms)")
    if lost:
        reasons.append(
            f"drops speech after a long context: {lost} of {speech} speech"
            f" seconds on {dropped} of {len(items)} items")
    if mean < DRIFT_MEAN_DB or low < DRIFT_MIN_DB:
        reasons.append(
            f"drifts over a long input: {mean:+.1f} dB SI-SDR on average,"
            f" {low:+.1f} dB at worst, on {scorestream.TAIL_SECONDS:.0f} s of"
            f" 45 s items (limits {DRIFT_MEAN_DB:+.0f} / {DRIFT_MIN_DB:+.0f})")
    out = dict(side)
    out.update(
        verdict="yes" if causal and holds else "no",
        causal=causal,
        holds=holds,
        reasons=reasons,
        rule=dict(side.get("rule") or {},
                  board_look_ahead_max_ms=LOOK_AHEAD_MAX_MS,
                  board_drift_mean_db=DRIFT_MEAN_DB,
                  board_drift_min_db=DRIFT_MIN_DB),
    )
    return out


def _stale_streaming(side, d, records):
    """Returns why a streaming sidecar may not be shown, or None.

    Another build, another probe rule, another scorer, another render than the
    base scores read, or whole-render outputs other than the ones the base
    records scored (``records``: the system's per-item record file, when at
    hand).
    """
    if side.get("_set_id") != d.get("_set_id"):
        return (f"probed on set {str(side.get('_set_id'))[:12]}, metrics on "
                f"{str(d.get('_set_id'))[:12]}")
    if side.get("_rule_version") != scorestream.RULE_VERSION:
        return (f"follows probe rule {side.get('_rule_version')!r}, the current"
                f" rule is {scorestream.RULE_VERSION!r}")
    if side.get("_scorer_sha256") != scorestream.SCORER_SHA256:
        return "scored by another implementation of the probe"
    if side.get("_outputs_sha256") != d.get("_outputs_sha256"):
        return "compared with another render than the base scores read"
    outputs = scorefiles.record_outputs(records)
    if outputs:
        stale = [
            it["id"] for it in side.get("items", []) if outputs.get(
                it["id"], it.get("output_sha256")) != it.get("output_sha256")
        ]
        if stale:
            return (f"read other outputs than the base scores for {len(stale)}"
                    f" items ({stale[0]})")
    return None


def load_streaming(results, scores):
    """Returns {system: its streaming sidecar} for current probes.

    The systems whose probe was scored on this build with the current rule and
    scorer, on the outputs the base scores read; anything else is ignored with a
    warning, never shown. The probe is cut from the whole set, so a subset view
    (results scored with ``BENCH_SUBSET``) reads the sidecar of the whole-set
    directory beside it when its own has none.
    """
    out = {}
    for name, d in scores.items():
        dirs = [results]
        if d.get("_subset", "all") != "all":
            dirs.append(os.path.dirname(os.path.abspath(results.rstrip("/"))))
        for where in dirs:
            path = os.path.join(where, f"{name}{STREAM_EXT}")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as fh:
                side = json.load(fh)
            if bool(side.get("_sealed")) != bool(d.get("_sealed")):
                raise ValueError(
                    f"{name}{STREAM_EXT}: private/public scope differs from its"
                    " base metrics")
            why = _stale_streaming(side, d,
                                   os.path.join(where, f"{name}.items.jsonl"))
            if why:
                print(
                    f"[warn] {os.path.basename(path)} {why}: ignored, re-run"
                    " scorestream.py",
                    file=sys.stderr,
                )
                continue
            out[name] = judged(side)
            break
    return out


def streaming_cell(stream, name):
    """yes, no, or – for a system that was not probed."""
    side = (stream or {}).get(name)
    return (side["verdict"]
            if side and side.get("verdict") in ("yes", "no") else "–")


def streaming_summary(side):
    """Returns what the JSON carries of a streaming sidecar.

    The verdict and the measurements behind it.
    """
    keys = (
        "verdict",
        "causal",
        "holds",
        "look_ahead_ms",
        "look_ahead_max_ms",
        "look_ahead_capped",
        "look_ahead_median_capped",
        "look_ahead_max_capped",
        "drift_mean_db",
        "drift_min_db",
        "cold_start_db",
        "dropped_seconds",
        "speech_seconds",
        "dropout_items",
        "path_switch",
        "context_diff_db",
        "n_items",
        "reasons",
        "rule",
    )
    return {k: side.get(k) for k in keys}


def markdown_streaming(stream, order, tiers):
    """Returns the streaming probe of the ranked systems as markdown.

    In board order: the measurements behind the yes or no, then the systems not
    probed.
    """
    ranked = [n for rows in order.values() for n in rows]
    lines = [
        ("| system | future input (ms) | drift over a long input (dB) | cold"
         " start (dB) | speech dropped (s) | streaming probe |"),
        "|---|---|---|---|---|---|",
    ]
    switched = False
    for n in ranked:
        side = stream.get(n)
        if not side:
            continue
        cold = side.get("cold_start_db")
        cold_text = "–" if cold is None or cold != cold else f"{cold:+.1f}"
        mark = " †" if side.get("path_switch") else ""
        switched = switched or bool(side.get("path_switch"))
        lines.append(f"| {tiers[n][1]} | {scorestream.look_ahead_text(side)} |"
                     f" {side['drift_mean_db']:+.1f} (min"
                     f" {side['drift_min_db']:+.1f}){mark} | {cold_text} |"
                     f" {side.get('dropped_seconds', 0)} of"
                     f" {side.get('speech_seconds', 0)} | {side['verdict']} |")
    text = "\n".join(lines)
    notes = []
    if switched:
        notes.append(
            "† the release takes different inference paths for the whole item"
            " and the cut (whole-file inference up to a length, segmented"
            " beyond), so this difference includes the change of path.")
    missing = [tiers[n][1] for n in ranked if n not in stream]
    if missing:
        notes.append("Not probed: " + ", ".join(missing) + ".")
    if notes:
        text += "\n\n" + "\n".join(notes)
    return text
