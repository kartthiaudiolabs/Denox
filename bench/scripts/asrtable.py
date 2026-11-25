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
"""Downstream-ASR table: error rate per condition with and without processing.

    asrtable.py [--results DIR] [--asr large-v3]
                [--systems INPUT,voice_isolator_v0,...]
                [--ours voice_isolator_v0] [--markdown-out FILE]

Reads ``<system>.asr.json`` (scoreasr.py). One row per condition: the
unprocessed input first, then every system, each as the pooled error rate with
its 95 % interval, the best value in bold; a last column gives the --ours
system's change against the input. Rows come from the conditions the input was
scored on; a system missing a condition shows a dash.
"""

import argparse
import glob
import json
import os
import sys

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roster
import scoreasr

INPUT = "INPUT"
# scoreasr.FALLBACK: no golden text, so the clean reference's transcript
FALLBACK_TIER = "clean-asr"
# two contracts a recogniser meets under competing speech, never pooled
# together: chatter is to be removed, a conversation's second voice is to be
# kept
SLICES = (
    ("background chatter removed", lambda c: c.startswith("babble@")),
    ("a conversation kept, both voices", lambda c: c == "conversation@0"),
)


def load(results, tag):
    """Returns {system: {condition: block}} for one recogniser.

    The blocks come from every sidecar in ``results``.

    Refuses sidecars of different set builds or scorers, scored on a partial
    selection, or on other audio than their system's base records
    (``scoreasr.stale_sidecar``).
    """
    out, set_ids, scorers = {}, set(), set()
    for f in sorted(glob.glob(os.path.join(results, "*.asr.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        blk = (d.get("asr") or {}).get(tag)
        if not blk:
            continue
        problem = scoreasr.stale_sidecar(d, results)
        if problem:
            raise SystemExit(
                f"{os.path.basename(f)} {problem}: re-run scoreasr.py")
        out[d["_system"]] = blk["conditions"]
        set_ids.add(d.get("_set_id"))
        scorers.add(d.get("_scorer_sha256"))
    if len(set_ids) > 1:
        raise SystemExit(f"asr sidecars from different set builds: {set_ids}")
    if len(scorers) > 1:
        raise SystemExit(
            "asr sidecars scored by different implementations: re-run"
            " scoreasr.py")
    return out


def complete(block, reference):
    """Returns whether a condition's block covers the input's items.

    It must cover every item the unprocessed input's block covers, with none
    missing: a partial row would read as a result on easier items.
    """
    return (block is not None and not block.get("n_missing") and
            (reference is None or set(block.get("items") or {}) == set(
                reference.get("items") or {})))


def cell(block):
    if block is None:
        return "-"
    lo, hi = block["ci95"]
    half = (hi - lo) / 2 if hi == hi else float("nan")
    return (f"{block['wer_pct']:.1f} ± {half:.1f}"
            if half == half else f"{block['wer_pct']:.1f}")


def relative_change(ours, inp):
    """Returns the --ours system's change in error rate, in % of the input's.

    Negative means fewer errors.
    """
    if ours is None or inp is None or not inp["wer_pct"]:
        return float("nan")
    return 100.0 * (ours["wer_pct"] - inp["wer_pct"]) / inp["wer_pct"]


def reference_phrase(conditions):
    """Returns what the transcripts were scored against.

    It is derived from the reference tiers the sidecar recorded.

    Args:
        conditions: {condition: block} of one system, each block with
            ``reference_tiers``.

    Returns:
        The golden transcripts with the count of items that fell back to the
        recogniser's transcript of the clean reference, or that fallback alone
        when no item had a golden text.
    """
    tiers = {}
    for block in conditions.values():
        for tier, n in (block.get("reference_tiers") or {}).items():
            tiers[tier] = tiers.get(tier, 0) + n
    total, fallback = sum(tiers.values()), tiers.get(FALLBACK_TIER, 0)
    if total == fallback:
        return "the same recogniser's transcript of the clean reference"
    if not fallback:
        return "the set's golden transcripts"
    return (
        f"the set's golden transcripts ({total - fallback} of {total} items;"
        f" the other {fallback} against the same recogniser's transcript of the"
        " clean reference)")


def table(sidecars, systems, ours, tag, label=None):
    label = label or roster.display_name
    cols = [INPUT] + [s for s in systems if s != INPUT]
    conds = list(sidecars.get(INPUT, {}).keys()) or sorted(
        {c for s in sidecars.values() for c in s})
    names = ["unprocessed input" if c == INPUT else label(c) for c in cols]
    head = f"| condition | {' | '.join(names)} | {label(ours)} vs input |"
    sep = "|---|" + "---:|" * (len(cols) + 1)
    lines = [
        (f"Error rate (%) of `{tag}` transcripts against"
         f" {reference_phrase(sidecars.get(INPUT) or {})}; words, characters"
         " for Chinese and Japanese; pooled per condition, ± the 95 % interval"
         " half-width. Lower is better; best in bold."),
        "",
        head,
        sep,
    ]
    for cond in conds:
        blocks = {c: sidecars.get(c, {}).get(cond) for c in cols}
        whole = {
            c: b for c, b in blocks.items() if complete(b, blocks.get(INPUT))
        }
        finite = [b["wer_pct"] for b in whole.values()]
        best = min(finite) if finite else None
        cells = []
        for c in cols:
            b = blocks[c]
            txt = cell(b) if b is None or c in whole else "incomplete"
            if c in whole and best is not None and b["wer_pct"] == best:
                txt = f"**{txt}**"
            cells.append(txt)
        rel = relative_change(blocks.get(ours), blocks.get(INPUT))
        rel_txt = "-" if rel != rel else f"{rel:+.0f} %"
        lines.append(f"| `{cond}` | " + " | ".join(cells) + f" | {rel_txt} |")
    return "\n".join(lines)


# a pooled rate over fewer items is not shown: it is one or two recordings
MIN_SLICE_ITEMS = 20


def slices(by_tag,
           rows,
           golden,
           systems,
           ours,
           label=None,
           min_items=MIN_SLICE_ITEMS):
    """Returns pooled error rates over each competing-speech slice.

    One row per recogniser and unit (words, or characters for the languages
    without word boundaries), with the --ours system's paired change against the
    input under source-clustered resampling (compare.interval): the slice spans
    a ladder, so the same recording enters at several levels and a per-item
    bootstrap would overstate the precision.
    """
    # pylint: disable-next=import-outside-toplevel
    import compare
    # pylint: disable-next=import-outside-toplevel
    import observations

    label = label or roster.display_name
    by_id = {r["id"]: r for r in rows}
    cols = [INPUT] + [x for x in systems if x != INPUT]
    lines = [
        ("Error rate (%) pooled over the slice's items (total edits over"
         " total reference tokens), one row per recogniser and unit; the"
         f" last column is {label(ours)}'s paired change against the"
         " unprocessed input in points, with a 95 % interval that resamples"
         " the recordings and beds behind the items rather than the items"
         f" themselves. Lower is better. A unit with fewer than {min_items}"
         " items in a slice is not shown."),
        "",
        "| slice | recogniser | unit | items | " + " | ".join(
            "unprocessed input" if c == INPUT else label(c) for c in cols) +
        f" | {label(ours)} vs input |",
        "|---|---|---|---:|" + "---:|" * (len(cols) + 1),
    ]
    for name, member in SLICES:
        for tag in sorted(by_tag):
            sidecars = by_tag[tag]
            conds = [c for c in sidecars.get(INPUT, {}) if member(c)]
            for unit in ("word", "char"):
                ids = [
                    i for c in conds
                    for i in sidecars[INPUT][c].get("items", {})
                    if i in by_id and scoreasr.unit_for(
                        scoreasr.spoken_language(by_id[i], golden)) == unit
                ]
                if len(ids) < min_items:
                    continue
                cells, rates = [], {}
                for c in cols:
                    blocks = sidecars.get(c, {})
                    have = [(i, blocks[k]["items"][i]) for k in conds
                            if k in blocks for i in blocks[k].get("items", {})
                            if i in ids]
                    if len(have) < len(ids):
                        cells.append("-")
                        continue
                    e = sum(v[0] for _, v in have)
                    n = sum(v[1] for _, v in have)
                    rates[c] = (100.0 * e / n) if n else float("nan")
                    cells.append(f"{rates[c]:.1f}")
                best = min((v for v in rates.values() if v == v), default=None)
                cells = [(f"**{t}**"
                          if best is not None and rates.get(c) == best else t)
                         for c, t in zip(cols, cells)]
                delta = "-"
                if ours in rates and INPUT in rates:
                    pairs = {}
                    for k in conds:
                        for i, v in sidecars[ours][k]["items"].items():
                            if i in ids and i in sidecars[INPUT][k].get(
                                    "items", {}):
                                pairs[i] = (v, sidecars[INPUT][k]["items"][i])
                    order = sorted(pairs)
                    e_ours = [pairs[i][0][0] for i in order]
                    e_in = [pairs[i][1][0] for i in order]
                    n_ref = [pairs[i][0][1] for i in order]
                    recs = [observations.item(by_id[i]) for i in order]
                    groups = [[r.get(g) for r in recs] for g in compare.GROUPS]
                    # pylint: disable-next=import-outside-toplevel
                    import numpy as np

                    eo, ei, nr = (
                        np.array(x, dtype=float) for x in (e_ours, e_in, n_ref))

                    def reduce(w, eo=eo, ei=ei, nr=nr):
                        d = w @ nr
                        return (100.0 * (w @ eo - w @ ei) /
                                d if d else float("nan"))

                    try:
                        d = compare.interval(reduce, groups)
                        delta = (f"{d['delta']:+.1f} [{d['ci95'][0]:+.1f},"
                                 f" {d['ci95'][1]:+.1f}]")
                    except ValueError as e:
                        delta = f"no interval ({e})"
                lines.append(f"| {name} | `{tag}` | {unit}s | {len(ids)} | " +
                             " | ".join(cells) + f" | {delta} |")
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=os.environ.get("BENCH_RESULTS",
                               os.path.join(os.path.dirname(here), "results")),
    )
    ap.add_argument("--asr", default="large-v3")
    ap.add_argument(
        "--systems",
        default="",
        help="comma-separated; default: every sidecar found",
    )
    ap.add_argument("--ours", default="voice_isolator_v0")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument(
        "--slices-out",
        default="",
        help=(
            "also write the competing-speech slices over every recogniser found"
        ),
    )
    a = ap.parse_args()
    sidecars = load(a.results, a.asr)
    if not sidecars:
        raise SystemExit(f"no asr sidecars for {a.asr!r} in {a.results}")
    systems = [s for s in a.systems.split(",") if s
              ] or [s for s in sidecars if s != INPUT]
    md = table(sidecars, systems, a.ours, a.asr)
    print(md)
    if a.markdown_out:
        with open(a.markdown_out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {a.markdown_out}")
    if a.slices_out:
        # pylint: disable-next=import-outside-toplevel
        import common

        tags = set()
        for f in glob.glob(os.path.join(a.results, "*.asr.json")):
            with open(f, encoding="utf-8") as fh:
                tags |= set(json.load(fh).get("asr") or {})
        by_tag = {t: load(a.results, t) for t in sorted(tags)}
        md = slices(by_tag, common.manifest(), scoreasr.load_golden(), systems,
                    a.ours)
        with open(a.slices_out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(md)
        print(f"wrote {a.slices_out}")


if __name__ == "__main__":
    main()
