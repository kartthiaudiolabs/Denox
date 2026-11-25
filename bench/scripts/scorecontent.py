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
"""Content preservation: which words survive, by kind, as a recogniser reads.

    python bench/scripts/scorecontent.py --systems INPUT,voice_isolator_v0[,...]
        [--asr large-v3] [--conds clean,babble@+4,...]
        [--markdown-out staging/content_table.md]

The composite's retention metric reads 100 ms blocks through a percentile: it
cannot see one dropped consonant, syllable or short word (CONTROLS.md). This
sidecar is an ASR-based content diagnostic. Every system's output is transcribed
by one recogniser, the transcript is aligned word by word to the item's verified
transcript, and each reference word is counted as kept, substituted or deleted,
with insertions counted on the hypothesis side. Reference words are classed as
numbers, short words (three letters or fewer), consonant-final words and the
rest, because numbers, short function words and word-final consonants are what a
front end loses first.

The panel is every English item of the named conditions with a verified
transcript (the corpus text confirmed against the audio by a language model, or
the model's own transcript the maintainer accepted by ear), so the reference is
never a recogniser's unchecked guess. Rates are pooled per class over the panel,
as errors per hundred reference words of that class, with a bootstrap interval
that resamples the recordings and beds behind the items rather than the items
(the panel repeats a recording at several levels, so items are not independent
draws), and are read next to the unprocessed input's: the change column is the
difference of two pooled rates under the same resampling. A front end is judged
by what it changes, so the same items must have been scored for every system
compared: a system that misses items is marked incomplete and never compared on
the intersection.

``--controls`` renders the metric controls of controls.py (the reference
untouched, with 20 to 320 ms of speech removed, re-synthesised, with its top
octave replaced by noise) from the panel items' own references as systems named
``ctl_<control>`` and scores them too: the table then says what the check itself
can and cannot see, next to the systems.

Writes ``<system>.content.json`` beside the other sidecars, per-item counts
included so systems can be paired, and with ``--markdown-out`` one table over
the systems given.
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf

import anchors
import common
import compare
import controls
import observations
import scoreasr
import scorebench

SCORER_SHA256 = observations.implementation(__file__, scoreasr.__file__,
                                            common.__file__,
                                            observations.__file__,
                                            scorebench.__file__)

VERIFIED = ("corpus-verified", "gemini-accepted", "human-verified")
CLASSES = ("number", "short", "consonant-final", "other")
CONSONANTS = frozenset("bcdfghjklmnpqrstvwxz")
N_BOOT = 1000


def word_class(token: str) -> str:
    """The class of a normalised reference token."""
    if any(ch.isdigit() for ch in token):
        return "number"
    if len(token) <= 3:
        return "short"
    if token[-1] in CONSONANTS:
        return "consonant-final"
    return "other"


def align(hyp: list, ref: list) -> tuple:
    """Levenshtein alignment of ``hyp`` to ``ref``.

    Returns:
        A list with one entry per reference token, ``kept``, ``sub`` or
        ``del``, and the number of hypothesis tokens inserted. Ties go to a
        match or substitution first, then a deletion.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    ops, ins, i, j = [None] * n, 0, n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and
                d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])):
            ops[i - 1] = "kept" if ref[i - 1] == hyp[j - 1] else "sub"
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops[i - 1] = "del"
            i -= 1
        else:
            ins += 1
            j -= 1
    return ops, ins


def panel(rows: list, golden: dict, conds: tuple, language: str = "en") -> list:
    """Returns the items scored.

    Paired, in ``conds``, in ``language``, with a verified transcript.
    """
    return [
        r for r in rows if r["condition"] in conds and
        r.get("language") == language and r.get("clean_path") and
        (golden.get(r["id"]) or {}).get("tier") in VERIFIED
    ]


def count(ops: list, ref: list) -> dict:
    """Per-class counts of reference tokens kept, substituted and deleted."""
    out = {c: {"n": 0, "kept": 0, "sub": 0, "del": 0} for c in CLASSES}
    for op, token in zip(ops, ref):
        cell = out[word_class(token)]
        cell["n"] += 1
        cell[op] += 1
    return out


def groups_of(records: dict) -> list:
    """Returns the two cluster vectors of the items, in id order.

    One for the foreground recording, one for the bed.
    """
    ids = sorted(records)
    return [
        [records[i].get("speech_group") for i in ids],
        [records[i].get("bed_group") for i in ids],
    ]


def _interval(numer: list,
              denom: list,
              groups: list,
              n_boot: int = N_BOOT,
              seed: int = 0) -> list:
    """Returns the 95 % interval of a pooled rate under clustered resampling.

    The rate is 100 * sum(numer) / sum(denom), resampled by source
    (compare.weights); [nan, nan] when the items span too few sources for one.
    """
    k, n = np.array(numer, dtype=float), np.array(denom, dtype=float)
    if len(k) < 2 or n.sum() == 0:
        return [float("nan"), float("nan")]
    try:
        d = compare.interval(
            lambda w: 100.0 * (w @ k) / (w @ n) if (w @ n) else float("nan"),
            groups,
            n_boot,
            seed,
        )
    except ValueError:
        return [float("nan"), float("nan")]
    return d["ci95"]


def pooled(records: dict) -> dict:
    """Returns per-class and overall error rates per hundred reference words.

    The deletion and substitution rates carry source-clustered bootstrap
    intervals.
    """
    ids = sorted(records)
    items = [records[i] for i in ids]
    groups = groups_of(records)
    out = {}
    for cls in CLASSES + ("all",):
        take = ((lambda r, cls=cls: r["classes"][cls]) if cls != "all" else
                (lambda r: {
                    k: sum(r["classes"][c][k] for c in CLASSES)
                    for k in ("n", "kept", "sub", "del")
                }))
        n = sum(take(r)["n"] for r in items)
        cell = {"n": n}
        for key in ("del", "sub"):
            total = sum(take(r)[key] for r in items)
            cell[f"{key}_pct"] = 100.0 * total / n if n else float("nan")
            cell[f"{key}_ci95"] = _interval(
                [take(r)[key] for r in items],
                [take(r)["n"] for r in items],
                groups,
            )
        out[cls] = cell
    n_all = out["all"]["n"]
    out["all"]["n_items"] = len(items)
    out["all"]["n_recordings"] = len({g for g in groups[0] if g is not None})
    out["all"]["n_beds"] = len({g for g in groups[1] if g is not None})
    ins = sum(r["ins"] for r in items)
    out["all"]["ins_pct"] = 100.0 * ins / n_all if n_all else float("nan")
    out["all"]["ins_ci95"] = _interval(
        [r["ins"] for r in items],
        [sum(r["classes"][c]["n"] for c in CLASSES) for r in items],
        groups,
    )
    return out


def label_groups(records: dict, rows: list) -> None:
    """Returns opaque group labels per item, numbered within this panel.

    ``s<k>`` for the foreground recording, ``b<k>`` for the bed: the clustered
    bootstrap needs equal labels for equal sources and nothing else, and a
    number cannot be run back to a corpus the way a digest could.
    """
    by_id = {r["id"]: r for r in rows}
    seen = {"speech_source": {}, "bed_source": {}}
    for uid in sorted(records):
        rec = observations.item(by_id[uid])
        for key, prefix, field in (
            ("speech_source", "s", "speech_group"),
            ("bed_source", "b", "bed_group"),
        ):
            value = rec.get(key)
            records[uid][field] = (None
                                   if value is None else seen[key].setdefault(
                                       value, f"{prefix}{len(seen[key])}"))


def score_system(name: str, rows: list, asr, cache, norm, golden: dict) -> dict:
    """Returns one system's per-item content counts over the panel.

    The pooled rates come with them.
    """
    outd = os.path.join(common.OUT_ROOT, name)
    records, missing = {}, []
    for r in rows:
        path = r["mix_path"] if name == "INPUT" else os.path.join(outd, r["id"])
        if not os.path.exists(path):
            missing.append(r["id"])
            continue
        lang = scoreasr.spoken_language(r, golden)
        ref = norm.tokens(golden[r["id"]]["text"], lang)
        hyp = norm.tokens(cache.get(asr, path, lang), lang)
        ops, ins = align(hyp, ref)
        records[r["id"]] = {
            "condition": r["condition"],
            "classes": count(ops, ref),
            "ins": ins,
        }
    label_groups(records, rows)
    return {
        "items": records,
        "missing": missing,
        "rates": pooled(records) if records else {},
        "asr_identity": getattr(asr, "identity", None),
    }


def asr_identity(doc: dict):
    """The recogniser identity of a scored result or a loaded sidecar."""
    return doc.get("asr_identity") or doc.get("_asr_identity")


def paired_delta(left: dict,
                 right: dict,
                 key: str = "del",
                 cls: str = "all") -> dict:
    """Returns two systems' difference in pooled error rate, with its interval.

    Left minus right, in points, over the panel, with a source-clustered
    bootstrap interval: the same quantity the table's rate columns show. Only on
    identical complete cohorts read by the same recogniser against the same
    reference tokens: a system that misses items is reported as incomplete,
    never compared on the intersection, and a mismatch of recogniser or
    reference is reported as incompatible.
    """
    nan = float("nan")
    ids = sorted(left["items"])
    if (set(left["items"]) != set(right["items"]) or left.get("missing") or
            right.get("missing")):
        differ = set(left.get("missing", [])) | set(right.get("missing", []))
        differ |= set(left["items"]) ^ set(right["items"])
        return {
            "delta": nan,
            "ci95": [nan, nan],
            "n_items": len(set(left["items"]) & set(right["items"])),
            "incomplete": True,
            "missing": sorted(differ),
        }

    # the same recogniser read both sides, and both sides were aligned to the
    # same reference tokens: otherwise a difference of pooled rates compares
    # nothing
    li, ri = asr_identity(left), asr_identity(right)
    if li is not None and ri is not None and li != ri:
        return {
            "delta": nan,
            "ci95": [nan, nan],
            "n_items": len(ids),
            "incomplete": False,
            "incompatible": "recognisers differ",
        }
    for i in ids:
        if any(left["items"][i]["classes"][c]["n"] != right["items"][i]
               ["classes"][c]["n"] for c in CLASSES):
            return {
                "delta": nan,
                "ci95": [nan, nan],
                "n_items": len(ids),
                "incomplete": False,
                "incompatible": "reference tokens differ",
            }

    def counts(rec):
        cells = ([rec["classes"][cls]]
                 if cls != "all" else list(rec["classes"].values()))
        return sum(c[key] for c in cells), sum(c["n"] for c in cells)

    kl = np.array([counts(left["items"][i])[0] for i in ids], dtype=float)
    kr = np.array([counts(right["items"][i])[0] for i in ids], dtype=float)
    n = np.array([counts(left["items"][i])[1] for i in ids], dtype=float)
    if len(ids) < 2 or n.sum() == 0:
        return {
            "delta": nan,
            "ci95": [nan, nan],
            "n_items": len(ids),
            "incomplete": False,
        }
    delta = 100.0 * (kl.sum() - kr.sum()) / n.sum()
    try:
        d = compare.interval(
            lambda w: 100.0 * (w @ kl - w @ kr) / (w @ n) if (w @ n) else nan,
            groups_of(left["items"]),
            N_BOOT,
            0,
        )
        ci = d["ci95"]
    except ValueError:
        ci = [nan, nan]
    return {
        "delta": float(delta),
        "ci95": ci,
        "n_items": len(ids),
        "incomplete": False,
    }


def write_sidecar(
    path: str,
    name: str,
    tag: str,
    result: dict,
    identity: dict,
    rows: list,
    conds: tuple,
) -> None:
    doc = {
        "_system": name,
        "_set_id": common.set_id(),
        "_subset": common.SUBSET,
        "_asr": tag,
        "_asr_identity": identity,
        "_panel": {
            "language": "en",
            "tiers": list(VERIFIED),
            "conditions": list(conds),
            "n_items": len(rows),
        },
        "_scorer_sha256": SCORER_SHA256,
        "_selection": {
            "conds": ",".join(conds),
            "limit": 0
        },
        "_outputs": scoreasr.output_hashes(name, rows),
        "rates": result["rates"],
        "missing": result["missing"],
        "items": result["items"],
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


# a metric control rendered onto the panel, read against ctl_reference
CONTROL_PREFIX = "ctl_"
# beside each control's folder: item -> digest of (ref, mix, code)
LEDGER = ".provenance.json"
CONTROLS = (
    "reference",
    "delete20ms",
    "delete40ms",
    "delete80ms",
    "delete160ms",
    "delete320ms",
    "resynth",
    "hfinvented",
)


def render_controls(rows: list, names: tuple = CONTROLS) -> list:
    """Renders the named controls into ``BENCH_OUT`` as systems ``ctl_<name>``.

    Each is built by controls.py from the panel's references and stamped with
    the set id and the controls' implementation digest; an existing file is
    reused only while its reference, mixture and the control code are unchanged.
    Returns the system names.
    """
    controls_sha = observations.implementation(controls.__file__)
    hashes = common.file_hashes()
    systems = []
    for name in names:
        system = CONTROL_PREFIX + name
        outd = os.path.join(common.OUT_ROOT, system)
        os.makedirs(outd, exist_ok=True)
        # a control's output is a function of the reference, the mixture and the
        # control code; a file already there is reused only while the ledger
        # says all three are what they were
        ledger_path = os.path.join(common.OUT_ROOT, system + LEDGER)
        ledger = {}
        if os.path.exists(ledger_path):
            with open(ledger_path, encoding="utf-8") as fh:
                ledger = json.load(fh)
        for r in rows:
            out = os.path.join(outd, r["id"])
            stamp = anchors.digest([
                hashes(r["clean_path"]),
                hashes(r["mix_path"]),
                name,
                controls_sha,
            ])
            if os.path.exists(out) and ledger.get(r["id"]) == stamp:
                continue
            ref = common.load48(r["clean_path"])
            mix = common.load48(r["mix_path"])
            n = min(len(ref), len(mix))
            y = controls.CONTROLS[name][1](ref[:n], mix[:n], r["id"])[:n]
            sf.write(out,
                     y.numpy().astype("float32"),
                     common.SR,
                     subtype="FLOAT")
            ledger[r["id"]] = stamp
        with open(ledger_path, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=0)
        with open(os.path.join(common.RESULTS, f"{system}.meta.json"),
                  "w",
                  encoding="utf-8") as fh:
            json.dump(
                {
                    "name":
                        system,
                    "display":
                        f"control: {controls.CONTROLS[name][0]}",
                    "set_id":
                        common.set_id(),
                    "n_files":
                        len(rows),
                    "controls_sha256":
                        controls_sha,
                    "notes": ("a metric control rendered onto the content panel"
                              " (scorecontent.py --controls)"),
                },
                fh,
                indent=1,
            )
        systems.append(system)
    return systems


def _table(docs: dict, order: list, baseline: str, against: str, label) -> list:

    def fmt(v):
        return "-" if v != v else f"{v:.1f}"

    # the classes' sizes travel with their columns: a class of a few dozen
    # tokens is read as such
    sizes = docs[order[0]]["rates"] if order else {}

    def size(c):
        return f" ({sizes[c]['n']})" if c in sizes and sizes[c].get("n") else ""

    lines = [
        (f"| system | items | deleted: all{size('all')} |"
         f" numbers{size('number')} | short words{size('short')} |"
         f" consonant-final{size('consonant-final')} | substituted |"
         f" inserted | deletions vs {against} |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    expected = max(
        len(docs[n]["items"]) + len(docs[n].get("missing", [])) for n in order)
    for n in order:
        r = docs[n]["rates"]
        cells = [fmt(r["all"]["del_pct"])
                ] + [fmt(r[c]["del_pct"]) for c in CLASSES[:3]]
        cells += [fmt(r["all"]["sub_pct"]), fmt(r["all"]["ins_pct"])]
        items = str(len(docs[n]["items"]))
        if docs[n].get("missing") or len(docs[n]["items"]) < expected:
            items = f"{len(docs[n]['items'])} of {expected} (incomplete)"
        delta = "-"
        if n != baseline and baseline in docs:
            d = paired_delta(docs[n], docs[baseline])
            if d.get("incomplete"):
                delta = f"incomplete ({len(d['missing'])} items differ)"
            elif d.get("incompatible"):
                delta = f"incompatible ({d['incompatible']})"
            elif d["delta"] == d["delta"]:
                delta = (f"{d['delta']:+.1f} [{d['ci95'][0]:+.1f},"
                         f" {d['ci95'][1]:+.1f}]")
        lines.append(f"| {label(n)} | {items} | " + " | ".join(cells) +
                     f" | {delta} |")
    return lines


def markdown(docs: dict, label=None) -> str:
    """Returns the content tables over the systems given.

    Deletions by class, substitutions, insertions, and the paired change in
    deletions against the unprocessed input. Controls (``ctl_*``) get their own
    table, read against the untouched reference they were made from.
    """
    label = label or (lambda n: n)
    systems = [n for n in docs if not n.startswith(CONTROL_PREFIX)]
    control_names = [n for n in docs if n.startswith(CONTROL_PREFIX)]
    lines = [
        ("Words lost per hundred reference words of each kind, from one"
         " recogniser's reading of every output against the item's verified"
         " transcript; substitutions and insertions per hundred reference"
         " words overall; the last column is the difference of the pooled"
         " deletion rates against the unprocessed input (points, 95 %"
         " interval resampling the recordings and beds behind the items, not"
         " the items). Lower is better. The number in each class header is"
         " how many reference words the class holds; a system that misses"
         " items is marked incomplete and not compared."),
        "",
    ]
    order = (["INPUT"] if "INPUT" in systems else []) + sorted(
        n for n in systems if n != "INPUT")
    rates = docs[order[0]]["rates"].get("all", {}) if order else {}
    if rates.get("n_recordings"):
        lines[0] += (
            f" The intervals rest on the {rates['n_recordings']} recordings and"
            f" {rates['n_beds']} beds behind the {rates['n_items']} items.")
    lines += _table(docs, order, "INPUT", "input", label)
    if control_names:
        ref = CONTROL_PREFIX + "reference"
        order = ([ref] if ref in control_names else []) + sorted(
            n for n in control_names if n != ref)
        lines += [
            "",
            ("What the check itself can see: the metric controls rendered"
             " onto the same panel, read against the untouched reference"
             " they were made from."),
            "",
        ]
        lines += _table(docs, order, ref, "reference", label)
    return "\n".join(lines)


def missing_render(name: str) -> str:
    """Why a system has nothing to score here, or an empty string.

    The identity check (``scorebench.render_problem``) reads only
    ``<name>.meta.json``, which a results directory seeded from another run
    also holds; the outputs themselves live under ``BENCH_OUT``. Without them
    the scorer would write an empty sidecar over the seeded one.
    """
    if name == "INPUT" or os.path.isdir(os.path.join(common.OUT_ROOT, name)):
        return ""
    return f"no outputs under {common.OUT_ROOT}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems",
                    required=True,
                    help="comma-separated system names")
    ap.add_argument(
        "--asr",
        default="large-v3",
        help="one recogniser, as scoreasr.py names it",
    )
    ap.add_argument(
        "--conds",
        default=",".join(
            c for c in scoreasr.DEFAULT_CONDS if not c.startswith("wild")),
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--markdown-out", default="")
    ap.add_argument(
        "--controls",
        action="store_true",
        help="also render and score the metric controls on the panel",
    )
    ap.add_argument(
        "--table-only",
        action="store_true",
        help="render the table from existing sidecars; score nothing",
    )
    a = ap.parse_args()
    if a.table_only:
        docs = {}
        for name in [s for s in a.systems.split(",") if s]:
            path = os.path.join(common.RESULTS, f"{name}.content.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                problem = scoreasr.stale_sidecar(doc, common.RESULTS) or (
                    "was scored by another implementation"
                    if doc.get("_scorer_sha256") != SCORER_SHA256 else None)
                if problem:
                    raise SystemExit(
                        f"{name}.content.json {problem}: re-run scorecontent.py"
                    )
                docs[name] = doc
        md = markdown(docs)
        if a.markdown_out:
            with open(a.markdown_out, "w", encoding="utf-8") as fh:
                fh.write(md + "\n")
        print(md)
        return
    rows_all = common.manifest()
    common.guard_private_results(rows_all)
    if common.is_sealed(rows_all):
        raise SystemExit("sealed audio cannot be sent to an ASR service")
    conds = tuple(c for c in a.conds.split(",") if c)
    golden = scoreasr.load_golden()
    rows = panel(rows_all, golden, conds)
    if not rows:
        raise SystemExit(
            "no verified English transcripts in the named conditions")
    print(
        f"content panel: {len(rows)} items over {len(conds)} conditions",
        flush=True,
    )
    norm = scoreasr.Normalizer()
    asr = scoreasr.make_asr(a.asr, a.device, a.beam_size)
    cache = scoreasr.Cache(scoreasr.CACHE_ROOT, scoreasr.cache_tag(asr),
                           asr.identity)
    os.makedirs(common.RESULTS, exist_ok=True)
    names = [s for s in a.systems.split(",") if s]
    if a.controls:
        names += render_controls(rows)
    docs = {}
    for name in names:
        problem = (name != "INPUT" and
                   scorebench.render_problem(name)) or missing_render(name)
        if problem:
            print(f"skip {name}: {problem}", flush=True)
            continue
        result = score_system(name, rows, asr, cache, norm, golden)
        if not result["items"]:
            print(
                f"skip {name}: none of the panel's {len(rows)} outputs is"
                " there; an earlier sidecar is left as it was",
                flush=True,
            )
            continue
        write_sidecar(
            os.path.join(common.RESULTS, f"{name}.content.json"),
            name,
            a.asr,
            result,
            asr.identity,
            rows,
            conds,
        )
        docs[name] = result
        r = result["rates"]["all"]
        print(
            f"{name}: deleted {r['del_pct']:.1f}  substituted"
            f" {r['sub_pct']:.1f}  inserted {r['ins_pct']:.1f} per 100 words;"
            f" {len(result['missing'])} missing",
            flush=True,
        )
    if a.markdown_out and docs:
        md = markdown(docs)
        with open(a.markdown_out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(md)


if __name__ == "__main__":
    main()
