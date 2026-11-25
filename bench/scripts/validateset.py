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
"""Checks a built eval set for properties the scorer and leaderboard rely on.

    python bench/scripts/validateset.py [--set ROOT] [--full]
        [--json report.json]

Every check prints PASS, WARN or FAIL with counts; the exit status is 1 when
anything FAILs. WARN marks properties that do not break scoring but limit what
the set can support (a condition below the leaderboard's item minimum, an uneven
language spread). --full re-hashes every file against checksums.json instead of
a sample.

Checked: manifest schema and unique ids; every referenced file exists, is 48 kHz
mono, finite, unclipped and paired to a reference of the same length;
input waveforms are unique across the entire inventory, including quick runs;
checksums.json covers every file and its set id reproduces; per-condition counts
against the leaderboard minimum; language balance of the paired items; the
scored region left by score_from; the realised SNR of additive conditions
against the nominal one (mix minus reference is the corruptor, so the shared
gain must have preserved the ratio); the measured bandwidth gate of the speech
(items with ``score_policy: retention`` are exempt: their reference is a
recording made in the wild); band-limited twins of the rate sweep.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import soundfile as sf

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import ranking
import scoreevents

FIELDS = (
    "id",
    "condition",
    "family",
    "language",
    "speaker",
    "mix_path",
    "clean_path",
    "snr",
    "score_from",
    "effective_bandwidth_hz",
)
MIN_SCORED_SECONDS = 2.0
# active-speech RMS; speech is set to -26 dBFS, the shared peak gain may lower
# it
REF_FLOOR_DBFS = -36.0
PUBLIC_FIELDS = (
    "id",
    "condition",
    "family",
    "language",
    "mix_path",
    "clean_path",
    "snr",
    "score_from",
)  # the item semantics the set id binds
REF_MIN_MARGIN_DB = 20.0
# the condition inventory a complete build promises (--expect-inventory):
# name -> minimum items
EXPECTED_INVENTORY = {
    **{
        c: 72 for c in (
            "babble@+10",
            "babble@+4",
            "babble@+0",
            "babble@-5",
            "ambience@-5",
            "ambience@+0",
            "ambience@+5",
            "events@+5",
            "music@-6",
            "music_vocal@-6",
            "music_vocal@+6",
            "wind@+0",
            "wind@+5",
            "cheer@+5",
            "cheer_tail",
            "expressive_tail@-5",
            "reverb_real",
            "reverb_synth",
            "clean",
            "srsweep_8k",
            "srsweep_16k",
            "srsweep_24k",
            "srsweep_44.1k",
            "srsweep_48k",
            "quiet@-30dBFS",
        )
    },
    **{
        c: 48 for c in (
            "singer@-2",
            "singer@-5",
            "conversation@0",
            "stationary@-5",
            "stationary@+0",
            "stationary@+5",
            "household@+0",
            "household@+5",
            "rain@+0",
            "animals@+0",
            "animals@+5",
            "water@+0",
            "traffic@-5",
            "traffic@+0",
            "traffic@+5",
            "appliance@+0",
            "appliance@+5",
            "tools@+5",
            "nonverbal@+0",
            "nonverbal_clean",
        )
    },
    "long@+4": 16,
    **{
        c: 168 for c in ("wild@-10", "wild@-5", "wild@+0", "wild@+5")
    },
    "babble@-10": 72,
    "ambience@-10": 72,
    "events@-5": 72,
    "stationary@-10": 48,
    "traffic@-10": 48,
    "soft@+5": 48,
    "soft@+0": 48,
    "soft@-5": 48,
    "whisper@+5": 48,
    "whisper@+0": 48,
    "real_noisy": 48,
    "real_dirty": 20,
    "real_wild": 40,
    "real_multi": 48,
    # mild rungs, transitions, noise alone, one background talker, everyday
    # speech with pauses, everyday real recordings
    "babble@+20": 72,
    "ambience@+20": 72,
    "nonverbal@+20": 48,
    "transition@+10": 48,
    "noise_only": 96,
    "talker@+5": 48,
    "everyday@+10": 54,
    "everyday@+20": 54,
    "real_everyday": 200,
    # the same-room block: speech and noise through one room and microphone
    "coroom@pause": 127,
    "coroom@speech": 121,
    "coroom@floor": 127,
    "coroom@both": 127,
    # speech level over the reference's own floor (its quietest in-utterance
    # frames)
}
SNR_TOLERANCE_DB = 1.0
# the same-room block's labels, which its sidecar scores by: the event spans,
# the pauses, the microphone chain, the split, and the room's share of the
# voice's level on its loudest frames (a number)
BLOCK_LABELS = ("events", "pauses", "chain", "split", "room_share_db",
                "room_share_rule")
BANDWIDTH_FLOOR_HZ = 20000
# the bed of a transition item is present for part of the item only: its
# nominal SNR is the bed's while it is on
SKIP_SNR_FAMILIES = ("reverb", "clean", "srsweep", "policy", "real", "long",
                     "transition", "coroom")


class Report:
    """Collects check outcomes; ok is False once anything failed."""

    def __init__(self):
        self.rows = []
        self.ok = True
        self.failing = {}

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))
        if status == "FAIL":
            self.ok = False
        print(f"{status:4s} {name:34s} {detail}", flush=True)

    def check(self, name, failures, total, warn_only=False, detail=""):
        status = "PASS" if not failures else ("WARN" if warn_only else "FAIL")
        self.add(
            name,
            status,
            f"{len(failures)}/{total} flagged" +
            (f"; {detail}" if detail else "") +
            (f"; e.g. {failures[:3]}" if failures else ""),
        )
        # the complete list goes to the JSON report
        self.failing[name] = [str(f) for f in failures]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audio_sha256(path: str) -> str:
    """Hashes decoded samples and their layout, independent of WAV encoding."""
    with sf.SoundFile(path) as audio:
        layout = (audio.samplerate, audio.channels, len(audio))
        digest = hashlib.sha256(json.dumps(layout).encode())
        for block in audio.blocks(blocksize=65536,
                                  dtype="float64",
                                  always_2d=True):
            block[block == 0] = 0.0  # signed zero has the same audio value
            digest.update(block.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _read(root, rel):
    x, sr = sf.read(os.path.join(root, rel), dtype="float32", always_2d=True)
    return x, sr


def _band_energy_ratio(x, sr, edge_hz):
    """Energy above edge_hz relative to the total, in dB."""
    spec = np.abs(np.fft.rfft(x.astype(np.float64)))**2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    hi = spec[freqs >= edge_hz].sum()
    return 10 * math.log10((hi + 1e-20) / (spec.sum() + 1e-20))


def validate(root, full=False, sample=0.05, seed=0, expect_inventory=False):
    rep = Report()
    with open(os.path.join(root, "manifest.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    rng = random.Random(seed)

    # ---- manifest schema
    missing = [r.get("id") for r in rows if any(k not in r for k in FIELDS)]
    rep.check("manifest fields", missing, len(rows))
    ids = collections.Counter(r["id"] for r in rows)
    rep.check("unique ids", [k for k, v in ids.items() if v > 1], len(rows))
    block = [r for r in rows if r["condition"].startswith("coroom@")]
    unlabelled = [
        r["id"]
        for r in block
        if any(k not in (r.get("provenance") or {}) for k in BLOCK_LABELS) or
        not isinstance((r.get("provenance") or {}).get("room_share_db"),
                       (int, float))
    ]
    rep.check("same-room labels",
              unlabelled,
              len(block),
              detail=", ".join(BLOCK_LABELS) + " in the provenance")
    other_rule = [
        r["id"]
        for r in block
        if "room_share_rule" in (r.get("provenance") or {}) and
        r["provenance"]["room_share_rule"] != scoreevents.ROOM_SHARE_RULE
    ]
    rep.check("same-room allowance rule",
              other_rule,
              len(block),
              detail="room_share_db measured under the rule the scorer reads: "
              f"{scoreevents.ROOM_SHARE_RULE!r}")
    paired = [r for r in rows if r.get("clean_path")]
    real = [r for r in rows if not r.get("clean_path")]
    # a bed with no speech at all is reference-free by construction, as a
    # real recording is
    rep.check(
        "real items are reference-free",
        [
            r["id"] for r in real if not (r["condition"].startswith("real") or
                                          r["family"] == "noise_only")
        ],
        len(real),
    )
    rep.check(
        "paired items carry a language",
        [
            r["id"] for r in paired if not r.get("language") or
            r["language"] == "und" and r["family"] != "singer"
        ],
        len(paired),
    )

    # ---- files: existence, format, pairing, clipping, finiteness
    bad_exist = []
    bad_fmt = []
    bad_pair = []
    clipped = []
    nonfinite = []
    silent = []
    peaks = []
    for r in rows:
        for key in ("mix_path", "clean_path"):
            rel = r.get(key)
            if not rel:
                continue
            p = os.path.join(root, rel)
            if not os.path.exists(p):
                bad_exist.append(rel)
                continue
            info = sf.info(p)
            if info.samplerate != common.SR or info.channels != 1:
                bad_fmt.append(rel)
        if (r.get("clean_path") and
                os.path.exists(os.path.join(root, r["clean_path"])) and
                os.path.exists(os.path.join(root, r["mix_path"]))):
            if (sf.info(os.path.join(root, r["mix_path"])).frames
                    != sf.info(os.path.join(root, r["clean_path"])).frames):
                bad_pair.append(r["id"])
    rep.check("files exist", bad_exist, 2 * len(paired) + len(real))
    # every check below reads audio: a row whose files are missing was reported
    # above and is skipped
    rows = [
        r for r in rows if os.path.exists(os.path.join(root, r["mix_path"])) and
        (not r.get("clean_path") or
         os.path.exists(os.path.join(root, r["clean_path"])))
    ]
    paired = [r for r in rows if r.get("clean_path")]
    rep.check("48 kHz mono", bad_fmt, 2 * len(paired) + len(real))
    rep.check("mix/ref lengths match", bad_pair, len(paired))
    input_identities = collections.defaultdict(list)
    for row in rows:
        identity = audio_sha256(os.path.join(root, row["mix_path"]))
        input_identities[identity].append(row["id"])
    rep.check("unique input waveforms",
              [ids for ids in input_identities.values() if len(ids) > 1],
              len(rows),
              detail="all inputs decoded; reference reuse is allowed")
    audio_rows = (rows if full else rng.sample(rows,
                                               max(1, int(sample * len(rows)))))
    for r in audio_rows:
        x, _ = _read(root, r["mix_path"])
        peak = float(np.abs(x).max()) if len(x) else 0.0
        peaks.append(peak)
        if not np.isfinite(x).all():
            nonfinite.append(r["id"])
        if peak >= 0.999:
            clipped.append(r["id"])
        if 20 * math.log10(float(np.sqrt((x**2).mean())) + 1e-12) < -60:
            silent.append(r["id"])
    quiet_ref, noisy_ref, floors = [], [], []
    for r in audio_rows:
        # quiet@-30dBFS is quiet by design
        if r.get("clean_path") and r["family"] != "level":
            ref, _ = _read(root, r["clean_path"])
            fr = 480
            e = (ref[:len(ref) // fr * fr, 0].reshape(-1, fr)**2).mean(1)
            # frames within 30 dB of the loudest: the speech itself
            act = e > e.max() * 1e-3
            speech_db = 10 * math.log10(float(e[act].mean()) + 1e-12)
            if speech_db < REF_FLOOR_DBFS:
                quiet_ref.append(r["id"])
            # the reference's own noise floor: the quietest frames inside the
            # utterance span (pauses between words), against the speech level; a
            # noisy "clean" target defeats every reference metric, so a small
            # margin is reported and a tiny one flagged
            idx = np.nonzero(act)[0]
            span = e[idx[0]:idx[-1] + 1]
            if (len(span) >= 100 and r["family"] != "singer" and r.get(
                    # sung notes leave no pauses to measure; a
                    "score_policy") != "retention"):
                # retention item's reference is a wild recording with a floor of
                # its own

                floor_db = 10 * math.log10(
                    float(np.percentile(span, 5)) + 1e-12)
                floors.append(speech_db - floor_db)
                if speech_db - floor_db < REF_MIN_MARGIN_DB:
                    noisy_ref.append(r["id"])
    rep.check(
        "references at speech level",
        quiet_ref,
        len(audio_rows),
        detail=f"active-speech RMS >= {REF_FLOOR_DBFS} dBFS",
    )
    rep.check(
        "references clean (speech over their own floor)",
        noisy_ref,
        len(floors),
        warn_only=True,
        detail=(
            f"median margin {np.median(floors) if floors else float('nan'):.0f}"
            f" dB; flagged below {REF_MIN_MARGIN_DB} dB"),
    )
    rep.check("finite samples", nonfinite, len(audio_rows))
    rep.check(
        "no clipping in mixes",
        clipped,
        len(audio_rows),
        detail=f"peak median {np.median(peaks):.3f} max {max(peaks):.3f}",
    )
    rep.check("mixes are not silent", silent, len(audio_rows))

    # ---- checksums and set id
    ck_path = os.path.join(root, "checksums.json")
    if not os.path.exists(ck_path):
        rep.add("checksums.json", "FAIL", "missing")
    else:
        with open(ck_path, encoding="utf-8") as fh:
            ck = json.load(fh)
        expected = {
            r[k] for r in rows for k in ("mix_path", "clean_path") if r.get(k)
        }
        rep.check(
            "checksums cover every file",
            sorted(expected - set(ck["files"])),
            len(expected),
        )
        lines = "".join(f"{k}:{v}\n" for k, v in sorted(ck["files"].items()))
        if ck.get("manifest_sha256"):
            # the id binds the audio and the item semantics: a changed crop,
            # target policy or scored region is a different set
            # even when every file hash still matches
            fields = ck.get("public_fields") or PUBLIC_FIELDS
            h = hashlib.sha256()
            for r in sorted(rows, key=lambda r: r["id"]):
                public = {k: r.get(k) for k in fields}
                h.update(json.dumps(public, sort_keys=True).encode())
                h.update(b"\n")
            semantics_ok = h.hexdigest() == ck["manifest_sha256"]
            digest = hashlib.sha256(
                (lines + f"manifest:{ck['manifest_sha256']}\n").encode())
            rep.add(
                "manifest semantics match checksums",
                "PASS" if semantics_ok else "FAIL",
                "public fields hashed: " + ", ".join(fields),
            )
        else:
            digest = hashlib.sha256(lines.encode())
        rep.add(
            "set id reproduces from the hashes",
            "PASS" if digest.hexdigest() == ck["set_id"] else "FAIL",
            ck["set_id"][:16],
        )
        rehash = (list(ck["files"]) if full else rng.sample(
            sorted(ck["files"]), max(1, int(sample * len(ck["files"])))))
        stale = [
            rel for rel in rehash
            if not os.path.exists(os.path.join(root, rel)) or
            _sha256(os.path.join(root, rel)) != ck["files"][rel]
        ]
        rep.check(
            "file hashes match" + ("" if full else " (sample)"),
            stale,
            len(rehash),
        )

    # ---- counts and balance
    counts = collections.Counter(r["condition"] for r in rows)
    small = [
        f"{c}={n}" for c, n in sorted(counts.items()) if n < ranking.MIN_ITEMS
    ]
    rep.check(
        "conditions at the leaderboard minimum",
        small,
        len(counts),
        warn_only=True,
        detail=f"< {ranking.MIN_ITEMS} items cannot feed an axis",
    )
    if expect_inventory:
        short = [
            f"{c}={counts.get(c, 0)}<{m}"
            for c, m in EXPECTED_INVENTORY.items()
            if counts.get(c, 0) < m
        ]
        rep.check(
            "condition inventory complete",
            short,
            len(EXPECTED_INVENTORY),
            detail=("every promised condition present with at least its minimum"
                    " items"),
        )
    langs = collections.Counter(
        r["language"]
        for r in paired
        if r["family"] != "singer" and r.get("language") not in (None, "und"))
    top, top_n = max(langs.items(), key=lambda kv: kv[1]) if langs else ("?", 0)
    others = [n for l, n in langs.items() if l != top]
    lo, hi = (min(others), max(others)) if others else (top_n, top_n)
    # one language may carry a designed share (the applications' first
    # language); the others must be balanced among themselves so every language
    # is covered comparably
    rep.add(
        "language balance (paired items)",
        "PASS" if hi <= 1.25 * lo else "WARN",
        f"{top} {top_n / max(1, sum(langs.values())):.0%} of paired items; "
        f"{len(others)} other languages, {lo}-{hi} items each",
    )

    # ---- golden transcripts (an optional file; when it ships, every row must
    # fit the manifest)
    tpath = os.path.join(root, "transcripts.jsonl")
    if os.path.exists(tpath):
        by_id = {r["id"]: r for r in rows}
        bad, seen, tiers, n_t = [], set(), collections.Counter(), 0
        with open(tpath, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                t = json.loads(line)
                n_t += 1
                tid = t.get("id")
                if tid in seen:
                    bad.append(f"{tid}: duplicate")
                seen.add(tid)
                r = by_id.get(tid)
                if r is None:
                    bad.append(f"{tid}: not in the manifest")
                    continue
                if not r.get("clean_path"):
                    bad.append(f"{tid}: a reference-free item")
                if not (t.get("text") or "").strip():
                    bad.append(f"{tid}: empty text")
                if t.get("language") != r.get("language"):
                    bad.append(f"{tid}: language {t.get('language')} !="
                               f" {r.get('language')}")
                # optional: the language heard, where the label is wrong
                spoken = t.get("spoken")
                if spoken is not None and not (
                        isinstance(spoken, str) and 2 <= len(spoken) <= 3 and
                        spoken.isalpha() and spoken != t.get("language")):
                    bad.append(
                        f"{tid}: spoken {spoken!r} is not another language code"
                    )
                if not t.get("tier"):
                    bad.append(f"{tid}: no tier")
                tiers[t.get("tier")] += 1
        rep.check(
            "golden transcripts consistent",
            bad,
            n_t,
            detail=f"{n_t} transcripts; tiers {dict(tiers)}",
        )

    # ---- scored region
    short = []
    for r in rows:
        if r.get("score_from"):
            n = sf.info(os.path.join(root, r["mix_path"])).frames
            if n - int(r["score_from"]) < MIN_SCORED_SECONDS * common.SR:
                short.append(r["id"])
    rep.check(
        "scored region after score_from",
        short,
        sum(1 for r in rows if r.get("score_from")),
        detail=f">= {MIN_SCORED_SECONDS:.0f} s",
    )

    # ---- realised SNR of additive conditions
    off, n_snr = [], 0
    for r in (audio_rows if not full else rows):
        if (r.get("snr") is None or not r.get("clean_path") or
                r["family"] in SKIP_SNR_FAMILIES):
            continue
        if r["condition"].endswith("_tail") or "tail" in r["condition"]:
            # the reference is tapered and padded: the nominal SNR refers to the
            # untapered speech
            continue
        mix, _ = _read(root, r["mix_path"])
        ref, _ = _read(root, r["clean_path"])
        noise = mix[:, 0] - ref[:, 0]
        p_ref, p_noise = float((ref**2).mean()), float((noise**2).mean())
        if p_noise <= 0:
            continue
        n_snr += 1
        realised = 10 * math.log10(p_ref / p_noise)
        if abs(realised - float(r["snr"])) > SNR_TOLERANCE_DB:
            off.append(f"{r['id']}:{realised:+.1f}dB")
    rep.check(
        "realised SNR within tolerance",
        off,
        n_snr,
        detail=f"±{SNR_TOLERANCE_DB} dB of nominal",
    )

    # ---- bandwidth gate (field, plus a measured sample)
    # a retention item's reference was recorded in the wild: phone bandwidth is
    # the point
    gated = [
        r for r in paired
        if r["family"] != "singer" and r.get("score_policy") != "retention"
    ]
    low = [
        r["id"]
        for r in gated
        if (r.get("effective_bandwidth_hz") or 0) < BANDWIDTH_FLOOR_HZ
    ]
    rep.check("speech bandwidth field >= 20 kHz", low, len(gated))
    try:
        measured = []
        for r in [
                r for r in audio_rows
                if r.get("clean_path") and r["family"] not in
            ("singer", "srsweep") and r.get("score_policy") != "retention"
        ][:40]:
            ref, sr = _read(root, r["clean_path"])
            # pylint: disable-next=import-outside-toplevel
            import torch

            bw = common.effective_bandwidth_hz(torch.from_numpy(ref[:, 0]), sr)
            if bw < BANDWIDTH_FLOOR_HZ:
                measured.append(f"{r['id']}:{bw:.0f}Hz")
        rep.check("speech bandwidth measured (sample)",
                  measured,
                  40,
                  warn_only=True)
    except ImportError:
        rep.add(
            "speech bandwidth measured (sample)",
            "WARN",
            "builder not importable",
        )

    # ---- rate-sweep twins are band-limited identically
    twin_bad, twin_n = [], 0
    for r in [r for r in rows if r["family"] == "srsweep"]:
        rate = r["condition"].split("_")[-1].replace("k", "")
        edge = float(rate) * 1000 / 2
        if edge >= common.SR / 2:
            continue
        mix, _ = _read(root, r["mix_path"])
        ref, _ = _read(root, r["clean_path"])
        twin_n += 1
        a = _band_energy_ratio(mix[:, 0], common.SR, edge * 1.1)
        b = _band_energy_ratio(ref[:, 0], common.SR, edge * 1.1)
        if a > -40 or b > -40:
            twin_bad.append(f"{r['id']}:{a:.0f}/{b:.0f}dB")
    rep.check(
        "rate-sweep twins band-limited",
        twin_bad,
        twin_n,
        detail="energy above the band < -40 dB",
    )
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=common.EVAL)
    ap.add_argument(
        "--full",
        action="store_true",
        help="hash and inspect every file, not a sample",
    )
    ap.add_argument(
        "--expect-inventory",
        action="store_true",
        help=("require every promised condition with its minimum item count (a"
              " complete build)"),
    )
    ap.add_argument("--json", default="", help="write the check table here")
    a = ap.parse_args()
    rep = validate(a.set, full=a.full, expect_inventory=a.expect_inventory)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    dict(
                        check=n,
                        status=s,
                        detail=d,
                        failing=rep.failing.get(n, []),
                    ) for n, s, d in rep.rows
                ],
                fh,
                indent=1,
            )
    print("\nRESULT:", "OK" if rep.ok else "FAILURES")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
