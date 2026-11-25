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
"""Noise recorded with the voice: transient events and quiet floors in the
voice's own room and microphone, and the voice's level while they go.

The same-room items are paired: the voice after the room and the microphone
is the reference, and the mixture adds, through the same room and microphone,
isolated transient events (one-shots in the pauses or under the speech) and a
quiet floor, every one of them labelled in the item's provenance. The block
is read from two renders of every system: the mixtures, as for every other
item, and the references, which ``--pair`` links under ``<pair>/in`` by the
same file names and which ``runfield.sh`` with ``PROBE_IN`` and
``runsystem.py --inputs`` render into ``<pair>/out/<system>`` in the
environment that rendered the board (the receipts say so). What the noise
did to the output is the difference between the two renders, whatever the
system makes of the voice's own room: a system that removes the room is
charged for the room on neither row. Three rows are read per item, each
bounded to ``ITEM_CLIP_DB`` before pooling:

``event_resid_db`` is, over each labelled event span (kept open ``DECAY_S``
after its end for the room's decay), the energy of the difference between
the two renders against the energy the mixture added there, in dB, every
event bounded to ``ITEM_CLIP_DB`` and the item reading their mean: zero
leaves the event as it was, ``-ITEM_CLIP_DB`` is an event removed outright.
``floor_resid_db`` is the same ratio over the labelled pauses outside the
events. ``voice_level_db`` is the lower of two reads of the voice on the
reference's loudest fifth of speech frames, in dB: the mixture's render
against the reference's render, what the noise costs the voice; and the
mixture's render against the reference itself with the room's share of
those frames allowed (``room_share_db``: how much louder the voice is on
them with its room than without it at the recording's level, the room-free
voice placed at the direct sound's arrival, where a system that removes the
room renders it; measured when the set was built), capped at zero. A
system that removes the room is charged for it on neither; one that turns
the voice down, with the noise there or without, is; and a render with no
voice on those frames reads the bound. The system's delay (its median lag
on clean speech, as its base scores record
it) and the gain its base scores record are undone on both renders first; a
flipped polarity cannot matter, since every row reads energies of the
difference between the two renders or ratios of their energies. The
unbounded means stand beside the two residual rows (``*_raw_db``, floored
at ``RAW_FLOOR_DB``) for reads outside the board, and every cell counts the
items carrying each row (``metric_counts``). Removal is credited only while
the voice is kept: an item whose voice row is at or under ``VOICE_GATE_DB``
reads its event and floor rows as left (zero), and an event under speech
whose span lost the voice in the reference's render, beyond the room's
share, reads as left; silencing the output, or deleting the speech an event
sits on, earns nothing.

The real recordings have no reference. ``_real_events`` pools the items on
which the shared detector finds a transient on the INPUT outside the speech
the VAD marks (a rise of ``JUMP_DB`` within one 10 ms frame to
``ABOVE_MEDIAN_DB`` over the running median of the previous half second, kept
until the level is back within 6 dB of that median): ``event_att_db`` is the
level change of the output over those spans, ``voice_level_db`` the level
change on the input's loudest fifth of speech frames. An item without such an
event is counted as having none, never as missing.

Usage: scoreevents.py --pair [--pair-dir DIR]
       scoreevents.py --systems a,b [--pair-dir DIR] [--reuse-from DIR]
                      [--aggregate-only DIR]
Output: <BENCH_RESULTS>/<system>.events.json and .events.items.jsonl
"""

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np
import torch

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors
import common
import levels
import observations
import scorebench

sys.path.insert(0, os.path.join(common.BENCH_DIR, "wrappers"))
import receipt

SCORER_SHA256 = observations.implementation(
    __file__,
    common.__file__,
    observations.__file__,
    levels.__file__,
    scorebench.__file__,
)

FRAME = 480
DECAY_S = 0.15
EDGE_S = 0.3
JUMP_DB, ABOVE_MEDIAN_DB = 12.0, 15.0
RELEASE_DB = 6.0
HISTORY_FRAMES = 50
# the loudest fifth of the speech frames give the voice's level
TOP_SHARE = 0.2
# no item reads beyond this much credit or penalty in any pooled row
ITEM_CLIP_DB = 20.0
# the unbounded rows stop here, where the difference between the two renders
# is numerically nothing
RAW_FLOOR_DB = -60.0
# removal is credited only while the voice is kept: an item whose voice row
# is at or under this (the declared floor of the preservation row) reads its
# noise rows as left, and an event under speech whose span lost the voice in
# the reference's render reads as left; removing the voice is not removing
# the noise
VOICE_GATE_DB = -3.0
# the pair of renders: the references under <pair>/in, identified like the
# streaming probe's inputs (receipt.py reads probe.json beside the folder)
PAIR_KIND = "same_room_references"
PAIR_VERSION = "1"
SAME_ROOM = ("coroom@pause", "coroom@speech", "coroom@floor", "coroom@both")
EVENT_CONDS = ("coroom@pause", "coroom@speech", "coroom@both")
FLOOR_CONDS = ("coroom@floor", "coroom@both")
REAL = ("real_dirty", "real_multi", "real_noisy", "real_wild", "real_everyday")


def frames_db(x: torch.Tensor) -> np.ndarray:
    """Level of every 10 ms frame of ``x`` in dBFS."""
    n = len(x) // FRAME
    energy = x[:n * FRAME].view(n, FRAME).pow(2).mean(1).numpy()
    return 10 * np.log10(energy + 1e-12)


def detect_events(db: np.ndarray) -> list:
    """Event spans [start, end) in frames by the shared definition."""
    db = np.asarray(db, dtype=np.float64)
    n = len(db)
    spans = []
    jumps = np.flatnonzero(np.diff(db) >= JUMP_DB) + 1
    allowed = 1
    for i in jumps:
        if i < allowed:
            continue
        lo = max(0, i - HISTORY_FRAMES)
        median = float(np.median(db[lo:i])) if i - lo > 5 else float(db[i])
        if db[i] - median < ABOVE_MEDIAN_DB:
            continue
        j = i
        while j + 1 < n and db[j + 1] > median + RELEASE_DB and j - i < 60:
            j += 1
        spans.append((int(i), int(j + 1)))
        allowed = j + 30
    return spans


def _frame_mask(n_frames: int,
                spans_s: list,
                tail_s: float = 0.0) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in spans_s:
        a = max(0, int(start * common.SR) // FRAME)
        b = min(n_frames, int((end + tail_s) * common.SR) // FRAME + 1)
        mask[a:b] = True
    return mask


def _energy(x: torch.Tensor, mask: np.ndarray) -> float:
    """The energy of ``x`` over the frames ``mask`` marks, summed in float64."""
    frames = x[:len(mask) * FRAME].view(len(mask), FRAME).double().pow(2)
    return float(frames.sum(1).numpy()[mask].sum())


def _loud_speech(reference_db: np.ndarray, speech: np.ndarray) -> np.ndarray:
    if not speech.any():
        return speech
    cut = np.quantile(reference_db[speech], 1.0 - TOP_SHARE)
    return speech & (reference_db >= cut)


def pair_dir(path: str = "") -> str:
    """Returns the pair's directory.

    The flag, ``BENCH_PAIR``, or ``pair_coroom`` beside ``BENCH_OUT``.
    """
    return (path or os.environ.get("BENCH_PAIR") or os.path.join(
        os.path.dirname(os.path.abspath(common.OUT_ROOT.rstrip("/"))),
        "pair_coroom",
    ))


def pair_rows(rows: list) -> list:
    """Returns the same-room items with a reference, in id order.

    The complete inventory.
    """
    return sorted(
        (r for r in rows
         if r["condition"] in SAME_ROOM and r.get("clean_path")),
        key=lambda r: r["id"],
    )


def identity_of(hashes) -> str:
    """Returns one identity for a set of inputs or outputs: the sha256 of their
    sorted sample hashes."""
    return hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()


def pair_problem(root: str, rows: list) -> str | None:
    """Returns why ``<root>/probe.json`` is not this build's pair, or None.

    It must name this build and the complete inventory of same-room items,
    every input verified by its samples.
    """
    path = os.path.join(root, "probe.json")
    if not os.path.isfile(path):
        return "no probe.json"
    with open(path, encoding="utf-8") as fh:
        pair = json.load(fh)
    if pair.get("kind") != PAIR_KIND:
        return "not the same-room pair (another probe's inputs)"
    if pair.get("set_id") != common.set_id():
        return (f"linked from set {str(pair.get('set_id'))[:12]}, the current"
                f" set is {str(common.set_id())[:12]}")
    if pair.get("pair_version") != PAIR_VERSION:
        return (f"linked by pair version {pair.get('pair_version')!r}, this"
                f" scorer links version {PAIR_VERSION!r}")
    wanted = [r["id"] for r in pair_rows(rows)]
    if [e["id"] for e in pair.get("items", [])] != wanted:
        return ("its items are not the complete inventory of the set's"
                " same-room items")
    if pair.get("identity") != identity_of(e["sha256"] for e in pair["items"]):
        return "its identity does not match its inputs"
    in_dir = os.path.join(root, "in")
    listed = {e["id"] for e in pair["items"]}
    stray = (sorted(set(os.listdir(in_dir)) -
                    listed) if os.path.isdir(in_dir) else [])
    if stray:  # runsystem.py --inputs identifies the directory as it stands
        return (f"{len(stray)} files under in/ are not pair inputs (first:"
                f" {stray[0]})")
    for entry in pair["items"]:
        file = os.path.join(in_dir, entry["id"])
        if (not os.path.isfile(file) or
                scorebench.sample_hash(common.load48(file)) != entry["sha256"]):
            return f"{entry['id']} is not the recorded pair input"
    return None


def write_pair(rows: list, root: str) -> dict:
    """Links the same-room references under ``<root>/in`` and records them.

    The record is ``<root>/probe.json``; a pair that is already current is
    left as it is, so renders made from it stay valid.
    """
    items = pair_rows(rows)
    if not items:
        raise SystemExit("the set has no same-room items with a reference")
    if pair_problem(root, rows) is None:
        with open(os.path.join(root, "probe.json"), encoding="utf-8") as fh:
            return json.load(fh)
    in_dir = os.path.join(root, "in")
    os.makedirs(in_dir, exist_ok=True)
    # the directory is exactly the pair, nothing else
    for stale in os.listdir(in_dir):
        os.remove(os.path.join(in_dir, stale))
    listed = []
    for r in items:
        target = os.path.join(in_dir, r["id"])
        try:
            os.link(r["clean_path"], target)
        except OSError:
            shutil.copyfile(r["clean_path"], target)
        listed.append({
            "id": r["id"],
            "sha256": scorebench.sample_hash(common.load48(target)),
        })
    pair = {
        "kind": PAIR_KIND,
        "pair_version": PAIR_VERSION,
        "set_id": common.set_id(),
        "source_conditions": list(SAME_ROOM),
        "items": listed,
    }
    pair["identity"] = identity_of(e["sha256"] for e in listed)
    tmp = os.path.join(root, f"probe.json.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(pair, fh, indent=1)
    os.replace(tmp, os.path.join(root, "probe.json"))
    return pair


def read_pair(root: str, rows: list) -> dict:
    """Returns the pair's record once its inputs are verified."""
    problem = pair_problem(root, rows)
    if problem:
        raise SystemExit(
            f"{os.path.join(root, 'probe.json')}: {problem}; link the pair"
            " again (scoreevents.py --pair) and render it")
    with open(os.path.join(root, "probe.json"), encoding="utf-8") as fh:
        return json.load(fh)


def pair_render_problem(name: str, results: str, root: str,
                        pair: dict) -> str | None:
    """Returns why the pair render of ``name`` does not stand for it, or None.

    Every pair render carries the receipts its processes wrote: they must
    agree on the environment the board's stamp records
    (``render_environment_sha256``: code, weights, packages, device and
    arguments, the inputs aside) and name this pair's inputs.
    """
    path = os.path.join(results, f"{name}.meta.json")
    if not os.path.isfile(path):
        return "no stamp for the board's render (meta.json missing)"
    with open(path, encoding="utf-8") as fh:
        board = json.load(fh)
    if not board.get("render_environment_sha256"):
        return ("the board's stamp records no render environment: re-render"
                " the board")
    out_dir = os.path.join(root, "out", name)
    if not os.path.isdir(out_dir):
        return f"no pair render under {os.path.join(root, 'out')}"
    docs = receipt.read_all(out_dir)
    if not docs:
        return "the pair render carries no receipt"
    environments = {d.get("environment_digest") for d in docs}
    if environments != {board["render_environment_sha256"]}:
        return (
            "the pair was rendered in another environment than the board's"
            " render (receipts"
            f" {', '.join(str(e)[:12] for e in sorted(map(str, environments)))}"
            f" vs stamp {board['render_environment_sha256'][:12]})")
    inputs = {
        (d.get("identity", {}).get("inputs") or {}).get("probe") for d in docs
    }
    if inputs != {pair["identity"]}:
        return "the pair render was made from other inputs: render it again"
    return None


def score_same_room(it: dict,
                    out: torch.Tensor,
                    mix: torch.Tensor,
                    ref: torch.Tensor,
                    out_ref: torch.Tensor,
                    gain_db: float,
                    lag: int = 0,
                    room_share_db: float = 0.0) -> dict:
    """Energies of one same-room item: per event, over the floor, on the voice.

    ``out`` renders the mixture, ``out_ref`` the reference; the system's
    delay ``lag`` (samples) and gain are undone on both. ``room_share_db``
    is the room's share of the reference's level on its loudest speech
    frames, allowed to a system that removes the room.
    """
    n = min(len(out), len(mix), len(ref), len(out_ref))
    scale = 10**(-gain_db / 20)
    o = scale * common.apply_lag(out, lag, n)
    p = scale * common.apply_lag(out_ref, lag, n)
    x, r = mix[:n], ref[:n]
    n_frames = n // FRAME
    prov = it.get("provenance") or {}
    events = [(e[0], e[1]) for e in prov.get("events", [])]
    event_mask = _frame_mask(n_frames, events, DECAY_S)
    pauses = _frame_mask(n_frames, prov.get("pauses", []))
    edge = np.zeros(n_frames, dtype=bool)
    k = int(EDGE_S * common.SR) // FRAME
    edge[:k] = edge[n_frames - k:] = True
    floor = pauses & ~event_mask & ~edge
    ref_db = frames_db(r)
    speech = (ref_db > np.median(ref_db) + 6.0) & ~pauses & ~edge
    loud = _loud_speech(ref_db, speech)
    added, left = o - p, x - r
    kinds = [str(e[2]) if len(e) > 2 else "" for e in prov.get("events", [])]
    return {
        "lag": lag,
        # per event: what the noise did, what the mixture added, and the voice
        # the reference's render keeps on the span against the reference
        "events": [[
            _energy(added, _frame_mask(n_frames, [e], DECAY_S)),
            _energy(left, _frame_mask(n_frames, [e], DECAY_S)),
            _energy(p, _frame_mask(n_frames, [e], DECAY_S)),
            _energy(r, _frame_mask(n_frames, [e], DECAY_S)),
        ] for e in events],
        "event_kinds": kinds,
        "floor": [_energy(added, floor),
                  _energy(left, floor)],
        "floor_frames": int(floor.sum()),
        "voice": [_energy(o, loud), _energy(p, loud)],
        "voice_ref": _energy(r, loud),
        "room_share_db": float(room_share_db),
        "loud_frames": int(loud.sum()),
    }


def score_real(it: dict, out: torch.Tensor, mix: torch.Tensor, vad,
               gain_db: float) -> dict:
    """Energies of one real recording over the events the input holds."""
    n = min(len(out), len(mix))
    lag, sign = common.best_alignment(out[:n], mix[:n])
    o = sign * common.apply_lag(out, lag, n) * 10**(-gain_db / 20)
    x = mix[:n]
    n_frames = n // FRAME
    _, segs = (scorebench.speech_segments(vad, x, it["mix_path"])
               if n == len(mix) else vad.segments(x))
    speech = np.zeros(n_frames, dtype=bool)
    for s in segs:
        speech[s["start"] * 3 // FRAME:min(n_frames, s["end"] * 3 // FRAME +
                                           1)] = True
    in_db = frames_db(x)
    spans = [(i, j) for i, j in detect_events(in_db) if not speech[i:j].any()]
    event_mask = np.zeros(n_frames, dtype=bool)
    for i, j in spans:
        event_mask[i:min(n_frames, j +
                         int(DECAY_S * common.SR) // FRAME)] = True
    loud = _loud_speech(in_db, speech & ~event_mask)
    return {
        "lag": lag,
        "polarity": sign,
        "events": [[
            _energy(
                o,
                _frame_mask(n_frames,
                            [(i * FRAME / common.SR, j * FRAME / common.SR)],
                            DECAY_S)),
            _energy(
                x,
                _frame_mask(n_frames,
                            [(i * FRAME / common.SR, j * FRAME / common.SR)],
                            DECAY_S)),
        ] for i, j in spans],
        "event_spans_s": [[
            round(i * FRAME / common.SR, 3),
            round(j * FRAME / common.SR, 3)
        ] for i, j in spans],
        "voice": [_energy(o, loud), _energy(x, loud)],
        "loud_frames": int(loud.sum()),
    }


def bounded(db: float,
            low: float = -ITEM_CLIP_DB,
            high: float = ITEM_CLIP_DB) -> float:
    return float(min(max(db, low), high))


def _ratio_db(num: float, den: float) -> float:
    if den <= 0:
        return math.nan
    return 10 * math.log10(max(num, 0.0) / den + 1e-20)


def same_room_metrics(m: dict, condition: str) -> dict:
    """The item's rows from its energies, each bounded, the residual rows
    beside their unbounded means; removal credited only while the voice is
    kept (``VOICE_GATE_DB``)."""
    out = {}
    share = float(m.get("room_share_db") or 0.0)
    if m.get("loud_frames", 0) > 0 and m.get("voice_ref", 0.0) > 0:
        with_noise, without = m["voice"]
        if with_noise <= 0 or without <= 0:
            # no voice on the loudest speech frames of a render: the voice
            # is gone, whatever the other render holds
            out["voice_level_db"] = -ITEM_CLIP_DB
        else:
            pair = _ratio_db(with_noise, without)
            kept = min(0.0, _ratio_db(with_noise, m["voice_ref"]) + share)
            out["voice_pair_db"] = bounded(pair)
            out["voice_kept_db"] = bounded(kept)
            out["voice_level_db"] = bounded(min(pair, kept))
    voice_gone = out.get("voice_level_db", 0.0) <= VOICE_GATE_DB
    if condition in EVENT_CONDS and m.get("events"):
        kinds = m.get("event_kinds") or [""] * len(m["events"])
        values = []
        for entry, kind in zip(m["events"], kinds):
            if entry[1] <= 0:
                continue
            value = _ratio_db(entry[0], entry[1])
            if voice_gone:
                value = 0.0
            elif kind == "speech" and len(entry) >= 4 and entry[3] > 0:
                # the voice the reference's render keeps on the span, the
                # room's share allowed: a span that lost it was not cleaned
                if _ratio_db(entry[2], entry[3]) + share <= VOICE_GATE_DB:
                    value = 0.0
            values.append(value)
        if values:
            # each event bounded on its own: one of two removed is half credit
            out["event_resid_db"] = float(np.mean([bounded(v) for v in values]))
            out["event_resid_raw_db"] = float(
                np.mean([max(v, RAW_FLOOR_DB) for v in values]))
    if (condition in FLOOR_CONDS and m.get("floor_frames", 0) > 0 and
            m["floor"][1] > 0):
        raw = 0.0 if voice_gone else max(_ratio_db(*m["floor"]), RAW_FLOOR_DB)
        out["floor_resid_db"] = bounded(raw)
        out["floor_resid_raw_db"] = raw
    return out


def real_metrics(m: dict) -> dict:
    out = {"n_events": len(m.get("events", []))}
    values = [_ratio_db(a, b) for a, b in m.get("events", []) if b > 0]
    if values:
        out["event_att_db"] = bounded(float(np.mean(values)))
    if m.get("loud_frames", 0) > 0 and m["voice"][1] > 0:
        out["voice_level_db"] = bounded(_ratio_db(*m["voice"]))
    return out


def _render(path: str, length: int) -> tuple:
    """Returns (samples, problem) of one render: the file at ``path``, of
    ``length`` samples and finite."""
    if not os.path.exists(path):
        return None, "missing"
    out = common.load48(path)
    if len(out) != length:
        return None, "wrong_length"
    if not bool(torch.isfinite(out).all()):
        return None, "nonfinite"
    return out, None


# the labels a same-room measurement reads, and the rule that produced the room
# allowance: both enter every observation's identity, so a label edit or a new
# allowance rule invalidates the cached measurements and the controls that read
# them instead of leaving a valid-looking cache
SCORE_LABELS = ("events", "pauses", "floor", "split", "room_share_db",
                "room_share_rule")
# v2: the room-free voice sits at the direct sound's arrival, where a system
# that removes the room renders it (v1 placed it at its cross-correlation peak
# with the reverberant reference, a median 12 ms later, which read a perfect
# room remover 0.4 dB under the allowance). The set's provenance records the
# rule its shares were measured under (``room_share_rule``); the scorer reads
# only shares measured under this one, and every label digest carries the
# recorded rule.
ROOM_SHARE_RULE = ("level-matched dry share on the reference's loudest fifth of"
                   " speech frames, the dry voice at the direct arrival, v2")


def labels_digest(row: dict) -> str | None:
    """Returns the digest of the labels a same-room item is scored by.

    None for an item outside the block, whose measurements read no label.
    """
    if row.get("condition") not in SAME_ROOM:
        return None
    prov = row.get("provenance") or {}
    labels = {key: prov.get(key) for key in SCORE_LABELS}
    return hashlib.sha256(json.dumps(labels,
                                     sort_keys=True).encode()).hexdigest()


def labels_identity(rows: list) -> str:
    """Returns one digest of the labels every same-room row of ``rows`` is
    scored by, which a sidecar carries so that a frozen view can be traced to
    the labels it measured."""
    return anchors.digest(
        sorted([r["id"], labels_digest(r)]
               for r in rows
               if r.get("condition") in SAME_ROOM))


def stale_labels(records: list, rows: list) -> str | None:
    """Returns why ``records`` cannot derive a view of ``rows``, or None.

    A same-room record measured under labels other than its row's: the
    scoring pass measures such an item again, a view never may.
    """
    want = {
        r["id"]: labels_digest(r)
        for r in rows
        if r.get("condition") in SAME_ROOM
    }
    stale = sorted(rec["id"]
                   for rec in records
                   if rec["id"] in want and rec.get("output_status") == "ok" and
                   (rec.get("identity") or {}).get("labels") != want[rec["id"]])
    if not stale:
        return None
    return (f"{len(stale)} same-room records were measured under other labels"
            f" (first: {stale[0]}): score the whole set again")


def room_shares(rows: list) -> dict:
    """Returns {item id: the room's share of the voice's level, dB} of the
    same-room items, from their provenance (``room_share_db``).

    An item without the figure allows nothing: a system that removes its
    room is then charged for it on the voice row.

    Raises:
        SystemExit: An item's share was measured under another rule than
            ``ROOM_SHARE_RULE`` (or under none recorded, the v1 sets): the
            set's shares must be re-measured before the block is scored.
    """
    shares = {}
    for r in rows:
        if r["condition"] in SAME_ROOM:
            prov = r.get("provenance") or {}
            rule = prov.get("room_share_rule")
            if rule != ROOM_SHARE_RULE:
                raise SystemExit(
                    f"{r['id']}: room_share_db was measured under "
                    f"{rule or 'an unrecorded rule (v1)'}; this scorer reads "
                    f"{ROOM_SHARE_RULE!r}. Re-measure the set's shares before "
                    "scoring the same-room block.")
            value = prov.get("room_share_db")
            shares[r["id"]] = float(value) if value is not None else 0.0
    return shares


def assemble(name,
             rows,
             vad,
             gain_db,
             reuse=None,
             hashes=None,
             pair_out=None,
             lag=0):
    """Returns per-item records over ``rows``.

    ``pair_out`` holds the system's renders of the same-room references by
    the items' names (``<pair>/out/<system>``); INPUT's are the references
    themselves; ``lag`` is the system's delay in samples. An item whose
    output samples are unchanged since an earlier pass keeps its record
    (``reuse``), read at the same gain and delay.
    """
    hashes = hashes or common.file_hashes()
    reuse = reuse or {}
    shares = room_shares(rows)
    records, counts = [], collections.Counter()
    for it in rows:
        record = observations.item(it)
        records.append(record)
        op = (it["mix_path"] if name == "INPUT" else os.path.join(
            common.OUT_ROOT, name, it["id"]))
        mix = common.load48(it["mix_path"])
        out, problem = _render(op, len(mix))
        if problem:
            record["output_status"] = problem
            continue
        record["identity"] = scorebench.item_identity(
            it, scorebench.sample_hash(out), hashes)
        record["identity"]["labels"] = labels_digest(it)
        ref = out_ref = None
        if it.get("clean_path"):
            ref = common.load48(it["clean_path"])
            if name == "INPUT":
                out_ref = ref
            else:
                out_ref, problem = _render(
                    os.path.join(pair_out or "", it["id"]), len(ref))
                if problem:
                    record["output_status"] = f"{problem}_pair"
                    continue
            record["identity"]["pair_output"] = scorebench.sample_hash(out_ref)
        old = reuse.get(scorebench.reuse_key(record["identity"]))
        if (old is not None and old.get("gain_db") == gain_db and
                old.get("lag", 0) == lag):
            record["measurements"] = dict(old["measurements"])
            counts["reused"] += 1
        elif ref is not None:
            record["measurements"] = observations.finite(
                score_same_room(it, out, mix, ref, out_ref, gain_db, lag,
                                shares.get(it["id"], 0.0)))
            counts["scored"] += 1
        else:
            record["measurements"] = observations.finite(
                score_real(it, out, mix, vad, gain_db))
            counts["scored"] += 1
        record["gain_db"] = gain_db
        record["lag"] = lag
    if reuse:
        print(
            f"  reused {counts['reused']} records, scored {counts['scored']}"
            " items",
            flush=True,
        )
    return records


def at_gain(m: dict, from_db: float | None, to_db: float, paired: bool) -> dict:
    """Returns an item's energies re-read at another gain.

    The output's energies were undone at ``from_db`` when the item was scored;
    a view calibrated at ``to_db`` reads them scaled by the difference. The
    mixture's and the reference's energies do not move.
    """
    if from_db is None or abs(float(from_db) - to_db) < 1e-12:
        return m
    k = 10**((float(from_db) - to_db) / 10)
    m = dict(m)
    if m.get("events"):
        # the output's energies move: what the noise did and the voice the
        # reference's render keeps; the mixture's and the reference's stay
        m["events"] = [[e[0] * k, e[1]] +
                       ([e[2] * k, e[3]] if len(e) >= 4 else [])
                       for e in m["events"]]
    if m.get("floor"):
        m["floor"] = [m["floor"][0] * k, m["floor"][1]]
    if m.get("voice"):
        o, other = m["voice"]
        m["voice"] = [o * k, other * k if paired else other]
    return m


def aggregate(records: list, rows: list, level_cal_db: float) -> dict:
    """Returns the sidecar file for ``rows``: per condition the mean of its
    items' rows, pooled over the same-room block (``_coroom``) and over the
    real recordings with an event (``_real_events``). Pure, so any view of the
    set derives from the whole-set records, read at the view's own
    calibration ``level_cal_db``."""
    by_id = {r["id"]: r for r in records}
    out = {}
    pooled = collections.defaultdict(list)
    real_items, real_events, real_without = 0, 0, 0
    for it in rows:
        cond = it["condition"]
        rec = by_id.get(it["id"])
        cell = out.setdefault(cond, {"n_expected": 0, "n_missing": 0, "n": 0})
        cell["n_expected"] += 1
        if rec is None or rec.get("output_status") != "ok":
            cell["n_missing"] += 1
            continue
        m = observations.revive(rec.get("measurements", {}))
        m = at_gain(m, rec.get("gain_db"), level_cal_db, cond in SAME_ROOM)
        if cond in SAME_ROOM:
            values = same_room_metrics(m, cond)
            key = "_coroom"
        else:
            values = real_metrics(m)
            key = "_real_events"
            real_items += 1
            real_events += values["n_events"]
            if not values["n_events"]:
                real_without += 1
        rec["metrics"] = observations.finite(values)
        cell["n"] += 1
        for k, v in values.items():
            if isinstance(v, float) and math.isfinite(v):
                cell.setdefault("_values", {}).setdefault(k, []).append(v)
                pooled[(key, k)].append(v)
    for cell in out.values():
        values = cell.pop("_values", {})
        for k, vs in values.items():
            cell[k] = float(np.mean(vs))
        # the items carrying each row, so coverage is read per row
        cell["metric_counts"] = {k: len(vs) for k, vs in values.items()}
    if any(c in out for c in SAME_ROOM):
        block = {
            k: float(np.mean(vs))
            for (key, k), vs in pooled.items()
            if key == "_coroom"
        }
        block["metric_counts"] = {
            k: len(vs) for (key, k), vs in pooled.items() if key == "_coroom"
        }
        block.update({
            "n":
                sum(out[c]["n"] for c in SAME_ROOM if c in out),
            "n_expected":
                sum(out[c]["n_expected"] for c in SAME_ROOM if c in out),
            "n_missing":
                sum(out[c]["n_missing"] for c in SAME_ROOM if c in out),
        })
        out["_coroom"] = block
    if real_items:
        real = {
            k: float(np.mean(vs))
            for (key, k), vs in pooled.items()
            if key == "_real_events"
        }
        real["metric_counts"] = {
            k: len(vs)
            for (key, k), vs in pooled.items()
            if key == "_real_events"
        }
        real.update({
            "n": real_items - real_without,
            "n_expected": sum(out[c]["n_expected"] for c in REAL if c in out),
            "n_missing": sum(out[c]["n_missing"] for c in REAL if c in out),
            "n_without_events": real_without,
            "n_events": real_events,
        })
        out["_real_events"] = real
    out["_level_cal_db"] = level_cal_db
    return out


def base_calibration(results: str, name: str, outputs: str) -> tuple:
    """Returns the system's output gain (dB) and delay (samples) from its
    base scores: the level calibration and the median lag on clean speech.

    Bound to the render the base scores read.
    """
    path = os.path.join(results, f"{name}.metrics.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: score {name} with scorebench"
                         " before this sidecar")
    with open(path, encoding="utf-8") as fh:
        base = json.load(fh)
    if base.get("_scorer_sha256") != scorebench.SCORER_SHA256:
        raise SystemExit(f"{path}: scored by another base scorer; re-score"
                         f" {name} with scorebench first")
    if base.get("_outputs_sha256") != outputs:
        raise SystemExit(f"{path}: scores of another render of {name};"
                         " re-score it with scorebench first")
    lag = (base.get("clean") or {}).get("lag_median") or 0.0
    return float(base.get("_level_cal_db") or 0.0), int(round(lag))


def pair_outputs_digest(records: list) -> str | None:
    """Returns the identity of the pair renders the records read, or None."""
    hashes = [
        r["identity"]["pair_output"]
        for r in records
        if r.get("output_status") == "ok" and
        (r.get("identity") or {}).get("pair_output")
    ]
    return identity_of(hashes) if hashes else None


def write(name, records, out, rows, level_cal_db, outputs, pair=None, lag=0):
    """Writes the system's records and sidecar; ``pair`` names the pair of
    renders they read (its inputs' identity and its outputs' identity),
    ``lag`` the delay undone."""
    pair = pair or {}
    observations.sidecar(
        os.path.join(common.RESULTS, f"{name}.events.items.jsonl"),
        records,
        name,
        common.set_id(),
        common.SUBSET,
        common.is_sealed(rows),
        SCORER_SHA256,
        level_cal_db=level_cal_db,
        lag=lag,
        outputs_sha256=outputs,
        pair_sha256=pair.get("inputs"),
        pair_outputs_sha256=pair.get("outputs"),
    )
    out["_set_id"] = common.set_id()
    out["_subset"] = common.SUBSET
    out.update(anchors.stamp())
    out["_sealed"] = common.is_sealed(rows)
    out["_scorer_sha256"] = SCORER_SHA256
    out["_labels_sha256"] = labels_identity(rows)
    out["_outputs_sha256"] = outputs
    out["_pair_sha256"] = pair.get("inputs")
    out["_pair_outputs_sha256"] = pair.get("outputs")
    path = os.path.join(common.RESULTS, f"{name}.events.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(observations.finite(out), fh, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pair",
        action="store_true",
        help="link the same-room references under <pair>/in (or confirm the"
        " current ones) and stop",
    )
    ap.add_argument(
        "--pair-dir",
        default="",
        help="default: BENCH_PAIR or pair_coroom beside BENCH_OUT",
    )
    ap.add_argument("--systems", default="")
    ap.add_argument("--reuse-from", default="", metavar="DIR")
    ap.add_argument("--aggregate-only", default="", metavar="DIR")
    a = ap.parse_args()

    rows = common.manifest()
    common.guard_private_results(rows)
    root = pair_dir(a.pair_dir)
    if a.pair:
        before = pair_problem(root, rows)
        pair = write_pair(rows, root)
        n = len(pair["items"])
        print(f"pair is current: {n} references under {root}/in" if before is
              None else (
                  f"{n} references -> {root}/in ({before}); renders of the"
                  " earlier inputs no longer match: render the pair again into"
                  " a fresh folder"))
        return
    selected = [r for r in rows if r["condition"] in SAME_ROOM + REAL]
    pair = (read_pair(root, rows)
            if pair_rows(selected) and not a.aggregate_only else None)
    systems = [s for s in a.systems.split(",") if s] or sorted(
        os.path.basename(d)
        for d in glob.glob(common.OUT_ROOT + "/*")
        if os.path.isdir(d))
    vad = None if a.aggregate_only else common.Vad()
    os.makedirs(common.RESULTS, exist_ok=True)
    for name in systems:
        if a.aggregate_only:
            src = os.path.join(a.aggregate_only, f"{name}.events.items.jsonl")
            ids = {r["id"] for r in selected}
            problem, source, records = scorebench.view_records(
                src, name, ids, SCORER_SHA256)
            problem = problem or stale_labels(records, selected)
            if problem:
                print(f"skip {name}: {problem}", flush=True)
                continue
            outputs = source.get("outputs_sha256")
            # the view's own calibration, from the base scores derived for it
            level_cal_db, _ = base_calibration(common.RESULTS, name, outputs)
            lag = int(source.get("lag") or 0)
            renders = {
                "inputs": source.get("pair_sha256"),
                "outputs": source.get("pair_outputs_sha256"),
            }
        else:
            problem = name != "INPUT" and scorebench.render_problem(name)
            if problem:
                print(f"skip {name}: {problem}", flush=True)
                continue
            pair_out = None
            if pair is not None and name != "INPUT":
                problem = pair_render_problem(name, common.RESULTS, root, pair)
                if problem:
                    raise SystemExit(f"{name}: {problem}")
                pair_out = os.path.join(root, "out", name)
            outputs = common.outputs_digest(name)
            level_cal_db, lag = base_calibration(common.RESULTS, name, outputs)
            own = os.path.join(common.RESULTS, f"{name}.events.items.jsonl")
            reuse, why = {}, "no earlier records"
            if a.reuse_from:
                reuse, why = scorebench.reuse_index(
                    os.path.join(a.reuse_from, f"{name}.events.items.jsonl"),
                    name,
                    SCORER_SHA256,
                )
                why = f"reuse from {a.reuse_from}: {why}"
            elif os.path.isfile(own):
                reuse, why = scorebench.reuse_index(own, name, SCORER_SHA256)
                why = f"existing records: {why}"
            print(f"  {why}", flush=True)
            records = assemble(name,
                               selected,
                               vad,
                               level_cal_db,
                               reuse=reuse,
                               pair_out=pair_out,
                               lag=lag)
            renders = {
                "inputs": pair["identity"] if pair else None,
                "outputs": pair_outputs_digest(records),
            }
        out = aggregate(records, selected, level_cal_db)
        out["_lag"] = lag
        write(name, records, out, rows, level_cal_db, outputs, renders, lag)
        block, real = out.get("_coroom", {}), out.get("_real_events", {})
        print(
            f"{name:24s} same room: events"
            f" {block.get('event_resid_db', math.nan):6.1f} dB, floor"
            f" {block.get('floor_resid_db', math.nan):6.1f} dB, voice"
            f" {block.get('voice_level_db', math.nan):5.1f} dB over"
            f" {block.get('n', 0)} items; real events"
            f" {real.get('event_att_db', math.nan):6.1f} dB on"
            f" {real.get('n', 0)} items",
            flush=True,
        )


if __name__ == "__main__":
    main()
