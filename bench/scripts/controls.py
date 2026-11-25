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
"""Frozen metric controls: known damage in, what every metric says about it out.

    python bench/scripts/controls.py [--per-condition 24]
        [--conds clean,babble@+4,...] [--controls reference,delete80ms,...]
        [--out staging/controls] [--keep-renders]
        [--markdown-out bench/CONTROLS.md]

A metric is only worth ranking systems by if it reacts to damage someone could
hear and stays quiet about changes nobody could. Each control is a synthetic
"system" built from the paired audio of a fixed, language-stratified slice of
the set: the reference returned untouched, the reference with one known defect,
or something that is not the voice at all. Every control is scored by
``scorebench.score_system``, the code every real system goes through, so the
table describes the published metrics and not a reimplementation of them.

CONTROLS names the defects, EXPECTED states per control what a sound metric
suite must say (``check`` compares; the markdown marks every row that fails and
says which metrics are blind). On the same-room block every control is also
rendered on the reference track, and the block's rows are read from the two
renders as the board reads them (``scoreevents.py``). Processing an output a
second time needs a model and lives in twicetest.py.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf
import torch

import anchors
import common

SR = common.SR
DEFAULT_CONDS = (
    "clean",
    "babble@+4",
    "babble@+0",
    "babble@-5",
    "ambience@-5",
    "stationary@-5",
    "traffic@-5",
    "music@-6",
    "wild@-5",
    "whisper@+0",
    "coroom@pause",
    "coroom@speech",
    "coroom@floor",
    "coroom@both",
    "nonverbal_clean",
)
# a deletion lands on speech: a frame within this of the loudest
ACTIVE_DB = 30.0
# the same-room block (scoreevents.SAME_ROOM) and the gate that tells its
# transient events from its quiet floor
BLOCK_CONDS = ("coroom@pause", "coroom@speech", "coroom@floor", "coroom@both")
BLOCK_COLUMNS = ("event_resid_db", "floor_resid_db", "voice_level_db")
GATE_DB = -40.0
N_FFT, HOP = 2048, 512


def _seed(item_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{item_id}".encode()).hexdigest()[:8], 16)


def _stft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x,
                      N_FFT,
                      HOP,
                      window=torch.hann_window(N_FFT),
                      return_complex=True)


def _istft(z: torch.Tensor, n: int) -> torch.Tensor:
    return torch.istft(z, N_FFT, HOP, window=torch.hann_window(N_FFT), length=n)


def delete(ref: torch.Tensor, seconds: float, item_id: str) -> torch.Tensor:
    """The reference with one span of speech removed (2 ms raised-cosine edges).

    The span starts at a seeded position among the 10 ms frames within ACTIVE_DB
    of the loudest, so it always lands on the voice and the same item always
    loses the same span.
    """
    n, span, frame = len(ref), int(seconds * SR), SR // 100
    if n <= span + 2 * frame:
        return ref.clone()
    energy = ref[:n // frame * frame].reshape(-1, frame).pow(2).mean(1)
    level = 10 * torch.log10(energy + 1e-12)
    active = torch.nonzero(level > level.max() - ACTIVE_DB).flatten()
    active = active[active * frame + span < n]
    if active.numel() == 0:
        return ref.clone()
    rng = np.random.default_rng(_seed(item_id, f"delete{seconds}"))
    start = int(active[rng.integers(len(active))]) * frame
    gain = torch.ones(n)
    edge = int(0.002 * SR)
    ramp = 0.5 * (1 + torch.cos(torch.linspace(0, torch.pi, edge)))
    gain[start:start + span] = 0.0
    gain[start:start + edge] = ramp
    gain[start + span - edge:start + span] = ramp.flip(0)
    return ref * gain


def boundaries(ref: torch.Tensor,
               pause_s: float = 0.15) -> tuple[list[int], list[int]]:
    """Returns the speech onsets and offsets of a reference, in samples.

    A 10 ms frame within ACTIVE_DB of the loudest is speech. An onset is the
    first speech frame after at least ``pause_s`` of frames that are not (the
    item's start counts as such a pause); an offset is the end of the last
    speech frame before such a pause (the item's end counts as one).
    """
    frame = SR // 100
    n = len(ref) // frame
    if n == 0:
        return [], []
    energy = ref[:n * frame].reshape(-1, frame).pow(2).mean(1)
    level = 10 * torch.log10(energy + 1e-12)
    speech = (level > level.max() - ACTIVE_DB).tolist()
    gap = max(1, int(round(pause_s * 100)))
    onsets, offsets = [], []
    quiet = gap
    for i, active in enumerate(speech):
        if active:
            if quiet >= gap:
                onsets.append(i * frame)
            quiet = 0
            continue
        quiet += 1
        if quiet == gap and speech[i - gap]:
            offsets.append((i - gap + 1) * frame)
    if speech[-1] or 0 < quiet < gap:
        last = max(i for i, active in enumerate(speech) if active)
        offsets.append((last + 1) * frame)
    return onsets, offsets


def _silence(ref: torch.Tensor, start: int, span: int) -> torch.Tensor:
    """The reference with ``span`` samples from ``start`` silenced.

    2 ms raised-cosine edges, as ``delete`` uses.
    """
    gain = torch.ones(len(ref))
    edge = int(0.002 * SR)
    ramp = 0.5 * (1 + torch.cos(torch.linspace(0, torch.pi, edge)))
    gain[start:start + span] = 0.0
    gain[start:start + edge] = ramp
    gain[start + span - edge:start + span] = ramp.flip(0)
    return ref * gain


def cut_onset(ref: torch.Tensor, seconds: float,
              item_id: str) -> tuple[torch.Tensor, int | None]:
    """The reference with the first ``seconds`` of one word start removed.

    The onset (``boundaries``) is drawn, seeded by the item, among those
    followed by at least that much signal. Returns the signal and the cut's
    first sample, None (and the reference unchanged) when no onset fits.
    """
    span = int(seconds * SR)
    onsets, _ = boundaries(ref)
    usable = [o for o in onsets if o + span <= len(ref)]
    if not usable:
        return ref.clone(), None
    rng = np.random.default_rng(_seed(item_id, f"onset{seconds}"))
    start = usable[rng.integers(len(usable))]
    return _silence(ref, start, span), start


def cut_tail(ref: torch.Tensor, seconds: float,
             item_id: str) -> tuple[torch.Tensor, int | None]:
    """The reference with the last ``seconds`` before one trailing end removed.

    The offset (``boundaries``) is drawn, seeded by the item, among those with
    at least that much signal before them. Returns the signal and the cut's
    first sample (the cut ends at the offset), None when no offset fits.
    """
    span = int(seconds * SR)
    _, offsets = boundaries(ref)
    usable = [o for o in offsets if o - span >= 0]
    if not usable:
        return ref.clone(), None
    rng = np.random.default_rng(_seed(item_id, f"tail{seconds}"))
    end = usable[rng.integers(len(usable))]
    return _silence(ref, end - span, span), end - span


def invent_high_band(ref: torch.Tensor,
                     item_id: str,
                     low_hz: float = 8000.0) -> torch.Tensor:
    """Returns the reference with an invented band above ``low_hz``.

    Everything above ``low_hz`` is replaced by noise that follows the band's own
    envelope and carries its energy: the band a generative system invents, at
    the level it was. Overlap-add of random phases loses energy, so the noise is
    scaled back to the band's.
    """
    z = _stft(ref)
    rows = torch.fft.rfftfreq(N_FFT, 1 / SR) >= low_hz
    gen = torch.Generator().manual_seed(_seed(item_id, "hf"))
    phase = (torch.rand(int(rows.sum()), z.shape[1], generator=gen) * 2 *
             torch.pi)
    noise = torch.zeros_like(z)
    noise[rows] = z[rows].abs() * torch.exp(1j * phase)
    noise = _istft(noise, len(ref))
    low = cut_high_band(ref, low_hz)
    band = ref - low
    return low + noise * torch.sqrt(
        band.pow(2).sum() / (noise.pow(2).sum() + 1e-20))


def cut_high_band(ref: torch.Tensor, low_hz: float = 8000.0) -> torch.Tensor:
    """Returns the reference with everything above ``low_hz`` removed."""
    z = _stft(ref)
    z[torch.fft.rfftfreq(N_FFT, 1 / SR) >= low_hz] = 0
    return _istft(z, len(ref))


def resynthesise(ref: torch.Tensor,
                 item_id: str,
                 sigma: float = 0.1) -> torch.Tensor:
    """Returns the reference with every phase perturbed by N(0, sigma) radians.

    The same spectrogram as another waveform (its clean SI-SDR is in the table):
    what a vocoder-like system returns when it harms nothing.
    """
    z = _stft(ref)
    gen = torch.Generator().manual_seed(_seed(item_id, "resynth"))
    z = z * torch.exp(1j * sigma * torch.randn(z.shape, generator=gen))
    return _istft(z, len(ref))


def remove_floor(ref: torch.Tensor, floor_db: float = 50.0) -> torch.Tensor:
    """Returns the reference with its own floor gated to silence.

    Every 10 ms frame more than ``floor_db`` under the loudest is silenced: the
    recording's own floor removed, the speech untouched.
    """
    frame = SR // 100
    n = len(ref) // frame * frame
    level = 10 * torch.log10(ref[:n].reshape(-1, frame).pow(2).mean(1) + 1e-12)
    keep = (level > level.max() - floor_db).float().repeat_interleave(frame)
    out = ref.clone()
    out[:n] = ref[:n] * keep
    return out


def delay(ref: torch.Tensor, seconds: float) -> torch.Tensor:
    k = int(seconds * SR)
    return torch.nn.functional.pad(ref, (k, 0))[:len(ref)]


def transients(added: torch.Tensor, gate_db: float = GATE_DB) -> torch.Tensor:
    """Returns what the mixture added above ``gate_db`` dBFS, by 10 ms frame.

    Its transient events kept, its quiet floor (and the quiet tail of every
    event) gone.
    """
    frame = SR // 100
    n = len(added) // frame * frame
    frames = added[:n].reshape(-1, frame)
    level = 10 * torch.log10(frames.pow(2).mean(1) + 1e-12)
    out = torch.zeros_like(added)
    out[:n] = (frames * (level >= gate_db).float()[:, None]).reshape(-1)
    return out


def ducked(ref: torch.Tensor,
           mix: torch.Tensor,
           db: float = 3.0) -> torch.Tensor:
    """Returns the reference ``db`` quieter whenever the mixture added anything.

    A gate that turns the voice down to be rid of the noise; on a mixture that
    added nothing (the clean items) the reference stays as it is.
    """
    if float((mix - ref).pow(2).mean()) < 10**(-100 / 10):
        return ref.clone()
    return ref * 10**(-db / 20)


# name -> (what the "system" returns, builder(ref, mix, item_id))
CONTROLS = {
    "reference": ("the clean reference, untouched", lambda r, x, i: r.clone()),
    "input": ("the unprocessed mixture", lambda r, x, i: x.clone()),
    "silence": ("nothing", lambda r, x, i: torch.zeros_like(r)),
    "bedonly": (
        "the mixture minus the reference: everything but the voice",
        lambda r, x, i: x - r,
    ),
    "gainup6db": (
        "the reference 6 dB louder",
        lambda r, x, i: r * 10**(6 / 20),
    ),
    "gaindown6db": (
        "the reference 6 dB quieter",
        lambda r, x, i: r * 10**(-6 / 20),
    ),
    "polarity": ("the reference with its sign flipped", lambda r, x, i: -r),
    "delay20ms": ("the reference 20 ms late", lambda r, x, i: delay(r, 0.020)),
    "delete20ms": (
        "the reference with 20 ms of speech removed",
        lambda r, x, i: delete(r, 0.020, i),
    ),
    "delete40ms": (
        "the reference with 40 ms of speech removed",
        lambda r, x, i: delete(r, 0.040, i),
    ),
    "delete80ms": (
        "the reference with 80 ms of speech removed",
        lambda r, x, i: delete(r, 0.080, i),
    ),
    "delete160ms": (
        "the reference with 160 ms of speech removed",
        lambda r, x, i: delete(r, 0.160, i),
    ),
    "delete320ms": (
        "the reference with 320 ms of speech removed",
        lambda r, x, i: delete(r, 0.320, i),
    ),
    "hfinvented": (
        "the reference with 8-24 kHz replaced by noise of the same energy",
        lambda r, x, i: invent_high_band(r, i),
    ),
    "hfcut": (
        "the reference with 8-24 kHz removed",
        lambda r, x, i: cut_high_band(r),
    ),
    "resynth": (
        ("the reference re-synthesised: same spectrogram, phases perturbed"
         " (another waveform)"),
        lambda r, x, i: resynthesise(r, i),
    ),
    "floorless": (
        "the reference with its own noise floor gated out",
        lambda r, x, i: remove_floor(r),
    ),
    "halfnoise": (
        "the reference with half of what the mixture added",
        lambda r, x, i: r + 0.5 * (x - r),
    ),
    "transientsleft": (
        ("the reference with what the mixture added above -40 dBFS: its"
         " transients left, its quiet floor gone"),
        lambda r, x, i: r + transients(x - r),
    ),
    "floorleft": (
        ("the reference with what the mixture added under -40 dBFS: its"
         " quiet floor left, its transients gone"),
        lambda r, x, i: x - transients(x - r),
    ),
    "voiceducked": (
        ("the reference 3 dB quieter wherever the mixture added anything: a"
         " gate rid of the noise at the voice's expense, under the voice gate"),
        lambda r, x, i: ducked(r, x),
    ),
}

# What a sound metric suite must say. Keys are columns of ``summarise``; values
# (op, bound). "fidelity" is the published fidelity axis on the clean items of
# the slice (0-1).
# the block's rows under a control that returns the voice, whatever happened
# to it on the way: nothing of the noise left, the voice's level kept
BLOCK_CLEAN = {
    "event_resid_db": ("<=", -19.9),
    "floor_resid_db": ("<=", -19.9),
    "voice_level_db": (">=", -0.1),
}
HARMLESS = {
    "fidelity": (">=", 0.95),
    "kept_coh_p10": (">=", -1.0),
    "lost_pct": ("<=", 0.5),
    "nv_lost_pct": ("<=", 0.5),
    "spk_sim": (">=", 0.99),
    **BLOCK_CLEAN,
}
GONE = {
    "fidelity": ("<=", 0.5),
    "kept_coh_p10": ("<=", -40.0),
    "lost_pct": (">=", 99.0),
    "nv_lost_pct": (">=", 99.0),
}
EXPECTED = {
    "reference": dict(HARMLESS, si_sdr=(">=", 60.0)),
    # on clean the input is the reference; on the block it leaves the noise
    # as it was, by definition
    "input": {
        "fidelity": (">=", 0.95),
        "event_resid_db": (">=", -0.05),
        "floor_resid_db": (">=", -0.05),
        "voice_level_db": ("<=", 0.05),
    },
    "silence": GONE,
    # the bed returned is no suppression
    "bedonly": dict(GONE, babble_supp_db=(">=", 0.0)),
    # a gain on both renders alike: the block's rows read the voice's level
    # with the noise against without, so none of it shows
    "gainup6db": {
        "level_err": (">=", 5.5),
        "fidelity": ("<=", 0.85),
        "lost_pct": ("<=", 0.5),
        **BLOCK_CLEAN,
    },
    "gaindown6db": {
        "level_err": (">=", 5.5),
        "fidelity": ("<=", 0.85),
        "lost_pct": ("<=", 0.5),
        **BLOCK_CLEAN,
    },
    # a sign flip is inaudible: nothing may read it as damage
    "polarity": HARMLESS,
    "delay20ms": HARMLESS,  # inside the +-100 ms the scorer undoes
    "delete20ms": {
        "nisqa_mos": ("<=", 4.35)
    },
    "delete40ms": {
        "nisqa_mos": ("<=", 4.35)
    },
    "delete80ms": {
        "nisqa_mos": ("<=", 4.35)
    },
    "delete160ms": {
        "nisqa_mos": ("<=", 4.35),
        "fidelity": ("<=", 0.95)
    },
    # retention is read per 100 ms block: only a deletion that empties whole
    # blocks must show
    "delete320ms": {
        "lost_pct": (">=", 2.0),
        "fidelity": ("<=", 0.9)
    },
    "hfinvented": {
        "si_sdr": ("<=", 25.0)
    },
    "hfcut": {
        "hf_err": (">=", 3.0),
        "fidelity": ("<=", 0.85)
    },
    # another waveform, the same sound: must not read as damage, and sets the
    # magnitude ceiling
    "resynth": dict(HARMLESS, mag_snr=(">=", 30.0)),
    "floorless": HARMLESS,
    # the block's rows: a known share of the noise left reads as that share,
    # the transients and the floor are told apart, and a voice turned down to
    # be rid of the noise reads as the voice lost
    "halfnoise": {
        "event_resid_db": ("<=", -5.9),
        "floor_resid_db": ("<=", -5.9),
        "voice_level_db": (">=", -0.1),
    },
    "transientsleft": {
        "event_resid_db": (">=", -3.0),
        "floor_resid_db": ("<=", -15.0),
    },
    "floorleft": {
        "event_resid_db": ("<=", -10.0),
        "floor_resid_db": (">=", -0.5),
    },
    # the voice turned down to the declared floor reads as the voice lost,
    # and under the gate nothing counts as removed: the noise rows read as
    # left
    "voiceducked": {
        "voice_level_db": ("<=", -2.9),
        "floor_resid_db": (">=", -0.1),
        "event_resid_db": (">=", -0.1),
    },
}
# Damage the published metrics do not see, stated rather than hidden.
BLIND = {
    "delete20ms": (
        "retention is read per 100 ms block and as a 10th percentile: a dropped"
        " 20-80 ms (a consonant, a syllable onset) moves neither; only NISQA,"
        " waveform SI-SDR and the magnitude ratio react, the last from 40 ms on"
        " (a 20 ms hole stays above its ceiling). A recogniser's transcript"
        " does not react either: Whisper large-v3 restores the word from"
        " context for holes up to 160 ms (scorecontent.py's ctl_ rows), so the"
        " content check sees a dropout only from about 320 ms"),
    "delete40ms":
        "as delete20ms",
    "delete80ms":
        ("as delete20ms; `lost_pct` starts to see it when the span empties a"
         " block"),
    "hfinvented":
        ("a top octave replaced by noise of the same energy keeps every band"
         " level, the speaker embedding and NISQA; waveform SI-SDR reacts but"
         " cannot tell this from a harmless re-synthesis (`resynth`), and the"
         " magnitude ratio moves little, because the top octave holds little of"
         " the energy. Whether listeners hear it is an open listening question"
        ),
}
COLUMNS = (
    "si_sdr",
    "kept_coh_p10",
    "lost_pct",
    "level_err",
    "hf_err",
    "spk_sim",
    "nisqa_mos",
    "mag_snr",
    "nv_lost_pct",
    "fidelity",
    "babble_supp_db",
) + BLOCK_COLUMNS


def stratified(rows: list[dict], conds: tuple[str, ...],
               per_condition: int) -> list[dict]:
    """Returns ``per_condition`` paired rows of each condition.

    They are dealt round-robin over languages in id order, so every language of
    a condition is heard before any is heard twice.
    """
    out = []
    for cond in conds:
        by_lang = {}
        for r in sorted(
            (r for r in rows if r["condition"] == cond and r.get("clean_path")),
                key=lambda r: r["id"],
        ):
            by_lang.setdefault(r.get("language") or "", []).append(r)
        picked, depth = [], 0
        while len(picked) < per_condition and any(
                len(v) > depth for v in by_lang.values()):
            picked += [
                v[depth] for _, v in sorted(by_lang.items()) if len(v) > depth
            ]
            depth += 1
        out += picked[:per_condition]
    return out


def pair_dir(root: str, name: str) -> str:
    """Where a control's renders of the block's reference tracks go."""
    return os.path.join(root, "pair", name)


def render(name: str, rows: list[dict], root: str) -> None:
    """Writes one control's outputs for the slice as 48 kHz float WAV files.

    On the same-room block the control is rendered on the reference track as
    well, under ``pair_dir``, as every system is for the block's rows.
    """
    outd, paird = os.path.join(root, name), pair_dir(root, name)
    os.makedirs(outd, exist_ok=True)
    build = CONTROLS[name][1]
    for r in rows:
        ref, mix = common.load48(r["clean_path"]), common.load48(r["mix_path"])
        n = min(len(ref), len(mix))
        out = torch.zeros(len(mix))
        out[:n] = build(ref[:n], mix[:n], r["id"])[:n]
        sf.write(os.path.join(outd, r["id"]), out.numpy(), SR, subtype="FLOAT")
        if r["condition"] in BLOCK_CONDS:
            os.makedirs(paird, exist_ok=True)
            out = torch.zeros(len(ref))
            out[:n] = build(ref[:n], ref[:n], r["id"])[:n]
            sf.write(os.path.join(paird, r["id"]),
                     out.numpy(),
                     SR,
                     subtype="FLOAT")


def summarise(metrics: dict, calibration: dict, axes: dict) -> dict:
    """Returns one row of the table from a control's scores.

    The kept sidecar's scores are merged in: clean-item metrics, the fidelity
    axis, and babble suppression at +4 dB.
    """
    clean = dict(metrics.get("clean") or {})
    row = {
        k: clean.get(k, float("nan"))
        for k in ("si_sdr", "spk_sim", "nisqa_mos", "mag_snr")
    }
    # the board removes a system's global level offset before reading retention;
    # so does this
    row["kept_coh_p10"] = clean.get("kept_coh_p10", float("nan")) - metrics.get(
        "_level_cal_db", 0.0)
    row["lost_pct"] = clean.get("kept_coh_lost_pct", float("nan"))
    row["level_err"] = clean.get("level_err_db", float("nan"))
    row["hf_err"] = clean.get("hf_err_db", float("nan"))
    nonverbal = metrics.get("nonverbal_clean") or {}
    row["nv_lost_pct"] = nonverbal.get("nv_lost_pct", float("nan"))
    values = {
        cond: dict(cell)
        for cond, cell in metrics.items()
        if isinstance(cell, dict)
    }
    values["clean"] = dict(
        clean,
        kept_coh_p10=row["kept_coh_p10"],
        _level_err=row["level_err"],
        _hf_err=row["hf_err"],
    )
    subs = [
        anchors.normalize(
            float(values.get(cond, {}).get(metric, float("nan"))),
            calibration["points"][anchors.row_key(cond, metric)],
        ) for cond, metric, _ in axes["fidelity"]
    ]
    # a part that is missing (NaN) leaves the axis missing
    row["fidelity"] = float(np.mean(subs))
    babble = metrics.get("babble@+4") or {}
    row["babble_supp_db"] = babble.get("babble_supp_db", float("nan"))
    block = metrics.get("_coroom") or {}
    for key in BLOCK_COLUMNS:
        row[key] = block.get(key, float("nan"))
    return row


def kept(name: str, rows: list[dict], vad, level_cal_db: float) -> dict:
    """Returns a control's kept-sidecar cells, to merge into its scores."""
    # pylint: disable-next=import-outside-toplevel
    import scorekept

    selected = [r for r in rows if scorekept.wanted(r)]
    cells = scorekept.aggregate(scorekept.assemble(name, selected, vad=vad),
                                selected, level_cal_db)
    return {
        cond: {
            k: v for k, v in cell.items() if k.startswith(("nv_", "mag_"))
        } for cond, cell in cells.items()
    }


def events(name: str, rows: list[dict], level_cal_db: float, lag: int,
           paird: str) -> dict:
    """Returns a control's events-sidecar cells, to merge into its scores.

    Read over the slice's same-room items from the control's two renders, the
    system's delay and gain undone as the board does.
    """
    # pylint: disable-next=import-outside-toplevel
    import scoreevents

    selected = [r for r in rows if r["condition"] in scoreevents.SAME_ROOM]
    if not selected:
        return {}
    records = scoreevents.assemble(name,
                                   selected,
                                   None,
                                   level_cal_db,
                                   pair_out=paird,
                                   lag=lag)
    cells = scoreevents.aggregate(records, selected, level_cal_db)
    return {
        cond: {
            k: v
            for k, v in cell.items()
            if k.startswith(("event_", "floor_resid", "voice_level"))
        } for cond, cell in cells.items() if isinstance(cell, dict)
    }


def check(table: dict, block: bool = True) -> dict:
    """Returns {control: [failed expectation, ...]} over ``table``.

    For every control with a declared row; the block's columns are held only
    when the slice holds the block (``block``).
    """
    failures = {}
    for name, row in table.items():
        bad = []
        for column, (op, bound) in EXPECTED.get(name, {}).items():
            if column in BLOCK_COLUMNS and not block:
                continue
            value = row.get(column, float("nan"))
            ok = value == value and (value >= bound
                                     if op == ">=" else value <= bound)
            if not ok:
                bad.append(f"{column} {value:.2f} should be {op} {bound:g}")
        if bad:
            failures[name] = bad
    return failures


def markdown(
    table: dict,
    failures: dict,
    set_id: str,
    n_items: int,
    conds: tuple[str, ...],
) -> str:
    """Returns the controls table, each row with its expectation and verdict."""

    def fmt(v, nd):
        return "-" if v != v else f"{v:.{nd}f}"

    digits = {
        "spk_sim": 3,
        "nisqa_mos": 2,
        "fidelity": 2,
        "hf_err": 2,
        "level_err": 2,
    }
    lines = [
        "# Metric controls",
        "",
        ("Known damage in, what the published metrics say about it out."
         " Every row is a synthetic system built from the paired audio of a"
         f" fixed, language-stratified slice of the set ({n_items} items"
         f" over {len(conds)} conditions; set `{set_id[:16]}`) and scored by"
         " the same code as a real system. Columns are read on the `clean`"
         " items of the slice, where the reference is the input and a"
         " defect is the only thing that happened; `lost_pct` is the share"
         " of speech blocks kept below -10 dB, `mag_snr` the"
         " signal-to-error ratio of the magnitude spectrum, `nv_lost_pct`"
         " the share of vocal sounds outside speech lost on the clean"
         " recordings of laughs, breaths and sighs, `fidelity` the"
         " published axis, and babble suppression is read at +4 dB. A delay"
         " of up to 100 ms, a flipped polarity and a gain of up to 12 dB"
         " are conventions the scorer undoes, not damage. The last three"
         " columns are the same-room block's rows, read on its items from"
         " two renders of the control, of the mixture and of the reference,"
         " as the board reads every system (`scoreevents.py`): the noise"
         " left over the labelled events and pauses, in dB, and the voice's"
         " level with the noise against without; a slice without the block"
         " leaves them blank. Generated by"
         " `scripts/controls.py`; the expectations live in its `EXPECTED`"
         " table."),
        "",
        "| control | the system returns | " +
        " | ".join(f"`{c}`" for c in COLUMNS) + " | held |",
        "|---|---|" + "---:|" * len(COLUMNS) + "---|",
    ]
    for name, row in table.items():
        held = ("no: " + "; ".join(failures[name]) if name in failures else
                ("yes" if name in EXPECTED else ""))
        lines.append(f"| `{name}` | {CONTROLS[name][0]} | " + " | ".join(
            fmt(row[c], digits.get(c, 1)) for c in COLUMNS) + f" | {held} |")
    lines += ["", "## Blind spots", ""
             ] + [f"- `{name}`: {why}." for name, why in BLIND.items()]
    lines += [
        "",
        "Expectations: " +
        "; ".join(f"`{name}` " + ", ".join(f"{c} {op} {b:g}"
                                           for c, (op, b) in exp.items())
                  for name, exp in EXPECTED.items()) + ".",
    ]
    return "\n".join(lines) + "\n"


def identity(rows: list[dict]) -> dict:
    """Returns what a control's saved scores were made by and of.

    The scorers' digests, this module (the defects it builds), the anchor policy
    and the exact slice of items. A cache that differs in any of them is stale,
    however alike its numbers look.
    """
    # pylint: disable-next=import-outside-toplevel
    import observations
    # pylint: disable-next=import-outside-toplevel
    import scorebench
    # pylint: disable-next=import-outside-toplevel
    import scoreevents
    # pylint: disable-next=import-outside-toplevel
    import scorekept

    return {
        "set_id":
            common.set_id(),
        "scorer_sha256":
            scorebench.SCORER_SHA256,
        "kept_sha256":
            scorekept.SCORER_SHA256,
        "events_sha256":
            scoreevents.SCORER_SHA256,
        "controls_sha256":
            observations.implementation(__file__),
        "items":
            anchors.digest(sorted(r["id"] for r in rows)),
        # the block's labels: a corrected event span or room allowance makes
        # every saved control stale
        "labels":
            anchors.digest(
                sorted(
                    [r["id"], scoreevents.labels_digest(r) or ""] for r in rows)
            ),
        **anchors.stamp(),
    }


def cached(metrics: dict, rows: list[dict]) -> bool:
    """Returns whether a control's saved scores are current.

    They must stand for this set, slice, scorer and anchor policy.
    """
    return metrics.get("_controls_identity") == identity(rows)


def main() -> None:
    """Renders, scores and tabulates the controls.

    Renders are deleted once scored.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--conds", default=",".join(DEFAULT_CONDS))
    ap.add_argument("--per-condition", type=int, default=24)
    ap.add_argument("--controls", default=",".join(CONTROLS))
    ap.add_argument("--out",
                    default=os.path.join(common.REPO_DIR, "staging",
                                         "controls"))
    ap.add_argument("--keep-renders", action="store_true")
    ap.add_argument("--markdown-out", default="")
    a = ap.parse_args()
    # pylint: disable-next=import-outside-toplevel
    import ranking
    # pylint: disable-next=import-outside-toplevel
    import scorebench

    names = [n for n in a.controls.split(",") if n]
    unknown = sorted(set(names) - set(CONTROLS))
    if unknown:
        raise SystemExit(f"unknown controls: {unknown}")
    conds = tuple(c for c in a.conds.split(",") if c)
    rows = stratified(common.manifest(), conds, a.per_condition)
    if not any(r["condition"] == "clean" for r in rows):
        raise SystemExit(
            "the slice needs the clean condition: the table is read on it")
    calibration = anchors.load(os.path.join(common.RESULTS, anchors.RESOLVED),
                               anchors.definitions())
    os.makedirs(a.out, exist_ok=True)
    renders = os.path.join(a.out, "renders")
    common.OUT_ROOT = renders  # score_system reads a system's outputs from here
    set_tag = (common.set_id() or "noset")[:16]
    spk_cache = os.path.join(common.EVAL, ".cache", "spkemb", set_tag)
    spk = common.SpeakerEmbedder(cache_dir=spk_cache)
    vad = common.Vad()
    table = {}
    for name in names:
        path = os.path.join(a.out, f"{name}.metrics.json")
        metrics = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                metrics = json.load(fh)
        if metrics is None or not cached(metrics, rows):
            render(name, rows, renders)
            metrics = scorebench.score_system(name, rows, spk, vad, None)
            level_cal = metrics.get("_level_cal_db", 0.0)
            for cond, cell in kept(name, rows, vad, level_cal).items():
                metrics.setdefault(cond, {}).update(cell)
            lag = int(round((metrics.get("clean") or {}).get("lag_median") or
                            0))
            for cond, cell in events(name, rows, level_cal, lag,
                                     pair_dir(renders, name)).items():
                metrics.setdefault(cond, {}).update(cell)
            metrics["_controls_identity"] = identity(rows)
            anchors.atomic_json(path, metrics)
            if not a.keep_renders:
                shutil.rmtree(os.path.join(renders, name), ignore_errors=True)
                shutil.rmtree(pair_dir(renders, name), ignore_errors=True)
        table[name] = summarise(metrics, calibration, ranking.AXES_FULL)
        print(
            f"{name:12s} " +
            "  ".join(f"{c} {table[name][c]:.2f}" for c in COLUMNS),
            flush=True,
        )
    failures = check(table, any(r["condition"] in BLOCK_CONDS for r in rows))
    md = markdown(table, failures, common.set_id() or "", len(rows), conds)
    anchors.atomic_json(
        os.path.join(a.out, "controls.json"),
        {
            "set_id": common.set_id(),
            "n_items": len(rows),
            "conditions": conds,
            "table": table,
            "failures": failures,
        },
    )
    if a.markdown_out:
        with open(a.markdown_out, "w", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    print(
        f"{len(failures)} of {len(table)} controls miss their expectation",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
