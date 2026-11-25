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
"""Shared core for the benchmark: paths, 48 kHz audio IO, mixing, metrics.

Every scorer imports this module. Paths come from environment variables so
the same scripts run on any machine:

  VOICE_ISOLATION_EVALS  eval-set root: mix/ ref/ manifest.jsonl
                         (default: <repo>/data/voice-isolation-evals)
  BENCH_OUT              per-system outputs: <BENCH_OUT>/<system>/<id>.wav
                         (default: <repo>/data/outputs)
  BENCH_RESULTS          score JSONs (default: <repo>/bench/results)

The metric primitives are defined here once, so every scorer and sidecar reads
the same implementation.
"""

import json
import os
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio

from inworld import voice_isolation

SR = 48000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BENCH_DIR)
EVAL = os.environ.get(
    "VOICE_ISOLATION_EVALS",
    os.path.join(REPO_DIR, "data", "voice-isolation-evals"),
)
OUT_ROOT = os.environ.get(
    "BENCH_OUT",
    os.path.join(REPO_DIR, "data", "outputs"),
)
RESULTS = os.environ.get("BENCH_RESULTS", os.path.join(BENCH_DIR, "results"))
BANDS_HZ = (
    (0, 500),
    (500, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, 12000),
    (12000, 16000),
    (16000, 24000),
)
NB_RATE = 16000  # the 0-8 kHz band that 16 kHz-native systems can emit
MAX_LAG = int(0.100 * SR)  # +-100 ms search window for algorithmic delay
# a level offset beyond this is not a gain convention but another output
MAX_LEVEL_CAL_DB = 12.0


def set_id(root=None):
    """The eval set's content fingerprint from its checksums.json, or None."""
    p = os.path.join(root or EVAL, "checksums.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh).get("set_id")


def outputs_digest(name: str, results: str | None = None) -> str | None:
    """Returns the digest of a system's stamped outputs, or None.

    The ``outputs_sha256`` that stamprender.py writes; None when the stamp
    records none: the set's own mixtures and references, linked rather than
    rendered.
    """
    path = os.path.join(results or RESULTS, f"{name}.meta.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("outputs_sha256")


# The set has two halves. The core is the multilingual draw: for every condition
# the first CORE_INDEX_LIMIT indices of its ladder (real recordings are core
# throughout). The extension is English only: the indices added above that limit
# and the condition families that exist only in English (matched on the
# condition name, so nonverbal_clean belongs with nonverbal@+0). Scores are
# reported for both, so a language-heavy extension cannot move the multilingual
# result.
CORE_INDEX_LIMIT = 48
EXTENSION_FAMILIES = frozenset((
    "animals",
    "appliance",
    "household",
    "nonverbal",
    "rain",
    "soft",
    "stationary",
    "tools",
    "traffic",
    "water",
    "whisper",
))
# frozen development panels, drawn by panels.py
PANELS = ("smoke", "gate", "exact")
SUBSETS = ("all", "core", "extension") + PANELS
SUBSET = os.environ.get("BENCH_SUBSET", "all")
if SUBSET not in SUBSETS:
    raise SystemExit(f"BENCH_SUBSET must be one of {SUBSETS}, not {SUBSET!r}")


def item_index(row):
    """The ladder index of a row: the number before .wav in its id, or -1."""
    m = re.search(r"_(\d+)\.wav$", row["id"])
    return int(m.group(1)) if m else -1


def subset(row):
    """'core' or 'extension' for a manifest row (see CORE_INDEX_LIMIT)."""
    # the condition's family
    if re.split(r"[@_]", row["condition"])[0] in EXTENSION_FAMILIES:
        return "extension"
    if (row.get("clean_path") and row.get("language") == "en" and
            item_index(row) >= CORE_INDEX_LIMIT):
        return "extension"
    return "core"


def manifest(root=None, which=None):
    """Manifest rows with mix/clean paths made absolute under ``root``.

    ``which`` selects 'all', 'core' or 'extension' (default: the BENCH_SUBSET
    environment variable, 'all'), so every scorer sees the same items for a
    given subset.
    """
    root = root or EVAL
    which = which or SUBSET
    panel = panel_members(which, root) if which in PANELS else None
    rows = []
    with open(os.path.join(root, "manifest.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if panel is not None:
                if r["id"] not in panel:
                    continue
            elif which != "all" and subset(r) != which:
                continue
            for k in ("mix_path", "clean_path"):
                if r.get(k) and not os.path.isabs(r[k]):
                    r[k] = os.path.join(root, r[k])
            rows.append(r)
    return rows


def file_hashes(root=None):
    """Returns a function from a path to the sha256 of a file of the set.

    It reads checksums.json (the builder hashes every mixture and reference) and
    computes the hash of anything the file does not list.
    """
    # pylint: disable-next=import-outside-toplevel
    import hashlib

    root = root or EVAL
    path = os.path.join(root, "checksums.json")
    listed = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            listed = json.load(fh).get("files", {}) or {}
    computed = {}

    def digest(file_path):
        rel = os.path.relpath(file_path, root)
        if rel in listed:
            return listed[rel]
        if file_path not in computed:
            with open(file_path, "rb") as fh:
                computed[file_path] = hashlib.sha256(fh.read()).hexdigest()
        return computed[file_path]

    return digest


def panel_members(name: str, root: str | None = None) -> set:
    """Returns the item ids of a frozen panel (``bench/panels.json``).

    A panel drawn on another build of the set is refused.
    """
    path = os.path.join(BENCH_DIR, "panels.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no {path}: draw the panels with bench/scripts/panels.py")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    current = set_id(root)
    if current and doc.get("set_id") != current:
        raise SystemExit(
            f"{path} was drawn on set {str(doc.get('set_id'))[:8]}, not "
            f"{current[:8]}: rebuild it with bench/scripts/panels.py")
    if name not in doc.get("panels", {}):
        raise SystemExit(f"no panel {name!r} in {path}")
    return set(doc["panels"][name]["ids"])


def is_sealed(rows: list[dict], root: str | None = None) -> bool:
    """Checks both private-data markers, an inventory's receipt included."""
    if any(r.get("internal_only") for r in rows):
        return True
    path = os.path.join(root or EVAL, "checksums.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            return json.load(stream).get("sealed") is True
    return False


def guard_private_results(rows: list[dict]) -> None:
    """Keeps internal-only audio and its scores out of published results."""
    if not is_sealed(rows):
        return
    # pylint: disable-next=import-outside-toplevel
    import pathlib

    public = [(pathlib.Path(REPO_DIR) / p).resolve()
              for p in ("bench/results", "docs", "media")]
    for target in (pathlib.Path(RESULTS).resolve(),
                   pathlib.Path(OUT_ROOT).resolve()):
        if any(target == p or p in target.parents for p in public):
            raise ValueError(
                "internal-only evaluation material requires private"
                " scratch/results")


def load48(path, secs=None):
    """Any audio file -> mono float32 Tensor at 48 kHz; pad/trim to ``secs``."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    t = torch.from_numpy(x.mean(1))
    if sr != SR:
        t = voice_isolation.resample(t, sr, SR)
    if secs is not None:
        n = int(secs * SR)
        t = (t[:n] if len(t) >= n else torch.nn.functional.pad(
            t, (0, n - len(t))))
    return t


def nb8k(x):
    """48k -> 16k -> 48k round-trip: keeps only the 0-8 kHz band."""
    low = torchaudio.functional.resample(x, SR, NB_RATE)
    return torchaudio.functional.resample(low, NB_RATE, SR)[:len(x)]


def best_alignment(out, ref, max_lag=MAX_LAG):
    """Returns the (lag, sign) that line ``out`` up with ``ref``.

    The delay and the polarity a scorer undoes. The lag (samples) maximises
    xcorr(out shifted by -lag, ref). A delay of up to ``max_lag`` and a flipped
    sign are both inaudible, so neither may read as damage. The sign is -1 when
    the negative peak is the larger one, as it is for a voice upside down and
    never for an upright one (no lobe of a correlation exceeds its main peak),
    and carries at least a fifth of the correlation two identical signals would
    have, so an output that holds none of the voice is not flipped on a coin
    toss. (0, 1) if aligned or (near) silent.
    """
    n = min(len(out), len(ref))
    a, b = out[:n] - out[:n].mean(), ref[:n] - ref[:n].mean()
    if float(a.abs().max()) < 1e-8 or float(b.abs().max()) < 1e-8:
        return 0, 1
    # a clip cannot be shifted by more than its length: with a larger window
    # the two halves of the correlation would overlap and read a lag of
    # -max_lag for an aligned clip
    max_lag = min(max_lag, n - 1)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    # a 2^19-point FFT per clip: milliseconds on a GPU, 0.1 s on a CPU
    a, b = a.to(DEV), b.to(DEV)
    spec_out = torch.fft.rfft(a, size)
    spec_ref = torch.fft.rfft(b, size)
    cc = torch.fft.irfft(spec_out * spec_ref.conj(), size)
    cc = torch.cat([cc[-max_lag:], cc[:max_lag + 1]])
    low, high = float(cc.min()), float(cc.max())
    if -low > high and -low >= 0.2 * float(a.norm() * b.norm()):
        return int(cc.argmin()) - max_lag, -1
    return int(cc.argmax()) - max_lag, 1


def best_lag(out, ref, max_lag=MAX_LAG):
    """Returns the lag of ``best_alignment`` alone.

    For callers that compare levels and not waveforms.
    """
    return best_alignment(out, ref, max_lag)[0]


def apply_lag(out, lag, n):
    """Shifts ``out`` back by ``lag`` samples and pads/trims it to ``n``."""
    if lag > 0:
        out = out[lag:]
    elif lag < 0:
        out = torch.nn.functional.pad(out, (-lag, 0))
    return (out[:n] if len(out) >= n else torch.nn.functional.pad(
        out, (0, n - len(out))))


def effective_bandwidth_hz(x: torch.Tensor,
                           sr: int,
                           floor_db: float = -50.0) -> float:
    """Returns the highest frequency the speech spectrum reaches.

    The highest frequency below which the active-frame spectrum stays within
    ``floor_db`` of its 1-4 kHz level. A clip upsampled from 24 kHz reads about
    12 kHz here, whatever its container rate says.
    """
    n_fft = 2048
    spec = (torch.stft(
        x.float(),
        n_fft,
        n_fft // 4,
        window=torch.hann_window(n_fft),
        return_complex=True,
    ).abs().pow(2))
    frame_energy = spec.sum(0)
    active = frame_energy > frame_energy.median()
    if not bool(active.any()):
        return 0.0
    ps = spec[:, active].mean(1)
    freqs = torch.linspace(0, sr / 2, spec.shape[0])
    ref = ps[(freqs >= 1000) & (freqs < 4000)].mean()
    db = 10 * torch.log10(ps / (ref + 1e-12) + 1e-12)
    start = int(4000 / (sr / 2) * len(freqs))
    for i in range(start, len(freqs)):
        if bool((db[i:] < floor_db).all()):
            return float(freqs[i])
    return float(sr / 2)


def mix_at_snr(clean, noise, snr_db):
    """clean + g*noise at the requested whole-clip SNR; returns (mix, clean)."""
    ln = min(len(clean), len(noise))
    c, n = clean[:ln], noise[:ln]
    g = ((float(c.std()) + 1e-8) / (float(n.std()) + 1e-8) /
         (10**(snr_db / 20)))
    return c + g * n, c


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = (est * ref).sum() / (ref.pow(2).sum() + 1e-9)
    proj = a * ref
    return float(10 * torch.log10(
        proj.pow(2).sum() / ((est - proj).pow(2).sum() + 1e-9) + 1e-9))


def voice_kept_db(out, clean, frame=960):
    """Output-vs-clean energy (dB) inside speech-active frames of CLEAN."""
    n = min(len(out), len(clean))
    out, clean = out[:n], clean[:n]
    ce = clean[:n // frame * frame].reshape(-1, frame).pow(2).mean(1)
    oe = out[:n // frame * frame].reshape(-1, frame).pow(2).mean(1)
    act = ce > ce.median()
    return float(10 * torch.log10(
        (oe[act].mean() + 1e-9) / (ce[act].mean() + 1e-9)))


def block_mask(active16, n_blocks, frame16=1600):
    """Returns a 16 kHz speech-sample mask as a mask of 100 ms blocks.

    ``n_blocks`` long: a block counts as speech when more than half of its
    samples are marked.
    """
    m = torch.zeros(n_blocks, dtype=torch.bool)
    for b in range(n_blocks):
        seg = active16[b * frame16:(b + 1) * frame16]
        if len(seg) and float(seg.float().mean()) > 0.5:
            m[b] = True
    return m


def coherent_blocks(out, ref, frame=4800, floor_db=40.0):
    """Returns OUT's voice kept, residual and measurable blocks against REF.

    Three tensors of 100 ms block values at 48 kHz. ``kept`` is 20 log10 of the
    projection coefficient of the block of OUT on the block of REF: the part of
    the output that is the voice (a coefficient <= 0 reads as the -80 dB floor).
    ``resid`` is what remains once that projection is removed, relative to the
    reference block: the bed and the artifacts, whatever is not the voice.
    ``loud`` marks the blocks within ``floor_db`` of the loudest reference
    block; below that nothing measurable is there to keep. coherent_kept_db and
    the seam, re-pass and transition checks all read these.
    """
    n = min(len(out), len(ref))
    o = out[:n // frame * frame].reshape(-1, frame)
    r = ref[:n // frame * frame].reshape(-1, frame)
    if len(r) == 0:
        empty = torch.zeros(0)
        return empty, empty.clone(), torch.zeros(0, dtype=torch.bool)
    r = r - r.mean(1, keepdim=True)
    o = o - o.mean(1, keepdim=True)
    re_ = r.pow(2).mean(1)
    loud = (10 * torch.log10(re_ + 1e-12)
            > 10 * torch.log10(re_.max() + 1e-12) - floor_db)
    # a silent reference: nothing is measurable anywhere
    if float(re_.max()) < 1e-10:
        loud[:] = False
    # per-block projection of out on ref
    a = (o * r).sum(1) / (r.pow(2).sum(1) + 1e-9)
    kept = (20 * torch.log10(a.clamp(min=1e-4))).clamp(-80.0, 20.0)
    resid = (o - a.unsqueeze(1) * r).pow(2).mean(1)
    resid_db = (10 * torch.log10(resid / (re_ + 1e-12) + 1e-8)).clamp(
        -80.0, 60.0)
    return kept, resid_db, loud


def coherent_kept_db(out, ref, frame=4800, act=None, floor_db=40.0):
    """Returns the voice kept per 100 ms block, in dB relative to REF.

    The per-block energy of the part of OUT that is coherent with REF, relative
    to REF, over the speech blocks of REF, as a tensor of block values at 48
    kHz.

    An energy ratio cannot tell kept speech from an added bed: returning the bed
    alone scores above 0 dB on voice_kept_db. Projecting the output onto the
    reference does: the bed is uncorrelated with the voice and projects to
    nothing, so a deleted foreground reads as a large negative number whatever
    else the output holds. ``act`` names the speech blocks independently (a
    voice-activity mask, see block_mask), so quiet speech counts; without it,
    every block within ``floor_db`` of the loudest counts. Blocks more than
    ``floor_db`` under the loudest are skipped either way: nothing measurable is
    there to keep.
    """
    kept, _, loud = coherent_blocks(out, ref, frame, floor_db)
    use = loud if act is None else (act[:len(loud)] & loud)
    return kept[use]


def band_kept_db(out, ref, bands=BANDS_HZ, n_fft=2048):
    """Per-band out/ref energy ratio (dB) over speech-active frames of REF.

    ~0 dB = band preserved, very negative = dulled/lost, strongly positive =
    invented high frequency.
    """
    n = min(len(out), len(ref))
    o, r = out[:n], ref[:n]
    win = torch.hann_window(n_fft)
    hop = n_fft // 4
    spec_o = torch.stft(o, n_fft, hop, window=win, return_complex=True).abs()
    spec_r = torch.stft(r, n_fft, hop, window=win, return_complex=True).abs()
    fe = spec_r.pow(2).sum(0)
    act = fe > fe.median()
    if not bool(act.any()):
        return [float("nan")] * len(bands)
    freqs = torch.linspace(0, SR / 2, spec_o.shape[0])
    out_db = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if not bool(m.any()):
            out_db.append(float("nan"))
            continue
        oe = spec_o[m][:, act].pow(2).mean()
        rr = spec_r[m][:, act].pow(2).mean()
        out_db.append(float(10 * torch.log10((oe + 1e-12) / (rr + 1e-12))))
    return out_db


def tercile_kept_db(out, ref, frame=960, act_db=40.0, act=None):
    """Returns out/ref energy (dB) per loudness tercile of REF's active frames.

    The terciles are [quiet, mid, loud].
    """
    n = min(len(out), len(ref))
    re_ = ref[:n // frame * frame].reshape(-1, frame).pow(2).mean(1)
    oe = out[:n // frame * frame].reshape(-1, frame).pow(2).mean(1)
    # a scored region shorter than one frame carries no terciles
    if len(re_) == 0:
        return [float("nan")] * 3
    if act is None:
        fdb = 10 * torch.log10(re_ + 1e-12)
        act = fdb > fdb.max() - act_db
    else:
        act = act[:len(re_)]
    idx = act.nonzero(as_tuple=True)[0]
    if len(idx) < 9:
        return [float("nan")] * 3
    idx = idx[re_[idx].argsort()]
    k = len(idx)
    return [
        float(10 * torch.log10((oe[t].mean() + 1e-9) / (re_[t].mean() + 1e-9)))
        for t in (idx[:k // 3], idx[k // 3:2 * k // 3], idx[2 * k // 3:])
    ]


def add_model_args(ap):
    """Adds the model flags every check that runs the isolator shares.

    The same flags as runsystem.py's.
    """
    ap.add_argument(
        "--model",
        default="voice-isolator-v0",
        help="registry name or checkpoint path (.safetensors/.pt)",
    )
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--mode",
                    default="offline",
                    choices=("offline", "streaming"))
    ap.add_argument("--lookahead",
                    type=int,
                    default=0,
                    help="streaming look-ahead in frames")
    ap.add_argument("--precision",
                    default="fp32",
                    choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--device", default=None)


def isolator_runner(a):
    """Returns (isolator, run) for the model the command line names.

    ``run(x48, chunk_seconds, overlap_seconds)`` returns the 48 kHz output
    through the public path at the command line's mode, look-ahead and
    precision (chunking and resampling included). The streaming window
    knobs and the batching are runsystem.py's alone: an operational check
    reads the plain stream.
    """
    vi = voice_isolation.VoiceIsolator.load(
        a.model,
        stage2=a.stage2,
        device=a.device or DEV,
        precision=a.precision,
        mode=a.mode,
        lookahead_frames=a.lookahead,
    )

    def run(x, chunk_seconds=None, overlap_seconds=None):
        kw = dict(
            output_sample_rate=SR,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        if a.alpha is None:
            return vi.isolate(x, SR, **kw)
        return vi.enhance(x, SR, alpha=a.alpha, **kw)

    return vi, run


def model_tag(model):
    """Returns a file-name tag for a model argument.

    The registry name, or the checkpoint's stem.
    """
    base = os.path.basename(str(model).rstrip("/"))
    return re.sub(r"\.(safetensors|pt)$", "", base) or "model"


def ops_path(model, check, out=None):
    """Returns where an operational check writes its results.

    ``out`` when given, else ``staging/ops/<tag>.<check>.json`` under the
    repository; never bench/results, which holds the published scores only.
    """
    if out:
        return out
    return os.path.join(REPO_DIR, "staging", "ops",
                        f"{model_tag(model)}.{check}.json")


def pick_rows(rows, conds, limit=0, paired=None):
    """Returns the manifest rows of the named conditions.

    ``conds`` holds comma-separated exact names or one prefix, and every row is
    taken when it is empty; the first ``limit`` rows are kept, and ``paired``
    True keeps the rows with a reference, False those without.
    """
    want = [c for c in conds.split(",") if c]
    if not want:
        out = list(rows)
    elif len(want) == 1 and not any(r["condition"] == want[0] for r in rows):
        out = [r for r in rows if r["condition"].startswith(want[0])]
    else:
        out = [r for r in rows if r["condition"] in want]
    if paired is not None:
        out = [r for r in out if bool(r.get("clean_path")) == paired]
    return out[:limit] if limit else out


_NISQA = None


def nisqa(wav48, sr=None):
    """Returns NISQA v2's five scores of a clip (torchmetrics).

    (mos, noisiness, discontinuity, coloration, loudness).

    Returns NaNs rather than raising: a quality probe must never take down a
    run.
    """
    global _NISQA
    try:
        if _NISQA is None:
            # pylint: disable-next=import-outside-toplevel
            from torchmetrics.functional.audio import nisqa as nisqa_metric

            _NISQA = nisqa_metric.non_intrusive_speech_quality_assessment
        x = wav48 if isinstance(wav48, torch.Tensor) else torch.as_tensor(wav48)
        x = x.detach().float().cpu()
        if x.dim() > 1:
            x = x.mean(0)
        out = _NISQA(x.unsqueeze(0), sr or SR).reshape(-1)
        return tuple(float(v) for v in out[:5])
    # a clip NISQA cannot read is a missing value, never a crash of the pass
    except Exception:  # pylint: disable=broad-exception-caught
        return tuple([float(np.nan)] * 5)


class SpeakerEmbedder:
    """Frozen WavLM-SV embedding; cosine(out, clean) = timbre identity.

    ``cache_dir`` keeps embeddings of signals that never change between systems
    (the references) on disk, keyed by the caller's ``key`` under a folder of
    the model and its revision, so another embedding model never reads them:
    every system after the first then pays for its own outputs only.
    """

    def __init__(self,
                 model_name="microsoft/wavlm-base-plus-sv",
                 cache_dir=None):
        # pylint: disable-next=import-outside-toplevel
        import transformers

        model = transformers.WavLMForXVector.from_pretrained(model_name)
        self.m = model.to(DEV).eval()
        revision = getattr(self.m.config, "_commit_hash", None) or "local"
        folder = f"{model_name.replace('/', '--')}@{revision}"
        self.cache_dir = os.path.join(cache_dir, folder) if cache_dir else None
        self._mem = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    @torch.inference_mode()
    def emb(self, wav48, key=None):
        if key is not None:
            if key in self._mem:
                return self._mem[key]
            path = (os.path.join(self.cache_dir, f"{key}.pt")
                    if self.cache_dir else None)
            if path and os.path.exists(path):
                e = torch.load(path, weights_only=True)
                self._mem[key] = e
                return e
        x = torchaudio.functional.resample(wav48.float(), SR, 16000).to(DEV)
        out = self.m(x.unsqueeze(0) if x.ndim == 1 else x)
        e = getattr(out, "embeddings", None)
        if e is None:
            e = out[0] if isinstance(out, (tuple, list)) else out
        e = torch.nn.functional.normalize(e.squeeze(0), dim=-1).cpu()
        if key is not None:
            self._mem[key] = e
            if self.cache_dir:
                tmp = f"{path}.tmp{os.getpid()}"
                torch.save(e, tmp)
                os.replace(tmp, path)
        return e

    def sim(self, a48, b48, key_b=None):
        """Returns the cosine similarity of two clips.

        ``key_b`` names the second one for the cache.
        """
        return float(torch.dot(self.emb(a48), self.emb(b48, key=key_b)))


class Vad:
    """Silero VAD helpers for reference-free (real-file) metrics."""

    def __init__(self):
        # pylint: disable-next=import-outside-toplevel
        import importlib.metadata

        # pylint: disable-next=import-outside-toplevel
        import silero_vad

        self._get = silero_vad.get_speech_timestamps
        self.vad = silero_vad.load_silero_vad()
        # what decides the segments: a cache entry of another detector is never
        # read
        self.identity = (
            f"silero-vad {importlib.metadata.version('silero-vad')}, defaults,"
            " 16 kHz")

    def segments(self, wav48):
        w16 = torchaudio.functional.resample(wav48, SR, 16000)
        return w16, self._get(w16, self.vad, sampling_rate=16000)
