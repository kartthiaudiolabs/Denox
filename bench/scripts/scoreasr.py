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
"""Downstream ASR: what a recogniser gets out of the audio, processed or not.

    scoreasr.py --systems INPUT,voice_isolator_v0[,...]
        [--asr large-v3,small,inworld/inworld-stt-1] [--conds ...] [--limit N]
        [--device cuda] [--workers 8]

Recognisers: a bare name is a faster-whisper model size run locally;
``inworld/<model>`` is the Inworld speech-to-text API (key in
``INWORLD_API_KEY``), called concurrently with ``--workers``.

For every paired item of the chosen conditions the recogniser transcribes each
system's output and the transcript is scored against the item's golden text:
``transcripts.jsonl`` in the set root, one verified transcript per target
utterance with the tier of its verification. An item without a golden text falls
back to the recogniser's own transcript of the clean reference (cached), and the
sidecar records how many items each condition scored that way, so a number can
always be traced to its ground truth. Whisper's own English normaliser, or its
basic normaliser for the other languages, is applied to both sides; Chinese and
Japanese are scored per character. For each condition the sidecar holds the
pooled error rate (total edits over total reference tokens), a bootstrap 95 %
interval over items, and the per-item edit counts.

Writes ``<BENCH_RESULTS>/<system>.asr.json``, one block per recogniser.
Transcripts are cached by the audio a recogniser hears (the decoded samples,
their rate and channel count, never the file's bytes, which a float WAV also
stamps with the time it was written) and by decoding identity under
``BENCH_CACHE`` (default ``<repo>/data/asr_cache``), so a re-run, or a re-render
that reproduces the same audio, transcribes only new audio, and any change of
recogniser, version or decoding setting transcribes everything again.
"""

import argparse
import base64
from concurrent import futures
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import soundfile as sf
import torch
import torchaudio

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import observations
import scorebench

SCORER_SHA256 = observations.implementation(__file__, common.__file__,
                                            observations.__file__,
                                            scorebench.__file__)

DEFAULT_CONDS = (
    "clean",
    "babble@+4",
    "babble@+0",
    "babble@-5",
    "babble@-10",
    "ambience@-5",
    "ambience@-10",
    "stationary@-5",
    "stationary@-10",
    "traffic@-5",
    "traffic@-10",
    "events@-5",
    "music@-6",
    "wind@+0",
    "quiet@-30dBFS",
    "conversation@0",
    "long@+4",
    "wild@+0",
    "wild@-5",
    "wild@-10",
    "soft@+5",
    "soft@+0",
    "soft@-5",
    "whisper@+5",
    "whisper@+0",
)
CHAR_LANGS = frozenset(
    # no word boundaries: per character
    ("zh", "yue", "ja", "th", "lo", "my", "km"))
# a recogniser without a model for a language decodes it as its nearest written
# standard
NEAREST = {"yue": "zh"}
CACHE_ROOT = os.environ.get("BENCH_CACHE",
                            os.path.join(common.REPO_DIR, "data", "asr_cache"))
# verified transcripts of the target utterances, in the set root
GOLDEN = "transcripts.jsonl"
FALLBACK = "clean-asr"  # the recogniser's own transcript of the clean reference
N_BOOT = 1000


def edit_distance(a, b):
    """Levenshtein distance between two token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


class Normalizer:
    """Whisper's text normalisers, from the tokenizer that ships them.

    The English one for ``en`` (spelling, numbers, contractions), the basic one
    otherwise. Tokens are words, or characters for the languages without word
    boundaries.
    """

    def __init__(self, tokenizer_name="openai/whisper-tiny"):
        # pylint: disable-next=import-outside-toplevel
        import transformers

        self._tok = transformers.WhisperTokenizer.from_pretrained(
            tokenizer_name)

    def tokens(self, text, language):
        if language == "en":
            norm = self._tok.normalize(text or "")
        else:
            norm = self._tok.basic_normalize(text or "")
        if language in CHAR_LANGS:
            return [ch for ch in norm if not ch.isspace()]
        return norm.split()


def unit_for(language):
    return "char" if language in CHAR_LANGS else "word"


class FasterWhisper:
    """faster-whisper (CTranslate2) in fp16 on the GPU.

    ``tag`` is the model size.
    """

    def __init__(self, size="large-v3", device=None, beam_size=5):
        # pylint: disable-next=import-outside-toplevel
        import ctranslate2
        # pylint: disable-next=import-outside-toplevel
        import faster_whisper

        self.tag = size
        self.beam_size = beam_size
        on_gpu = str(device or common.DEV).startswith("cuda")
        compute_type = "float16" if on_gpu else "int8"
        # the cached snapshot: <...>/snapshots/<rev>
        weights = faster_whisper.download_model(size)
        self.model = faster_whisper.WhisperModel(
            weights,
            device="cuda" if on_gpu else "cpu",
            compute_type=compute_type,
        )
        # everything that can change a transcript of the same audio
        self.identity = {
            "engine": "faster-whisper",
            "version": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__,
            "model": size,
            "model_revision": os.path.basename(os.path.normpath(weights)),
            "compute_type": compute_type,
            "beam_size": beam_size,
            "condition_on_previous_text": False,
            "vad_filter": False,
        }

    def transcribe(self, path, language=None):
        # Cantonese entered Whisper with large-v3; smaller checkpoints list the
        # token, were never trained with it and emit noise, so they decode it as
        # written Chinese
        if language in NEAREST and not self.tag.startswith("large-v3"):
            language = NEAREST[language]
        segs, _ = self.model.transcribe(
            path,
            beam_size=self.beam_size,
            language=None if language in (None, "und") else language,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text for s in segs).strip()


API_LANG = {  # the set's language codes as the API's locale tags
    "en": "en-US",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "it": "it-IT",
    "pt": "pt-PT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "zh": "zh-CN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
}
API_SR = 16000


def wav_bytes_16k(path):
    """The clip as 16 kHz 16-bit mono WAV bytes: what the speech APIs take."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    if sr != API_SR:
        x = torchaudio.functional.resample(x, sr, API_SR)
    pcm = (x.numpy() * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(API_SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class InworldSTT:
    """The Inworld speech-to-text REST API, one request per clip.

    ``tag`` is the model id.
    """

    URL = "https://api.inworld.ai/stt/v1/transcribe"
    concurrent = True

    def __init__(self,
                 model="inworld/inworld-stt-1",
                 key=None,
                 tries=6,
                 timeout=120):
        self.tag = model
        self.key = key or os.environ.get("INWORLD_API_KEY")
        if not self.key:
            raise SystemExit(
                "INWORLD_API_KEY is not set (the Inworld API key, Basic auth)")
        self.tries, self.timeout = tries, timeout
        self.identity = {
            "engine": "inworld-stt-api",
            "model": model,
            "encoding": "LINEAR16 16 kHz mono",
        }

    def transcribe(self, path, language=None):
        # pylint: disable-next=import-outside-toplevel
        import requests

        body = {
            "transcribeConfig": {
                "modelId": self.tag,
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": API_SR,
                "numberOfChannels": 1,
            },
            "audioData": {
                "content": base64.b64encode(wav_bytes_16k(path)).decode()
            },
        }
        if language and language != "und":
            body["transcribeConfig"]["language"] = API_LANG.get(
                language, API_LANG.get(NEAREST.get(language), language))
        delay = 1.0
        for k in range(self.tries):
            try:
                r = requests.post(
                    self.URL,
                    headers={
                        "Authorization": f"Basic {self.key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    transcription = r.json().get("transcription") or {}
                    return transcription.get("transcript", "").strip()
                if r.status_code < 500 and r.status_code != 429:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            # network or a non-JSON body
            except (requests.RequestException, ValueError) as e:
                if k == self.tries - 1:
                    raise RuntimeError(f"{type(e).__name__}: {e}") from e
            if k == self.tries - 1:
                raise RuntimeError(
                    f"HTTP {r.status_code} after {self.tries} tries:"
                    f" {r.text[:200]}")
            time.sleep(delay + random.random())
            delay = min(delay * 2, 30)


def make_asr(spec, device=None, beam_size=5):
    """Returns a recogniser from its spec.

    ``inworld/<model>`` is the Inworld API, anything else a faster-whisper model
    size.
    """
    if spec.startswith("inworld/"):
        return InworldSTT(spec)
    return FasterWhisper(spec, device, beam_size)


def cache_tag(asr):
    return f"{asr.identity['engine']}_{asr.tag}".replace("/", "_")


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# bumped whenever the audio a recogniser is handed, or its key, changes
CACHE_FORMAT = 3


def audio_key(path):
    """Returns the cache key of a file: what a recogniser hears.

    The decoded samples with their rate and channel count; two files of the same
    audio share it whatever their container bytes.
    """
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    digest = hashlib.sha256(f"{rate}|{samples.shape[1]}|".encode())
    digest.update(np.ascontiguousarray(samples).tobytes())
    return digest.hexdigest()


def identity_key(identity: dict) -> str:
    """Returns the cache directory of one decoding identity.

    A digest of every setting that can change a transcript (engine, model,
    version, decoding options) and of the cache format.
    """
    doc = dict(identity, cache_format=CACHE_FORMAT)
    text = json.dumps(doc, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Cache:
    """Transcripts by ``audio_key``, one JSON file per audio.

    The files live under ``<root>/<asr tag>/<identity key>/``. A transcript is
    reused only for the same audio decoded with the same identity: every entry
    records the identity it was decoded with and is read back only when that
    matches, and a new beam size, model or recogniser version starts a directory
    of its own. Requests for the same audio from several threads decode it once;
    each write goes through its own temporary file.
    """

    def __init__(self, root, tag, identity):
        self.identity = dict(identity)
        self.dir = os.path.join(root, tag, identity_key(identity))
        os.makedirs(self.dir, exist_ok=True)
        stamp = os.path.join(self.dir, "identity.json")
        if not os.path.exists(stamp):
            _write_json(stamp, dict(self.identity, cache_format=CACHE_FORMAT))
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        self._pending = {}

    def _read(self, path, language):
        try:
            with open(path, encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            return None
        if (entry.get("identity") != self.identity or
                entry.get("language", language) != language):
            # decoded otherwise, or in another language: asked again
            return None
        return entry["text"]

    def get(self, asr, path, language):
        key = audio_key(path)
        target = os.path.join(self.dir, f"{key}.json")
        with self._lock:
            lock = self._pending.setdefault((key, language), threading.Lock())
        # a second request for the same audio waits and then reads the first's
        # entry
        with lock:
            text = self._read(target, language)
            if text is not None:
                with self._lock:
                    self.hits += 1
                return text
            text = asr.transcribe(path, language)
            _write_json(
                target,
                {
                    "text": text,
                    "language": language,
                    "source": os.path.basename(path),
                    "identity": self.identity,
                },
            )
            with self._lock:
                self.misses += 1
            return text


def _write_json(path, doc):
    """Writes ``doc`` to ``path`` through a temporary file of its own.

    The file is then renamed into place.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def pooled(items):
    """Returns (pooled error rate %, total edits, total reference tokens).

    Over {id: [edits, ref_len]}.
    """
    e = sum(v[0] for v in items.values())
    n = sum(v[1] for v in items.values())
    return (100.0 * e / n if n else float("nan")), e, n


def bootstrap_ci(items, n_boot=N_BOOT, seed=0):
    """95 % interval of the pooled rate under resampling of items."""
    vals = np.array(list(items.values()), dtype=float)
    if len(vals) < 2 or vals[:, 1].sum() == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        pick = vals[rng.integers(0, len(vals), len(vals))]
        n = pick[:, 1].sum()
        rates.append(100.0 * pick[:, 0].sum() / n if n else float("nan"))
    rates = np.array(rates)
    rates = rates[np.isfinite(rates)]
    return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]


def load_golden(root=None):
    """Returns {id: {"text", "tier", ...}} from the set's transcripts.jsonl.

    Empty when the set has none.
    """
    path = os.path.join(root or common.EVAL, GOLDEN)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def spoken_language(r, golden):
    """Returns the language an item is decoded and normalised in.

    The manifest's, unless its golden transcript says another one is spoken
    (``spoken``: a label found wrong by ear).
    """
    return ((golden.get(r["id"]) or {}).get("spoken") or r.get("language") or
            "und")


def reference_tokens(r, golden, asr, cache, norm):
    """Returns (tokens, tier) of the item's ground truth.

    The golden text, else the clean-reference transcript of the same recogniser.
    """
    lang = spoken_language(r, golden)
    g = golden.get(r["id"])
    if g and g.get("text"):
        return norm.tokens(g["text"], lang), g.get("tier") or "golden"
    return norm.tokens(cache.get(asr, r["clean_path"], lang), lang), FALLBACK


def prefill(asr, cache, rows, outd, workers, golden=None):
    """Transcribes every output, and every reference without a golden text.

    ``workers`` at a time: the API recognisers are latency-bound, so the
    sequential scoring pass then hits the cache.
    """
    golden = golden or {}
    jobs = []
    for r in rows:
        lang = spoken_language(r, golden)
        if not (golden.get(r["id"]) or {}).get("text"):
            jobs.append((r["clean_path"], lang))
        op = os.path.join(outd, r["id"])
        if os.path.exists(op):
            jobs.append((op, lang))
    with futures.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: cache.get(asr, *j), jobs))


def score_system(name, rows, asr, cache, norm, golden=None, workers=1):
    """Returns one system's per-condition error rates against the ground truth.

    ``rows`` are the paired manifest rows to score, ``golden`` the loaded
    transcripts.jsonl (may be empty).
    """
    golden = golden or {}
    outd = os.path.join(common.OUT_ROOT, name)
    if workers > 1 and getattr(asr, "concurrent", False):
        prefill(asr, cache, rows, outd, workers, golden)
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    scores = {}
    for cond, items in sorted(by_cond.items()):
        recs, units, n_empty_ref, n_missing = {}, {"word": 0, "char": 0}, 0, 0
        tiers = {}
        for r in items:
            op = os.path.join(outd, r["id"])
            if not os.path.exists(op):
                n_missing += 1
                continue
            lang = spoken_language(r, golden)
            ref, tier = reference_tokens(r, golden, asr, cache, norm)
            if not ref:
                n_empty_ref += 1
                continue
            hyp = norm.tokens(cache.get(asr, op, lang), lang)
            recs[r["id"]] = [edit_distance(hyp, ref), len(ref)]
            units[unit_for(lang)] += 1
            tiers[tier] = tiers.get(tier, 0) + 1
        if not recs:
            continue
        rate, e, n = pooled(recs)
        scores[cond] = {
            "n": len(recs),
            "n_missing": n_missing,
            "n_empty_reference": n_empty_ref,
            "units": units,
            "reference_tiers": tiers,
            "edits": e,
            "ref_tokens": n,
            "wer_pct": rate,
            "ci95": bootstrap_ci(recs),
            "items": recs,
        }
    return scores


def stale_sidecar(doc, results):
    """Returns why an ASR-family sidecar may not be shown, or None.

    It was scored by another implementation (``_scorer_sha256`` against
    ``expected``), on a partial selection, or on other audio than the system's
    base records describe (when they are at hand in ``results``).
    """
    name = doc.get("_system")
    if (doc.get("_selection") or {}).get("limit"):
        return (f"scored only {doc['_selection']['limit']} items per condition"
                " (--limit)")
    base = os.path.join(results, f"{name}.items.jsonl")
    if doc.get("_outputs") is None:
        return "records no audio identity (scored before outputs were recorded)"
    if os.path.isfile(base):
        _, records = observations.read(base)
        scored = {
            r["id"]: (r.get("identity") or {}).get("output") for r in records
        }
        other = [
            uid for uid, h in doc["_outputs"].items() if scored.get(uid, h) != h
        ]
        if other:
            return (
                f"read other audio than the base scores for {len(other)} items"
                f" ({other[0]})")
    return None


def output_hashes(name, rows):
    """Returns {item id: sample hash} of the outputs of ``name`` in ``rows``.

    Hashed as the base records identify the audio they scored.
    """
    outd = os.path.join(common.OUT_ROOT, name)
    return {
        r["id"]:
            scorebench.sample_hash(common.load48(os.path.join(outd, r["id"])))
        for r in rows
        if os.path.exists(os.path.join(outd, r["id"]))
    }


def write_sidecar(path, name, tag, block, asr_identity, normaliser_version,
                  outputs, selection):
    """Merges one recogniser's block into <system>.asr.json.

    The file records the audio its transcripts came from (``_outputs``: the
    sample hash of every output, as the base records identify it), the scorer
    that read them and the items selected. Blocks of other recognisers are kept
    only when they read the same audio with the same scorer and selection; a
    file of another set build or subset is replaced, not merged.
    """
    doc = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            doc = {}
        same = (doc.get("_set_id") == common.set_id() and
                doc.get("_subset", "all") == common.SUBSET and
                doc.get("_outputs") == outputs and
                doc.get("_scorer_sha256") == SCORER_SHA256 and
                doc.get("_selection") == selection)
        if not same:
            dropped = sorted(t for t in doc.get("asr") or {} if t != tag)
            if dropped:
                print(
                    f"  {name}: dropping {', '.join(dropped)}: read other"
                    " audio, scorer or items",
                    flush=True,
                )
            doc = {}
    doc.update({
        "_system": name,
        "_set_id": common.set_id(),
        "_subset": common.SUBSET,
        "_scorer_sha256": SCORER_SHA256,
        "_selection": selection,
        "_outputs": outputs,
        "_reference": f"{GOLDEN} where present, else {FALLBACK}",
        "_normaliser": (f"transformers {normaliser_version} WhisperTokenizer"),
    })
    doc.setdefault("asr", {})[tag] = {
        "identity": asr_identity,
        "conditions": block,
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--systems",
        required=True,
        help="comma-separated system names (INPUT included)",
    )
    ap.add_argument(
        "--asr",
        default="large-v3",
        help=(
            "comma-separated: faster-whisper sizes, or inworld/<model> for the"
            " Inworld API"),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="concurrent requests for API recognisers",
    )
    ap.add_argument("--conds", default=",".join(DEFAULT_CONDS))
    ap.add_argument("--limit",
                    type=int,
                    default=0,
                    help="items per condition (0: all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--beam-size", type=int, default=5)
    a = ap.parse_args()
    # pylint: disable-next=import-outside-toplevel
    import transformers

    rows_all = common.manifest()
    common.guard_private_results(rows_all)
    if common.is_sealed(rows_all):
        raise SystemExit(
            "sealed audio cannot be sent to an ASR service or exported through"
            " ASR reports")
    rows = []
    for cond in [c for c in a.conds.split(",") if c]:
        rows += common.pick_rows(rows_all, cond, a.limit, paired=True)
    norm = Normalizer()
    golden = load_golden()
    print(
        (f"golden transcripts: {len(golden)} items in"
         f" {os.path.join(common.EVAL, GOLDEN)}" if golden else
         f"no {GOLDEN} in the set: every item falls back to {FALLBACK}"),
        flush=True,
    )
    os.makedirs(common.RESULTS, exist_ok=True)
    for tag in [t for t in a.asr.split(",") if t]:
        asr = make_asr(tag, a.device, a.beam_size)
        cache = Cache(CACHE_ROOT, cache_tag(asr), asr.identity)
        for name in [s for s in a.systems.split(",") if s]:
            problem = name != "INPUT" and scorebench.render_problem(name)
            if problem:
                print(f"skip {name}: {problem}", flush=True)
                continue
            scores = score_system(name, rows, asr, cache, norm, golden,
                                  a.workers)
            outputs = output_hashes(name, rows)
            write_sidecar(
                os.path.join(common.RESULTS, f"{name}.asr.json"),
                name,
                tag,
                scores,
                asr.identity,
                transformers.__version__,
                outputs,
                {
                    "conds": a.conds,
                    "limit": a.limit
                },
            )
            worst = sorted(scores.items(), key=lambda kv: -kv[1]["wer_pct"])[:3]
            print(
                f"{tag} {name}: " +
                "  ".join(f"{c} {d['wer_pct']:.1f}%" for c, d in worst) +
                f"  (cache {cache.hits} hits / {cache.misses} new)",
                flush=True,
            )


if __name__ == "__main__":
    main()
