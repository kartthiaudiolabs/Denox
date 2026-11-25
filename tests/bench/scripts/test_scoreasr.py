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
"""Downstream-ASR sidecar: edit distance, pooling, references and the cache.

Golden references and their fallback, with a fake recogniser that reads its
transcript from a text file next to the audio.
"""

# pylint: disable=redefined-outer-name,unused-argument

import json
import math
import os
import shutil
import time
import zlib

import pytest
import soundfile as sf
import torch

import common
import scoreasr
import scorebench

SR = common.SR


class _Norm:
    """Stands in for Whisper's normaliser.

    Lower case, no punctuation; characters for zh/ja.
    """

    def tokens(self, text, language):
        clean = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " "
                        for ch in text or "")
        if language in scoreasr.CHAR_LANGS:
            return [ch for ch in clean if not ch.isspace()]
        return clean.split()


class _Asr:
    """Transcribes by reading ``<audio>.txt``.

    Counts calls so the cache can be checked.
    """

    tag = "fake"
    identity = {"engine": "fake", "model": "fake"}

    def __init__(self):
        self.calls = 0

    def transcribe(self, path, language=None):
        self.calls += 1
        with open(path + ".txt", encoding="utf-8") as fh:
            return fh.read()


def test_edit_distance_on_words_and_characters():
    assert scoreasr.edit_distance([], []) == 0
    assert scoreasr.edit_distance("a b c".split(), "a b c".split()) == 0
    # substitution
    assert scoreasr.edit_distance("a b c".split(), "a x c".split()) == 1
    # deletion
    assert scoreasr.edit_distance("a b c".split(), "a c".split()) == 1
    # insertion
    assert scoreasr.edit_distance("a c".split(), "a b c".split()) == 1
    # an empty hypothesis: every word lost
    assert scoreasr.edit_distance([], "a b c".split()) == 3
    assert scoreasr.edit_distance(list("kitten"), list("sitting")) == 3
    assert (scoreasr.unit_for("zh") == "char" and
            scoreasr.unit_for("en") == "word")


def test_pooled_rate_and_bootstrap_interval():
    items = {"a": [1, 10], "b": [0, 10], "c": [3, 10]}
    rate, e, n = scoreasr.pooled(items)
    assert (rate, e, n) == (pytest.approx(100 * 4 / 30), 4, 30)
    lo, hi = scoreasr.bootstrap_ci(items, n_boot=200)
    assert lo <= rate <= hi and lo >= 0
    # one item: no interval
    assert all(math.isnan(v) for v in scoreasr.bootstrap_ci({"a": [1, 10]}))
    assert math.isnan(scoreasr.pooled({})[0])


def _write(path, text, x=None, perturb=False):
    """Audio plus its fake transcript.

    Outputs are perturbed by a hash of their transcript, so two files that would
    transcribe differently never share bytes (the cache keys on bytes) while two
    systems saying the same thing do.
    """
    x = torch.zeros(SR // 4) if x is None else x
    if perturb:
        x = x * (1 + (zlib.crc32(text.encode()) % 997 + 1) * 1e-6) + 1e-7
    sf.write(str(path), x.numpy(), SR, subtype="FLOAT")
    with open(str(path) + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.fixture
def asr_set(tmp_path, monkeypatch):
    """A set with three conditions.

    The outputs' transcripts are chosen per system by the test.
    """
    root, out_root = tmp_path / "set", tmp_path / "out"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    rows = []
    specs = [
        ("clean", "en", "The quick brown fox jumps over the lazy dog"),
        ("clean", "en", "Hello there, General Kenobi!"),
        ("babble@-10", "en", "We hold these truths to be self evident"),
        ("babble@-10", "zh", "今天天气很好"),
        ("wild@-5", "de", "Guten Morgen liebe Sorgen"),
    ]
    for i, (cond, lang, text) in enumerate(specs):
        rid = f"{cond.replace('@', 'AT')}_{i:03d}.wav"
        g = torch.Generator().manual_seed(i)
        _write(root / "ref" / rid, text,
               torch.randn(SR // 4, generator=g) * 0.1)
        _write(
            root / "mix" / rid,
            text + " noise",
            torch.randn(SR // 4, generator=g) * 0.2,
        )
        rows.append({
            "id": rid,
            "condition": cond,
            "language": lang,
            "mix_path": f"mix/{rid}",
            "clean_path": f"ref/{rid}",
        })
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump(
        {
            "set_id": "tiny",
            "n_items": len(rows)
        },
        open(root / "checksums.json", "w", encoding="utf-8"),
    )
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(out_root))
    monkeypatch.setattr(common, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(scoreasr, "CACHE_ROOT", str(tmp_path / "cache"))
    return root, out_root, common.manifest(str(root))


def render(rows, out_root, name, fn):
    """Writes each output as its reference audio.

    Its transcript is ``fn(row)``.
    """
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        x, _ = sf.read(r["clean_path"], dtype="float32")
        _write(d / r["id"], fn(r), torch.from_numpy(x), perturb=True)


def _golden(rows):
    return {
        r["id"]: {
            "id": r["id"],
            "text": open(r["clean_path"] + ".txt", encoding="utf-8").read(),
            "tier": "corpus-verified",
        } for r in rows
    }


def test_perfect_output_reads_zero_and_a_muted_one_reads_one_hundred(
        asr_set, tmp_path):
    _, out_root, rows = asr_set
    render(
        rows,
        out_root,
        "perfect",
        lambda r: open(r["clean_path"] + ".txt", encoding="utf-8").read(),
    )
    render(rows, out_root, "mute", lambda r: "")
    asr = _Asr()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    perfect = scoreasr.score_system("perfect", rows, asr, cache, _Norm(),
                                    _golden(rows))
    assert set(perfect) == {"clean", "babble@-10", "wild@-5"}
    assert all(v["wer_pct"] == 0.0 for v in perfect.values())
    # one English, one Chinese item
    assert perfect["babble@-10"]["units"] == {"word": 1, "char": 1}
    assert perfect["babble@-10"]["reference_tiers"] == {"corpus-verified": 2}
    metrics = scoreasr.score_system("mute", rows, asr, cache, _Norm(),
                                    _golden(rows))
    assert all(v["wer_pct"] == 100.0 for v in metrics.values())
    assert metrics["clean"]["ref_tokens"] == 9 + 4
    assert metrics["clean"]["edits"] == 13


def test_a_partly_wrong_output_is_pooled_over_reference_tokens(
        asr_set, tmp_path):
    _, out_root, rows = asr_set
    render(
        rows,
        out_root,
        "half",
        lambda r: " ".join(
            # every other word dropped
            open(r["clean_path"] + ".txt", encoding="utf-8").read().split()[::2]
        ),
    )
    asr = _Asr()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    metrics = scoreasr.score_system("half", rows, asr, cache, _Norm(),
                                    _golden(rows))
    c = metrics["clean"]
    assert c["items"]["clean_000.wav"] == [
        4, 9
    ] and c["items"]["clean_001.wav"] == [2, 4]
    assert c["wer_pct"] == pytest.approx(100 * 6 / 13) and c["n"] == 2
    lo, hi = c["ci95"]
    assert lo <= c["wer_pct"] <= hi


def test_without_golden_text_clean_transcript_is_the_reference_and_is_recorded(
        asr_set, tmp_path):
    root, out_root, rows = asr_set
    render(
        rows,
        out_root,
        "perfect",
        lambda r: open(r["clean_path"] + ".txt", encoding="utf-8").read(),
    )
    asr = _Asr()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    golden = {
        k: v for k, v in _golden(rows).items() if not k.startswith("wild")
    }
    metrics = scoreasr.score_system("perfect", rows, asr, cache, _Norm(),
                                    golden)
    assert metrics["clean"]["reference_tiers"] == {"corpus-verified": 2}
    assert metrics["wild@-5"]["reference_tiers"] == {scoreasr.FALLBACK: 1}
    assert metrics["wild@-5"]["wer_pct"] == 0.0
    # no transcripts.jsonl in this set
    assert not scoreasr.load_golden(str(root))
    with open(root / scoreasr.GOLDEN, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({
                "id": "wildAT-5_004.wav",
                "text": "Guten Morgen",
                "tier": "t"
            }) + "\n")
    g = scoreasr.load_golden(str(root))
    assert g["wildAT-5_004.wav"]["tier"] == "t"
    metrics_again = scoreasr.score_system("perfect", rows, asr, cache, _Norm(),
                                          g)
    # two extra words against a 2-word gold
    assert metrics_again["wild@-5"]["wer_pct"] == pytest.approx(100 * 2 / 2)


def test_the_cache_transcribes_each_audio_once(asr_set, tmp_path):
    _, out_root, rows = asr_set
    render(
        rows,
        out_root,
        "perfect",
        lambda r: open(r["clean_path"] + ".txt", encoding="utf-8").read(),
    )
    asr = _Asr()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    # references + outputs
    scoreasr.score_system("perfect", rows, asr, cache, _Norm(), {})
    first = asr.calls
    assert first == 2 * len(rows) and cache.misses == first and cache.hits == 0
    scoreasr.score_system("perfect", rows, asr, cache, _Norm(), {})
    # everything from the cache the second time
    assert asr.calls == first and cache.hits == first
    # the same audio under another system name is a hit too (outputs are keyed
    # by content)
    # Rebuilding a WAV can change header metadata even with identical samples.
    shutil.copytree(out_root / "perfect", out_root / "twin")
    scoreasr.score_system("twin", rows, asr, cache, _Norm(), {})
    assert asr.calls == first


def test_a_transcript_can_name_the_language_actually_spoken(asr_set, tmp_path):
    """An item labelled German whose audio is English is decoded as English.

    The golden row says so, and both the decoding language and the normaliser
    follow it; the cache never serves the other decoding.
    """
    _, out_root, rows = asr_set
    wild = [r for r in rows if r["condition"] == "wild@-5"]
    render(wild, out_root, "sys",
           lambda r: open(r["clean_path"] + ".txt", encoding="utf-8").read())

    class _Langs(_Asr):

        def __init__(self):
            super().__init__()
            self.asked = []

        def transcribe(self, path, language=None):
            self.asked.append(language)
            return super().transcribe(path, language)

    asr = _Langs()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    scoreasr.score_system("sys", wild, asr, cache, _Norm(), {})
    # the manifest's label, as long as nothing says otherwise
    assert set(asr.asked) == {"de"}
    golden = {
        wild[0]["id"]: {
            "id": wild[0]["id"],
            "language": "de",
            "spoken": "en",
            "text": "Guten Morgen liebe Sorgen",
            "tier": "gemini-accepted",
        }
    }
    assert (scoreasr.spoken_language(wild[0], golden) == "en" and
            scoreasr.spoken_language(wild[0], {}) == "de")
    asr.asked.clear()
    out = scoreasr.score_system("sys", wild, asr, cache, _Norm(), golden)
    # the output is decoded again in the spoken language, not served
    assert asr.asked == ["en"]
    assert out["wild@-5"]["wer_pct"] == 0.0
    assert out["wild@-5"]["reference_tiers"] == {"gemini-accepted": 1}
    asr.asked.clear()
    scoreasr.score_system("sys", wild, asr, cache, _Norm(), golden)
    assert not asr.asked  # and that decoding is cached in turn


def test_missing_and_empty_reference_items_are_counted_not_scored(
        asr_set, tmp_path):
    _, out_root, rows = asr_set
    render(
        [r for r in rows if r["condition"] != "clean"],
        out_root,
        "partial",
        lambda r: open(r["clean_path"] + ".txt", encoding="utf-8").read(),
    )
    asr = _Asr()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    golden = _golden(rows)
    # a golden entry with no text: fall back
    golden["babbleAT-10_002.wav"]["text"] = ""
    # normalises to nothing: unscorable
    golden["wildAT-5_004.wav"]["text"] = "..."
    metrics = scoreasr.score_system("partial", rows, asr, cache, _Norm(),
                                    golden)
    assert "clean" not in metrics  # nothing rendered
    assert metrics["babble@-10"]["reference_tiers"] == {
        "corpus-verified": 1,
        scoreasr.FALLBACK: 1,
    }
    # its only item had an empty reference
    assert "wild@-5" not in metrics or metrics.get("wild@-5") is None


SEL = {"conds": "clean", "limit": 0}


def test_sidecar_merges_recognisers_and_replaces_another_build(
        asr_set, tmp_path, monkeypatch):
    path = tmp_path / "sys.asr.json"
    scoreasr.write_sidecar(
        str(path),
        "sys",
        "large-v3",
        {"clean": {
            "wer_pct": 1.0
        }},
        {"m": 1},
        "4.57",
        {"a.wav": "h"},
        SEL,
    )
    scoreasr.write_sidecar(
        str(path),
        "sys",
        "small",
        {"clean": {
            "wer_pct": 2.0
        }},
        {"m": 2},
        "4.57",
        {"a.wav": "h"},
        SEL,
    )
    d = json.load(open(path, encoding="utf-8"))
    assert set(d["asr"]) == {"large-v3", "small"} and d["_set_id"] == "tiny"
    assert d["_reference"].startswith(scoreasr.GOLDEN)
    monkeypatch.setattr(common, "set_id", lambda root=None: "other")
    scoreasr.write_sidecar(
        str(path),
        "sys",
        "small",
        {"clean": {
            "wer_pct": 3.0
        }},
        {"m": 2},
        "4.57",
        {"a.wav": "h"},
        SEL,
    )
    d = json.load(open(path, encoding="utf-8"))
    # another build: start over
    assert set(d["asr"]) == {"small"} and d["_set_id"] == "other"
    assert d["_outputs"] == {"a.wav": "h"}
    assert d["_scorer_sha256"] == scoreasr.SCORER_SHA256
    scoreasr.write_sidecar(
        str(path),
        "sys",
        "large-v3",
        {"clean": {
            "wer_pct": 1.0
        }},
        {"m": 1},
        "4.57",
        {"a.wav": "h"},
        SEL,
    )
    # the same audio: merged
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    assert set(d["asr"]) == {"small", "large-v3"}
    scoreasr.write_sidecar(
        str(path),
        "sys",
        "large-v3",
        {"clean": {
            "wer_pct": 1.0
        }},
        {"m": 1},
        "4.57",
        {"a.wav": "re-rendered"},
        SEL,
    )
    # other audio: never mixed
    assert set(json.load(open(path, encoding="utf-8"))["asr"]) == {"large-v3"}
    scoreasr.write_sidecar(
        str(path),
        "sys",
        "small",
        {"clean": {
            "wer_pct": 1.0
        }},
        {"m": 1},
        "4.57",
        {"a.wav": "re-rendered"},
        dict(SEL, limit=5),
    )
    # another selection: not merged either
    assert set(json.load(open(path, encoding="utf-8"))["asr"]) == {"small"}


def test_whisper_normaliser_when_its_tokenizer_is_cached():
    try:
        norm = scoreasr.Normalizer()
    except Exception as e:  # pylint: disable=broad-exception-caught
        # no cached tokenizer and no network: whatever stops it, the test skips
        pytest.skip(f"whisper tokenizer unavailable: {e}")
    assert (norm.tokens("Hello, World! It's 5 o'clock.",
                        "en") == "hello world it is 50 clock".split())
    assert norm.tokens("Ça va, l'ami ?", "fr") == ["ça", "va", "l", "ami"]
    assert norm.tokens("今天 天气很好。", "zh") == list("今天天气很好")


def test_main_writes_one_sidecar_per_system(asr_set, tmp_path, monkeypatch):
    pytest.importorskip("transformers", reason="needs the eval extra")
    root, out_root, rows = asr_set
    render(
        rows,
        out_root,
        "perfect",
        lambda r: open(r["clean_path"] + ".txt", encoding="utf-8").read(),
    )
    (out_root / "INPUT").mkdir()
    # INPUT: the mixtures, whose fake transcripts carry an extra word
    for r in rows:
        os.symlink(r["mix_path"], out_root / "INPUT" / r["id"])
        os.symlink(r["mix_path"] + ".txt",
                   out_root / "INPUT" / (r["id"] + ".txt"))
    with open(root / scoreasr.GOLDEN, "w", encoding="utf-8") as fh:
        for g in _golden(rows).values():
            fh.write(json.dumps(g) + "\n")
    monkeypatch.setattr(scoreasr, "Normalizer", _Norm)
    monkeypatch.setattr(scoreasr, "FasterWhisper",
                        lambda size, device, beam: _Asr())
    monkeypatch.setattr(scorebench, "render_problem", lambda name: None)
    monkeypatch.setattr(
        "sys.argv",
        ["scoreasr.py", "--systems", "INPUT,perfect", "--asr", "fake"],
    )
    scoreasr.main()
    results = tmp_path / "results"
    with open(results / "INPUT.asr.json", encoding="utf-8") as fh:
        inp = json.load(fh)["asr"]["fake"]["conditions"]
    with open(results / "perfect.asr.json", encoding="utf-8") as fh:
        per = json.load(fh)["asr"]["fake"]["conditions"]
    # one inserted word per item
    assert inp["clean"]["wer_pct"] == pytest.approx(100 * 2 / 13)
    assert per["clean"]["wer_pct"] == 0.0
    assert inp["clean"]["reference_tiers"] == {"corpus-verified": 2}


def test_inworld_backend_posts_16k_wav_and_reads_the_transcript(
        asr_set, tmp_path, monkeypatch):
    pytest.importorskip("requests", reason="needs the eval extra")
    _, _, rows = asr_set
    seen = []

    class _Resp:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def json(self):
            return {"transcription": {"transcript": "  hello there  "}}

    def fake_post(url, headers, json, timeout):
        seen.append((url, headers, json))
        return _Resp()

    # pylint: disable-next=import-outside-toplevel
    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("INWORLD_API_KEY", "k3y")
    asr = scoreasr.make_asr("inworld/inworld-stt-1")
    assert (asr.concurrent and
            scoreasr.cache_tag(asr) == "inworld-stt-api_inworld_inworld-stt-1")
    assert asr.transcribe(rows[0]["clean_path"], "de") == "hello there"
    url, headers, body = seen[0]
    assert (url == scoreasr.InworldSTT.URL and
            headers["Authorization"] == "Basic k3y")
    cfg = body["transcribeConfig"]
    assert cfg == {
        "modelId": "inworld/inworld-stt-1",
        "audioEncoding": "LINEAR16",
        "sampleRateHertz": 16000,
        "numberOfChannels": 1,
        "language": "de-DE",
    }
    # pylint: disable-next=import-outside-toplevel
    import base64
    # pylint: disable-next=import-outside-toplevel
    import io
    # pylint: disable-next=import-outside-toplevel
    import wave

    with wave.open(io.BytesIO(base64.b64decode(
            body["audioData"]["content"]))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (
            1,
            2,
            16000,
        )
        # the quarter-second test clip
        assert abs(w.getnframes() / 16000 - 0.25) < 0.01
    asr.transcribe(rows[0]["clean_path"], "und")
    # unknown language: let the API decide
    assert "language" not in seen[1][2]["transcribeConfig"]


def test_inworld_backend_retries_server_errors_and_fails_on_client_errors(
        asr_set, monkeypatch):
    pytest.importorskip("requests", reason="needs the eval extra")
    _, _, rows = asr_set
    # pylint: disable-next=import-outside-toplevel
    import requests

    calls = {"n": 0}

    class _Resp:
        headers = {"content-type": "application/json"}

        def __init__(self, code, text=""):
            self.status_code, self.text = code, text

        def json(self):
            return {"transcription": {"transcript": "ok"}}

    def flaky(url, headers, json, timeout):
        calls["n"] += 1
        return _Resp(503, "busy") if calls["n"] < 3 else _Resp(200)

    monkeypatch.setattr(requests, "post", flaky)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    asr = scoreasr.InworldSTT("inworld/inworld-stt-1", key="k", tries=4)
    assert (asr.transcribe(rows[0]["clean_path"], "en") == "ok" and
            calls["n"] == 3)
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(400, "unsupported model"))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        asr.transcribe(rows[0]["clean_path"], "en")
    monkeypatch.delenv("INWORLD_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        scoreasr.make_asr("inworld/x")


def test_concurrent_prefill_fills_the_cache_once_per_file(asr_set, tmp_path):
    _, out_root, rows = asr_set
    render(
        rows,
        out_root,
        "perfect",
        lambda r: open(r["clean_path"] + ".txt", encoding="utf-8").read(),
    )

    class _Api(_Asr):
        concurrent = True

    asr = _Api()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "api", _Asr.identity)
    metrics = scoreasr.score_system("perfect",
                                    rows,
                                    asr,
                                    cache,
                                    _Norm(),
                                    _golden(rows),
                                    workers=4)
    # outputs only: refs have gold
    assert asr.calls == len(rows) and cache.misses == asr.calls
    # the scoring pass reads everything the prefill wrote
    assert cache.hits == len(rows)
    assert all(v["wer_pct"] == 0.0 for v in metrics.values())
    golden = {
        k: v for k, v in _golden(rows).items() if not k.startswith("wild")
    }
    asr2 = _Api()
    cache2 = scoreasr.Cache(str(tmp_path / "cache2"), "api", _Asr.identity)
    scoreasr.score_system("perfect",
                          rows,
                          asr2,
                          cache2,
                          _Norm(),
                          golden,
                          workers=4)
    # plus the one reference without a golden text
    assert asr2.calls == len(rows) + 1


def test_cantonese_is_scored_per_character_and_decoded_as_chinese_if_no_model():
    assert ("yue" in scoreasr.CHAR_LANGS and
            scoreasr.unit_for("yue") == "char" and
            scoreasr.NEAREST["yue"] == "zh")

    class _Model:
        supported_languages = ["en", "zh", "yue"]

        def transcribe(self, path, **kw):
            self.language = kw["language"]
            return [], None

    for tag, expected in (
        ("large-v3", "yue"),
        ("small", "zh"),
        ("large-v2", "zh"),
    ):
        fw = scoreasr.FasterWhisper.__new__(scoreasr.FasterWhisper)
        fw.tag, fw.beam_size, fw.model = tag, 5, _Model()
        fw.transcribe("x.wav", "yue")
        assert fw.model.language == expected
        fw.transcribe("x.wav", "de")
        assert fw.model.language == "de"


def test_the_cache_never_serves_a_transcript_decoded_otherwise(
        asr_set, tmp_path):
    _, _, rows = asr_set
    path = rows[0]["clean_path"]
    asr = _Asr()
    cache_dir = str(tmp_path / "cache")
    beam5 = dict(_Asr.identity, beam_size=5)
    scoreasr.Cache(cache_dir, "fake", beam5).get(asr, path, "en")
    assert asr.calls == 1
    scoreasr.Cache(cache_dir, "fake", beam5).get(asr, path, "en")
    assert asr.calls == 1  # the same identity reads its entry
    beam1 = dict(beam5, beam_size=1)
    scoreasr.Cache(cache_dir, "fake", beam1).get(asr, path, "en")
    # another beam size decodes again, into a directory of its own
    assert asr.calls == 2
    version2 = dict(beam5, version="2")
    scoreasr.Cache(cache_dir, "fake", version2).get(asr, path, "en")
    assert asr.calls == 3
    dirs = sorted(p.name for p in (tmp_path / "cache" / "fake").iterdir())
    assert len(dirs) == 3 and all(len(d) == 16 for d in dirs)
    # an entry in the right place whose recorded identity differs is not read
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", beam5)
    entry = os.path.join(cache.dir, scoreasr.audio_key(path) + ".json")
    doc = json.load(open(entry, encoding="utf-8"))
    json.dump(dict(doc, identity={"engine": "else"}),
              open(entry, "w", encoding="utf-8"))
    cache.get(asr, path, "en")
    assert asr.calls == 4
    with open(entry, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["identity"] == beam5


def test_concurrent_requests_for_the_same_audio_decode_it_once(
        asr_set, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import threading

    _, _, rows = asr_set
    path = rows[0]["clean_path"]

    class _Slow(_Asr):

        def transcribe(self, path, language=None):
            time.sleep(0.05)
            return super().transcribe(path, language)

    asr = _Slow()
    cache = scoreasr.Cache(str(tmp_path / "cache"), "fake", _Asr.identity)
    errors = []

    def ask():
        try:
            cache.get(asr, path, "en")
        except Exception as e:  # pylint: disable=broad-exception-caught
            # the failure the test is about
            errors.append(e)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (not errors and asr.calls == 1 and cache.misses == 1 and
            cache.hits == 7)
    assert not list((tmp_path / "cache").rglob("*.tmp"))
