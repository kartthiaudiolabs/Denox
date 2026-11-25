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
"""The set validator on a tiny hand-made set.

A sound set passes; tampering is caught.
"""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

import scoreevents
import validateset

# pylint: disable=protected-access

SR = 48000


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _make_set(root, n_per_cond=3, wild=0, wild_policy="retention"):
    """Builds a small set of two additive conditions over two languages.

    White-noise 'speech' at -26 dBFS, plus one reference-free real item; a
    public-style manifest without provenance fields. ``wild`` adds that many
    in-the-wild items: a band-limited reference with a floor of its own under a
    bed.
    """
    os.makedirs(os.path.join(root, "mix"))
    os.makedirs(os.path.join(root, "ref"))
    rng = np.random.default_rng(0)
    rows, files = [], {}
    for i in range(wild):
        speech = rng.standard_normal(6 * SR).astype(np.float32)
        spec = np.fft.rfft(speech)
        # a phone microphone: nothing above 6 kHz
        spec[int(6000 * len(speech) / SR):] = 0
        speech = np.fft.irfft(spec, n=len(speech)).astype(np.float32)
        # its own noise floor, about 14 dB under the speech
        floor = np.sqrt((speech**2).mean())
        noise = rng.standard_normal(6 * SR).astype(np.float32)
        speech = speech + 0.2 * noise * floor
        speech *= 10**(-26 / 20) / np.sqrt((speech**2).mean())
        bed = rng.standard_normal(6 * SR).astype(np.float32)
        bed *= (np.sqrt((speech**2).mean()) / np.sqrt(
            (bed**2).mean()) / 10**(-5 / 20))
        base = f"wildAT-5_{i:03d}.wav"
        sf.write(os.path.join(root, "mix", base),
                 speech + bed,
                 SR,
                 subtype="FLOAT")
        sf.write(os.path.join(root, "ref", base), speech, SR, subtype="FLOAT")
        row = dict(
            id=base,
            condition="wild@-5",
            family="wild",
            language=("en", "de")[i % 2],
            speaker=f"wild{i}",
            mix_path=f"mix/{base}",
            clean_path=f"ref/{base}",
            snr=-5.0,
            score_from=0,
            effective_bandwidth_hz=6000,
        )
        if wild_policy:
            row["score_policy"] = wild_policy
        rows.append(row)
    for cond, snr in (("babble@+4", 4.0), ("ambience@-5", -5.0)):
        for i in range(n_per_cond):
            speech = rng.standard_normal(6 * SR).astype(np.float32)
            speech *= 10**(-26 / 20) / np.sqrt((speech**2).mean())
            noise = rng.standard_normal(6 * SR).astype(np.float32)
            noise *= (np.sqrt((speech**2).mean()) / np.sqrt(
                (noise**2).mean()) / 10**(snr / 20))
            base = f"{cond}_{i:03d}.wav".replace("@", "AT").replace("+", "p")
            sf.write(
                os.path.join(root, "mix", base),
                speech + noise,
                SR,
                subtype="FLOAT",
            )
            sf.write(os.path.join(root, "ref", base),
                     speech,
                     SR,
                     subtype="FLOAT")
            rows.append(
                dict(
                    id=base,
                    condition=cond,
                    family=cond.split("@", maxsplit=1)[0],
                    language=("en", "de")[i % 2],
                    speaker=f"spk{i}",
                    mix_path=f"mix/{base}",
                    clean_path=f"ref/{base}",
                    snr=snr,
                    score_from=0,
                    effective_bandwidth_hz=22000,
                ))
    real = "real_field_000.wav"
    sf.write(
        os.path.join(root, "mix", real),
        (rng.standard_normal(6 * SR) * 0.05).astype(np.float32),
        SR,
        subtype="FLOAT",
    )
    rows.append(
        dict(
            id=real,
            condition="real_field",
            family="real",
            language="und",
            speaker=None,
            mix_path=f"mix/{real}",
            clean_path=None,
            snr=None,
            score_from=0,
            effective_bandwidth_hz=None,
        ))
    with open(os.path.join(root, "manifest.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for r in rows:
        for k in ("mix_path", "clean_path"):
            if r.get(k):
                files[r[k]] = _sha256(os.path.join(root, r[k]))
    listing = "".join(f"{k}:{v}\n" for k, v in sorted(files.items()))
    manifest_sha = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["id"]):
        public = {k: r.get(k) for k in validateset.PUBLIC_FIELDS}
        manifest_sha.update(json.dumps(public, sort_keys=True).encode())
        manifest_sha.update(b"\n")
    manifest_sha = manifest_sha.hexdigest()
    id_text = listing + f"manifest:{manifest_sha}\n"
    set_id = hashlib.sha256(id_text.encode()).hexdigest()
    json.dump(
        {
            "set_id": set_id,
            "n_items": len(rows),
            "manifest_sha256": manifest_sha,
            "files": files,
        },
        open(os.path.join(root, "checksums.json"), "w", encoding="utf-8"),
    )
    return rows


def _statuses(rep):
    return {name: status for name, status, _ in rep.rows}


def test_a_sound_set_validates_with_only_the_expected_warnings(tmp_path):
    root = str(tmp_path / "set")
    _make_set(root)
    rep = validateset.validate(root, full=True)
    st = _statuses(rep)
    assert rep.ok, [r for r in rep.rows if r[1] == "FAIL"]
    assert st["realised SNR within tolerance"] == "PASS"
    assert st["references at speech level"] == "PASS"
    assert st["set id reproduces from the hashes"] == "PASS"
    # three items per condition
    assert st["conditions at the leaderboard minimum"] == "WARN"
    # the set carries no license contract to check
    assert not any("licen" in name.lower() for name in st)
    assert st["language balance (paired items)"] == "PASS"
    # never a FAIL
    assert st["references clean (speech over their own floor)"] in (
        "PASS",
        "WARN",
    )


def test_language_balance_allows_one_first_language_but_not_unbalanced_others(
    tmp_path,):
    root = str(tmp_path / "set")
    _make_set(root, n_per_cond=6)
    rows = [
        json.loads(l)
        for l in open(os.path.join(root, "manifest.jsonl"), encoding="utf-8")
    ]
    paired = [r for r in rows if r.get("clean_path")]
    # 8 en, 2 de, 2 fr: a first language, balanced others
    for i, r in enumerate(paired):
        r["language"] = "en" if i < 8 else ("de", "fr")[i % 2]
    with open(os.path.join(root, "manifest.jsonl"), "w",
              encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    rep = validateset.validate(root, full=False)
    st = dict((r[0], r) for r in rep.rows)["language balance (paired items)"]
    assert st[1] == "PASS" and st[2].startswith("en 67%")
    # 8 en, 3 de, 1 fr: the others are not balanced
    for i, r in enumerate(paired):
        r["language"] = "en" if i < 8 else ("de" if i < 11 else "fr")
    with open(os.path.join(root, "manifest.jsonl"), "w",
              encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    rep = validateset.validate(root, full=False)
    assert _statuses(rep)["language balance (paired items)"] == "WARN"


def test_tampering_with_files_or_the_set_id_fails(tmp_path):
    root = str(tmp_path / "set")
    rows = _make_set(root)
    victim = rows[0]
    x, sr = sf.read(os.path.join(root, victim["mix_path"]), dtype="float32")
    sf.write(os.path.join(root, victim["mix_path"]),
             x * 0.2,
             sr,
             subtype="FLOAT")  # silently re-levelled
    os.remove(os.path.join(root, rows[-1]["mix_path"]))  # and one file lost
    rep = validateset.validate(root, full=True)
    st = _statuses(rep)
    assert not rep.ok
    assert st["file hashes match"] == "FAIL" and st["files exist"] == "FAIL"
    # the re-levelled mix no longer matches its label
    assert st["realised SNR within tolerance"] == "FAIL"
    ck = json.load(open(os.path.join(root, "checksums.json"), encoding="utf-8"))
    ck["set_id"] = "0" * 64
    json.dump(ck,
              open(os.path.join(root, "checksums.json"), "w", encoding="utf-8"))
    st = _statuses(validateset.validate(root, full=True))
    assert st["set id reproduces from the hashes"] == "FAIL"


def test_changing_public_item_semantics_breaks_the_set_id(tmp_path):
    root = str(tmp_path / "set")
    _make_set(root)
    st = _statuses(validateset.validate(root, full=False))
    assert st["manifest semantics match checksums"] == "PASS"
    rows = [
        json.loads(l)
        for l in open(os.path.join(root, "manifest.jsonl"), encoding="utf-8")
    ]
    rows[0]["score_from"] = 4800  # same audio, different scored region
    with open(os.path.join(root, "manifest.jsonl"), "w",
              encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    st = _statuses(validateset.validate(root, full=False))
    assert st["manifest semantics match checksums"] == "FAIL"


def test_inventory_check_is_opt_in_and_names_missing_conditions(tmp_path):
    root = str(tmp_path / "set")
    _make_set(root)
    assert "condition inventory complete" not in _statuses(
        validateset.validate(root, full=False))
    st = _statuses(validateset.validate(root, full=False,
                                        expect_inventory=True))
    # the hand-made set has three conditions only
    assert st["condition inventory complete"] == "FAIL"


def test_retention_items_are_exempt_from_the_clean_reference_gates(tmp_path):
    root = str(tmp_path / "wild")
    _make_set(root, wild=2)
    rep = validateset.validate(root, full=True)
    st = _statuses(rep)
    assert rep.ok, [r for r in rep.rows if r[1] == "FAIL"]
    # the wild reference is not gated
    assert st["speech bandwidth field >= 20 kHz"] == "PASS"
    # the bed's SNR is still checked
    assert st["realised SNR within tolerance"] == "PASS"
    assert not [
        f for f in rep.failing["references clean (speech over their own floor)"]
        if f.startswith("wild")
    ]
    # the same audio without the policy is an ordinary paired item, and then it
    # fails the gate
    root2 = str(tmp_path / "plain")
    _make_set(root2, wild=2, wild_policy=None)
    rep2 = validateset.validate(root2, full=True)
    assert _statuses(rep2)["speech bandwidth field >= 20 kHz"] == "FAIL"


def test_golden_transcripts_are_checked_against_the_manifest_when_present(
    tmp_path,):
    root = str(tmp_path / "set")
    _make_set(root)
    rows = [
        json.loads(l)
        for l in open(os.path.join(root, "manifest.jsonl"), encoding="utf-8")
    ]
    paired = [r for r in rows if r.get("clean_path")]
    real = [r for r in rows if not r.get("clean_path")]
    tpath = os.path.join(root, "transcripts.jsonl")
    with open(tpath, "w", encoding="utf-8") as fh:
        for r in paired:
            fh.write(
                json.dumps({
                    "id": r["id"],
                    "language": r["language"],
                    "text": "some words",
                    "tier": "corpus-verified",
                }) + "\n")
    rep = validateset.validate(root, full=False)
    assert _statuses(rep)["golden transcripts consistent"] == "PASS"
    lines = open(tpath, encoding="utf-8").read().splitlines()
    for spoken, status in (
        ("en" if paired[0]["language"] != "en" else "de", "PASS"),
        (paired[0]["language"], "FAIL"),
        ("english", "FAIL"),
        (7, "FAIL"),
    ):
        first = dict(
            json.loads(lines[0]),
            # the language heard, where the label is wrong
            spoken=spoken,
        )
        with open(tpath, "w", encoding="utf-8") as fh:
            fh.write("\n".join([json.dumps(first)] + lines[1:]) + "\n")
        rep = validateset.validate(root, full=False)
        assert _statuses(rep)["golden transcripts consistent"] == status
    with open(tpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(tpath, "a",
              encoding="utf-8") as fh:  # every kind of bad row, one each
        fh.write(
            json.dumps({
                "id": paired[0]["id"],
                "language": paired[0]["language"],
                "text": "again",
                "tier": "t",
            }) + "\n")  # duplicate
        fh.write(
            json.dumps({
                "id": "ghost.wav",
                "language": "en",
                "text": "x",
                "tier": "t"
            }) + "\n")
        fh.write(
            json.dumps({
                "id": real[0]["id"],
                "language": real[0].get("language"),
                "text": "x",
                "tier": "t",
            }) + "\n")  # a reference-free item has no golden text
        fh.write(
            json.dumps({
                "id": paired[1]["id"],
                "language": "xx",
                "text": "  ",
                "tier": "",
            }) + "\n")  # wrong language, empty text, no tier (and a duplicate)
    rep = validateset.validate(root, full=False)
    assert (_statuses(rep)["golden transcripts consistent"] == "FAIL" and
            not rep.ok)
    flagged = " ".join(rep.failing["golden transcripts consistent"])
    for needle in (
            "duplicate",
            "not in the manifest",
            "reference-free",
            "empty text",
            "language xx",
            "no tier",
    ):
        assert needle in flagged, needle


@pytest.mark.parametrize("across_conditions", [False, True])
def test_quick_validation_rejects_duplicate_inputs(tmp_path, across_conditions):
    """A renamed copy fails even outside the sampled audio or condition."""
    root = str(tmp_path / "set")
    rows = _make_set(root)
    first, second = rows[0], rows[3 if across_conditions else 1]
    shutil.copyfile(os.path.join(root, first["mix_path"]),
                    os.path.join(root, second["mix_path"]))
    report = validateset.validate(root, full=False, sample=0.01)
    assert _statuses(report)["unique input waveforms"] == "FAIL"
    flagged = " ".join(report.failing["unique input waveforms"])
    assert first["id"] in flagged and second["id"] in flagged


def test_audio_identity_ignores_container_and_signed_zero(tmp_path):
    """Equivalent decoded samples collide across PCM and floating WAVs."""
    pcm = tmp_path / "pcm.wav"
    floating = tmp_path / "floating.wav"
    samples = np.array([0.0, 0.125, -0.25, 0.0])
    sf.write(pcm, samples, SR, subtype="PCM_16")
    samples[-1] = -0.0
    sf.write(floating, samples, SR, subtype="DOUBLE")
    assert _sha256(pcm) != _sha256(floating)
    assert validateset.audio_sha256(pcm) == validateset.audio_sha256(floating)


def test_audio_identity_preserves_precision_and_layout(tmp_path):
    """Distinct PCM32 samples and different timing must not collide."""
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    sf.write(first, np.array([2**28, 0], dtype=np.int32), SR, subtype="PCM_32")
    sf.write(second,
             np.array([2**28 + 1, 0], dtype=np.int32),
             SR,
             subtype="PCM_32")
    assert validateset.audio_sha256(first) != validateset.audio_sha256(second)
    samples, _ = sf.read(first)
    sf.write(second, samples, SR // 2, subtype="DOUBLE")
    assert validateset.audio_sha256(first) != validateset.audio_sha256(second)


def test_reference_reuse_does_not_make_inputs_duplicates(tmp_path):
    """The same target under different noise remains a valid pair of inputs."""
    root = str(tmp_path / "set")
    rows = _make_set(root)
    first = rows[0]
    second = rows[1]
    shutil.copyfile(os.path.join(root, first["clean_path"]),
                    os.path.join(root, second["clean_path"]))
    report = validateset.validate(root, full=True)
    assert _statuses(report)["unique input waveforms"] == "PASS"


def test_the_same_room_block_must_carry_its_labels(tmp_path):
    root = str(tmp_path / "set")
    _make_set(root)
    path = os.path.join(root, "manifest.jsonl")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows[0]["condition"], rows[0]["family"] = "coroom@pause", "coroom"

    def write():
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))

    def status():
        return _statuses(validateset.validate(root, full=False))

    write()
    assert status()["same-room labels"] == "FAIL"
    rows[0]["provenance"] = {
        "events": [[1.0, 1.1, "pause", "household", "x", -15.0]],
        "pauses": [[0.5, 1.5]],
        "chain": {
            "tilt_db_per_octave": 0.0,
            "peak_hz": 3000,
            "peak_db": 0.0
        },
        "split": "main",
        "room_share_db": 1.2,
        "room_share_rule": scoreevents.ROOM_SHARE_RULE,
    }
    write()
    assert status()["same-room labels"] == "PASS"
    rows[0]["provenance"]["room_share_db"] = "1.2"  # a number, not a string
    write()
    assert status()["same-room labels"] == "FAIL"
    # shares measured under another rule than the scorer's are refused
    rows[0]["provenance"]["room_share_rule"] = "another rule, v1"
    write()
    assert status()["same-room allowance rule"] == "FAIL"
