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
"""Double-processing check: second-pass cost, idempotence and determinism.

The fakes have a known second-pass cost.
"""

# pylint: disable=redefined-outer-name,unused-argument

import json
import math

import pytest
import soundfile as sf
import torch

import common
import twicetest

SR = common.SR
HALF_DB = 20 * math.log10(0.5)


def speech(seconds, seed=0, level=0.2):
    n = int(seconds * SR)
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    env = 0.2 + 0.8 * (0.5 + 0.5 * torch.sin(2 * math.pi * 2.0 * t)).pow(2)
    return torch.randn(n, generator=g) * env * level


class _Spk:

    def sim(self, a, b, key_b=None):
        n = min(len(a), len(b))
        return float(torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0))


def test_diff_db_floor_and_level():
    x = speech(1.0)
    assert twicetest.diff_db(x, x) == twicetest.DIFF_FLOOR_DB
    assert twicetest.diff_db(0.5 * x, x) == pytest.approx(HALF_DB, abs=0.01)


def test_identity_costs_nothing_and_is_idempotent_and_deterministic():
    x = speech(3.0)
    d = twicetest.measure_item(lambda a: a, x, x, spk=_Spk())
    assert (abs(d["pass1_kept_coh_mean_db"]) < 0.01 and
            abs(d["second_pass_kept_delta_db"]) < 0.01)
    assert (d["idempotence_db"] == twicetest.DIFF_FLOOR_DB and
            d["determinism_db"] == twicetest.DIFF_FLOOR_DB)
    assert (d["pass1_spk_sim"] == pytest.approx(1.0, abs=1e-4) and
            d["second_pass_spk_delta"] == 0.0)


def test_an_attenuator_loses_the_same_again_on_the_second_pass():
    x = speech(3.0, seed=1)
    d = twicetest.measure_item(lambda a: 0.5 * a, x, x)
    assert d["pass1_kept_coh_mean_db"] == pytest.approx(HALF_DB, abs=0.02)
    assert d["pass2_kept_coh_mean_db"] == pytest.approx(2 * HALF_DB, abs=0.02)
    assert d["second_pass_kept_delta_db"] == pytest.approx(HALF_DB, abs=0.02)
    # |0.25x - 0.5x| / |0.5x|
    assert d["idempotence_db"] == pytest.approx(HALF_DB, abs=0.02)
    assert d["determinism_db"] == twicetest.DIFF_FLOOR_DB


def test_a_soft_gate_that_erodes_quiet_blocks_costs_more_the_second_time():
    x = speech(4.0, seed=2)
    frame = 4800
    blocks = x[:len(x) // frame * frame].reshape(-1, frame)
    thr = 0.5 * float(blocks.pow(2).mean(1).median())

    # blocks under a fixed threshold are turned down in proportion: each pass
    # finds them quieter than the last and turns them down further
    def gate(a):
        y = a.clone()
        e = a[:len(a) // frame * frame].reshape(-1, frame).pow(2).mean(1)
        for b, g in enumerate((e / thr).clamp(max=1.0).tolist()):
            y[b * frame:(b + 1) * frame] *= g
        return y

    d = twicetest.measure_item(gate, x, x)
    # the quiet blocks lost something after one pass
    assert d["pass1_kept_coh_p10_db"] < -3
    assert d["second_pass_p10_delta_db"] < -3  # and more after two
    assert d["second_pass_kept_delta_db"] < -0.5
    assert d["idempotence_db"] > -30  # the second pass rewrote something


def test_a_nondeterministic_system_reads_its_own_noise_level():

    # the global generator advances: every call draws different noise
    def noisy(a):
        return a + torch.randn(len(a)) * float(
            a.pow(2).mean().sqrt()) * 10**(-20 / 20)

    x = speech(3.0, seed=3)
    d = twicetest.measure_item(noisy, x, x)
    # two draws: +3 dB
    assert d["determinism_db"] == pytest.approx(-20 + 3, abs=1.5)
    assert d["determinism_db"] > -40


def test_failures_are_named_by_pass_and_counted():
    x = speech(1.0)
    calls = []

    def flaky(a):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return a

    d = twicetest.measure_item(flaky, x, x)
    assert d == {"failure": "pass 2 raised: boom"}
    short = twicetest.measure_item(lambda a: a[:-1], x, x)
    assert short["failure"].startswith("pass 1 length")
    nan = torch.full((len(x),), float("nan"))
    nonfinite = twicetest.measure_item(lambda a: nan, x, x)
    assert nonfinite["failure"] == "pass 1 nan"
    passed = {"idempotence_db": -50.0, "determinism_db": -90.0}
    s = twicetest.aggregate([d, passed])
    assert (s["n_items"] == 2 and s["n_failed"] == 1 and
            s["failures"] == ["pass 2 raised: boom"])
    assert s["idempotence_db"] == -50.0 and s["determinism_max_db"] == -90.0


@pytest.fixture
def paired_set(tmp_path, monkeypatch):
    root = tmp_path / "set"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    rows = []
    for cond, count in (("babble@+4", 2), ("real_x", 1)):
        for i in range(count):
            x = speech(2.0, seed=20 + i)
            rid = f"{cond.replace('@', 'AT')}_{i:03d}.wav"
            sf.write(str(root / "mix" / rid), x.numpy(), SR, subtype="FLOAT")
            row = {
                "id": rid,
                "condition": cond,
                "mix_path": f"mix/{rid}",
                "clean_path": None,
            }
            if cond != "real_x":
                sf.write(str(root / "ref" / rid),
                         x.numpy(),
                         SR,
                         subtype="FLOAT")
                row["clean_path"] = f"ref/{rid}"
            rows.append(row)
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
    return root


def test_main_scores_paired_items_only_with_the_tiny_model(
        identity_model, paired_set, tmp_path, monkeypatch):
    # pylint: disable-next=import-outside-toplevel
    from inworld.voice_isolation import isolator

    vi = isolator.VoiceIsolator(identity_model, device="cpu")
    monkeypatch.setattr(
        common,
        "isolator_runner",
        lambda a: (vi, lambda x, c=None, o=None: vi.isolate(x, SR)),
    )
    out = tmp_path / "twice.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "twicetest.py",
            "--model",
            "tiny",
            "--conds",
            "babble@+4,real_x",
            "--no-speaker",
            "--out",
            str(out),
        ],
    )
    twicetest.main()
    r = json.load(open(out, encoding="utf-8"))
    assert r["set_id"] == "tiny" and r["model"] == "tiny"
    assert r["conditions"]["babble@+4"]["summary"]["n_items"] == 2
    # no reference: nothing to read
    assert r["conditions"]["real_x"]["summary"]["n_items"] == 0
    s = r["conditions"]["babble@+4"]["summary"]
    # identity model
    assert s["n_failed"] == 0 and abs(s["second_pass_kept_delta_db"]) < 0.5
    assert "pass1_spk_sim" not in s
