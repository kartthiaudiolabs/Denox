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
"""The streaming probe: look-ahead at several splices and long-input drift.

Read on renders of known behaviour, bound to the probe they were cut from and to
the audio the board scored.
"""

# pylint: disable=redefined-outer-name,unused-argument

import json
import os

import numpy as np
import pytest
import soundfile as sf
import torch

import common
import observations
import receipt
import scorebench
import scorestream

SR = common.SR
SECONDS = 2.5  # every long item of the tiny set
N_ITEMS = 3


def _speech(seed, seconds=SECONDS, level=0.2):
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * SR)
    t = torch.arange(n) / SR
    env = 0.3 + 0.7 * (0.5 + 0.5 * torch.sin(2 * np.pi * 3.0 * t)).pow(2)
    return (torch.randn(n, generator=g) * env * level).numpy().astype("float32")


def _write(path, x):
    sf.write(str(path), np.asarray(x, dtype="float32"), SR, subtype="FLOAT")


# fake systems: a function of the whole input it is handed, as a wrapper would
# be
def echo(x):
    return x.copy()


def lookahead(x, ms=30):
    """Each sample is the mean of the next ``ms`` of input.

    It is causal, with a known look-ahead.
    """
    k = SR * ms // 1000
    c = np.concatenate([[0.0], np.cumsum(np.pad(x, (0, k)))])
    return ((c[k:k + len(x)] - c[:len(x)]) / k).astype("float32")


def offline(x):
    """Scales the whole file by the level of its last 300 ms.

    Every sample depends on the end.
    """
    tail = x[-int(0.3 * SR):]
    return (x / (np.sqrt((tail**2).mean()) + 1e-9) * 0.1).astype("float32")


def blocks(x, seconds=0.5):
    """Normalises independent half-second blocks.

    Non-causal inside a block, and invisible to a splice that falls on a block
    boundary.
    """
    n = int(seconds * SR)
    y = x.copy()
    for i in range(0, len(x), n):
        seg = x[i:i + n]
        y[i:i + n] = seg / (np.sqrt((seg**2).mean()) + 1e-9) * 0.1
    return y.astype("float32")


def drifting(x):
    """Adds noise that grows with the time since its own start.

    From half a second on: state that goes bad, and a fresh start that is the
    better one.
    """
    g = np.random.default_rng(len(x))
    t = np.arange(len(x)) / SR
    ramp = np.maximum(0.0, t - 0.5)
    return (x + 0.3 * ramp * g.standard_normal(len(x))).astype("float32")


def gating(x, after=1.25):
    """Goes silent ``after`` seconds from its own start.

    Speech is dropped once the state is old.
    """
    y = x.copy()
    y[int(after * SR):] = 0.0
    return y


def muting(x):
    """Silences the last 0.6 s of every item whatever came before.

    It is a loss with no context in it.
    """
    y = x.copy()
    y[-int(0.6 * SR):] = 0.0
    return y


SYSTEMS = {
    "echo": echo,
    "ahead": lookahead,
    "offline": offline,
    "blocks": blocks,
    "drifty": drifting,
    "gater": gating,
    "muter": muting,
}


@pytest.fixture
def small_rule(monkeypatch):
    """Returns the probe geometry scaled to short items.

    A one-second stretch from 1.2 s after a 0.4 s lead-in (the cut starts at 0.8
    s), with splices into the stretch that avoid the half-second grid of the
    cut, and one on it (0.3 into the stretch is 0.7 into the cut; 0.1 is 0.5).
    """
    monkeypatch.setattr(scorestream, "GRID_SECONDS", 0.4)
    monkeypatch.setattr(scorestream, "TAIL_START", 1.2)
    monkeypatch.setattr(scorestream, "LEAD_IN", 0.4)
    monkeypatch.setattr(scorestream, "TAIL_SECONDS", 1.0)
    monkeypatch.setattr(scorestream, "SPLICES", (0.3, 0.1, 0.7))
    monkeypatch.setattr(scorestream, "COLD_START_SECONDS", 0.2)
    monkeypatch.setattr(scorestream, "DROP_DB", 10.0)
    monkeypatch.setattr(scorestream, "MIN_ITEMS", 2)


def _set(tmp_path, monkeypatch, n_items=N_ITEMS):
    root = tmp_path / "set"
    out_root = tmp_path / "out"
    results = tmp_path / "results"
    for d in (root / "mix", root / "ref", results):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_items):
        rid = f"longATp4_{i:03d}.wav"
        ref = _speech(i)
        noise = np.random.default_rng(100 + i).standard_normal(len(ref))
        mix = ref + 0.05 * noise.astype("float32")
        _write(root / "ref" / rid, ref)
        _write(root / "mix" / rid, mix)
        rows.append({
            "id": rid,
            "condition": "long@+4",
            "mix_path": f"mix/{rid}",
            "clean_path": f"ref/{rid}",
        })
    rows.append({
        "id": "clean_000.wav",
        "condition": "clean",
        "mix_path": "mix/longATp4_000.wav",
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
    monkeypatch.setattr(common, "RESULTS", str(results))
    return root, out_root, results


def _render(name, fn, manifest, probe, probe_root, out_root, results):
    """Leaves what a wrapper run or runsystem.py leaves behind.

    The whole renders scored into records, the board's stamp, and the probe
    outputs with the receipt of the same environment.
    """
    (out_root / name).mkdir(parents=True, exist_ok=True)
    records = []
    for r in scorestream.source_rows(manifest):
        _write(out_root / name / r["id"],
               fn(common.load48(r["mix_path"]).numpy()))
        rec = observations.item(r)
        output = common.load48(str(out_root / name / r["id"]))
        rec["identity"] = {"output": scorebench.sample_hash(output)}
        records.append(rec)
    observations.write(
        results / f"{name}.items.jsonl",
        records,
        {
            "system": name,
            "set_id": "tiny",
            "subset": "all",
            "n_expected": len(records),
        },
    )
    pout = probe_root / "out" / name
    pout.mkdir(parents=True, exist_ok=True)
    environment = {"system": name, "wrapper_sha256": name * 4}
    board_ident = dict(environment, inputs={"set_id": "tiny"})
    board = {
        "name": name,
        "set_id": "tiny",
        "render_receipt_sha256": receipt.digest(board_ident),
        "render_environment_sha256": receipt.environment_digest(board_ident),
    }
    receipt.write(str(pout),
                  dict(environment, inputs={"probe": probe["identity"]}))
    json.dump(board, open(results / f"{name}.meta.json", "w", encoding="utf-8"))
    for entry in probe["items"]:
        for pname in [entry["tail"]] + [f["name"] for f in entry["futures"]]:
            _write(
                pout / pname,
                fn(common.load48(str(probe_root / "in" / pname)).numpy()),
            )


def _receipt_only(out_dir, ident):
    """Replaces the receipts in ``out_dir`` with one of ``ident``."""
    for name in os.listdir(out_dir):
        if name.startswith(".receipt-"):
            os.remove(os.path.join(out_dir, name))
    receipt.write(out_dir, ident)


@pytest.fixture
def probe_set(tmp_path, monkeypatch, small_rule):
    """Returns a tiny set whose long items every fake system rendered whole.

    Each was scored on it, and the probe cut from it was rendered by the same
    fakes.
    """
    root, out_root, results = _set(tmp_path, monkeypatch)
    manifest = common.manifest(str(root))
    probe_root = tmp_path / "probe"
    probe = scorestream.write_probe(manifest, str(probe_root))
    for name, fn in SYSTEMS.items():
        _render(name, fn, manifest, probe, probe_root, out_root, results)
    _render("ours", echo, manifest, probe, probe_root, out_root, results)
    return {
        "rows": manifest,
        "probe": probe,
        "root": str(probe_root),
        "results": str(results),
    }


def test_probe_cuts_a_lead_in_a_stretch_and_three_futures_per_item(probe_set):
    probe, root, rows = probe_set["probe"], probe_set["root"], probe_set["rows"]
    assert (probe["set_id"] == "tiny" and
            probe["probe_version"] == scorestream.PROBE_VERSION)
    ids = [e["id"] for e in probe["items"]]
    assert ids == [r["id"] for r in scorestream.source_rows(rows)]
    assert probe["identity"] == scorestream.identity_of(
        scorestream.probe_hashes(probe))
    assert probe["cut_seconds"] == 0.8 and probe["lead_in_seconds"] == 0.4
    first = probe["items"][0]
    # on the 0.4 s grid
    assert first["cut_seconds"] == 0.8 and first["stretch_seconds"] == 1.2
    tail = common.load48(os.path.join(root, "in", first["tail"])).numpy()
    mix = common.load48(rows[0]["mix_path"]).numpy()
    a, k, b = scorestream.cut_bounds()
    assert (a, k, b) == (int(0.8 * SR), int(1.2 * SR), int(2.2 * SR))
    assert len(tail) == b - a and np.array_equal(tail, mix[a:b])
    other = common.load48(rows[1]["mix_path"]).numpy()
    splices = [f["splice_seconds"] for f in first["futures"]]
    assert splices == pytest.approx([0.7, 0.5, 1.1])
    for f in first["futures"]:
        future = common.load48(os.path.join(root, "in", f["name"])).numpy()
        j = int(round(f["splice_seconds"] * SR))
        assert np.array_equal(future[:j], tail[:j]) and np.array_equal(
            future[j:], other[a:b - j])
    assert scorestream.read_probe(root, rows)["items"] == probe["items"]


def test_probe_is_rewritten_only_if_stale_and_refuses_what_it_cannot_vouch_for(
        probe_set, monkeypatch):
    root, rows, probe = probe_set["root"], probe_set["rows"], probe_set["probe"]
    assert scorestream.probe_problem(root, rows) is None
    before = os.path.getmtime(os.path.join(root, "probe.json"))
    # current: left alone, renders stay valid
    assert scorestream.write_probe(rows, root) == probe
    assert os.path.getmtime(os.path.join(root, "probe.json")) == before
    path = os.path.join(root, "in", probe["items"][0]["tail"])
    x = common.load48(path).numpy()
    _write(path, x * 0.5)
    assert "not the recorded probe input" in scorestream.probe_problem(
        root, rows)
    with pytest.raises(SystemExit, match="write the probe again"):
        scorestream.read_probe(root, rows)
    # the same samples again: current, whatever the file's bytes say
    _write(path, x)
    assert scorestream.probe_problem(root, rows) is None
    stray = os.path.join(root, "in", "leftover.wav")
    _write(stray, x)
    assert "not probe inputs" in scorestream.probe_problem(root, rows)
    # rewritten without it
    assert scorestream.write_probe(rows, root)["identity"] == probe["identity"]
    assert (not os.path.exists(stray) and
            scorestream.probe_problem(root, rows) is None)
    # a new verdict rule re-reads, never re-cuts
    monkeypatch.setattr(scorestream, "RULE_VERSION", "99")
    assert scorestream.probe_problem(root, rows) is None
    monkeypatch.setattr(scorestream, "PROBE_VERSION", "99")
    assert "probe version" in scorestream.probe_problem(root, rows)
    monkeypatch.setattr(scorestream, "PROBE_VERSION", probe["probe_version"])
    fewer = [r for r in rows if r["id"] != probe["items"][-1]["id"]]
    assert "complete inventory" in scorestream.probe_problem(root, fewer)
    monkeypatch.setattr(common, "set_id", lambda root=None: "other")
    assert "cut from set" in scorestream.probe_problem(root, rows)


def test_probe_needs_enough_long_items_of_full_length(tmp_path, monkeypatch,
                                                      small_rule):
    rows = [{
        "id": "a.wav",
        "condition": "long@+4",
        "clean_path": "x",
        "mix_path": "x",
    }]
    with pytest.raises(SystemExit, match="needs 2"):
        scorestream.write_probe(rows, str(tmp_path))
    short = tmp_path / "short.wav"
    _write(short, _speech(0, seconds=2.0))
    rows = [{
        "id": f"{i}.wav",
        "condition": "long@+4",
        "clean_path": str(short),
        "mix_path": str(short),
    } for i in range(2)]
    with pytest.raises(SystemExit, match="shorter than"):
        scorestream.write_probe(rows, str(tmp_path))


def test_look_ahead_is_how_far_before_the_splice_the_output_differs(small_rule):
    x = _speech(7)
    y = x.copy()
    k = int(round(0.7 * SR))
    y[k:] = _speech(8)[k:]
    assert scorestream.look_ahead_ms(echo(x), echo(y), 0.7) == (0, False)
    ahead_x, ahead_y = lookahead(x, 30), lookahead(y, 30)
    ms, capped = scorestream.look_ahead_ms(ahead_x, ahead_y, 0.7)
    assert 20 <= ms <= 40 and not capped
    assert scorestream.look_ahead_ms(offline(x), offline(y), 0.7) == (700, True)
    # a block normaliser passes a splice on its grid and is caught by one off it
    z = x.copy()
    z[int(0.5 * SR):] = _speech(8)[int(0.5 * SR):]
    assert scorestream.look_ahead_ms(blocks(x), blocks(z), 0.5) == (0, False)
    ms, capped = scorestream.look_ahead_ms(blocks(x), blocks(y), 0.7)
    assert 190 <= ms <= 210 and not capped
    z = np.zeros_like(x)  # silence that stays silent is not a difference
    assert scorestream.look_ahead_ms(z, z, 0.7) == (0, False)


def test_verdict_needs_a_bounded_look_ahead_no_drift_and_no_dropped_speech():
    good = [{
        "look_ahead_ms": 20,
        "look_ahead_capped": False,
        "drift_db": 0.1,
        "cold_start_db": -0.4,
        "dropped_seconds": 0,
        "speech_seconds": 12,
        "context_diff_db": -40,
    }] * 3
    v = scorestream.verdict(good)
    assert (v["verdict"] == "yes" and v["causal"] and v["holds"] and
            not v["reasons"])
    assert (v["cold_start_db"] == pytest.approx(-0.4) and
            v["speech_seconds"] == 36)
    assert not v["path_switch"]
    far = good + [
        dict(
            good[0],
            look_ahead_ms=scorestream.LOOK_AHEAD_MAX_MS + 10,
            look_ahead_capped=True,
        )
    ]
    v = scorestream.verdict(far)
    assert v["verdict"] == "no" and not v["causal"] and v["look_ahead_capped"]
    assert "at least 110 ms of future input" in v["reasons"][0]
    drifts = good + [dict(good[0], drift_db=scorestream.DRIFT_MIN_DB - 1)]
    v = scorestream.verdict(drifts)
    assert v["verdict"] == "no" and v["causal"] and not v["holds"]
    assert "drifts over a long input" in v["reasons"][0]
    slow = [dict(good[0], drift_db=scorestream.DRIFT_MEAN_DB - 0.5)] * 3
    assert not scorestream.verdict(slow)["holds"]
    gated = good + [dict(good[0], dropped_seconds=3, path_switch=True)]
    v = scorestream.verdict(gated)
    assert (v["verdict"] == "no" and v["causal"] and not v["holds"] and
            v["path_switch"])
    assert v["dropped_seconds"] == 3 and v["dropout_items"] == 1
    assert (
        "drops speech after a long context: 3 of 48 speech seconds on 1 of 4"
        " items" in v["reasons"][0])
    assert scorestream.look_ahead_text(v) == "20 (max 20)"
    assert (scorestream.look_ahead_text(
        scorestream.verdict(far)) == "20 (max ≥110)"
           )  # only the maximum reached the start
    most = [dict(good[0], look_ahead_ms=700, look_ahead_capped=True)] * 3 + [
        good[0]
    ]
    assert scorestream.look_ahead_text(
        scorestream.verdict(most)) == "≥700 (max ≥700)"
    one_short = good + [
        dict(good[0], look_ahead_ms=300, look_ahead_capped=False),
        dict(good[0], look_ahead_ms=200, look_ahead_capped=True),
    ]
    # the 300 ms one was measured
    assert scorestream.look_ahead_text(
        scorestream.verdict(one_short)) == "20 (max 300)"


def test_measure_reads_the_fakes_as_built(probe_set):
    rows = probe_set["rows"]
    probe = probe_set["probe"]
    root = probe_set["root"]
    results = probe_set["results"]
    out = {
        n: scorestream.measure(n, rows, probe, root, results) for n in SYSTEMS
    }
    assert (out["echo"]["verdict"] == "yes" and
            out["echo"]["look_ahead_max_ms"] == 0)
    assert out["echo"]["drift_mean_db"] == pytest.approx(0.0, abs=1e-6)
    assert out["echo"]["cold_start_db"] == pytest.approx(0.0, abs=1e-6)
    assert (out["echo"]["dropped_seconds"] == 0 and
            out["echo"]["speech_seconds"] == N_ITEMS)
    assert (out["ahead"]["verdict"] == "yes" and
            20 <= out["ahead"]["look_ahead_max_ms"] <= 40)
    assert out["offline"]["verdict"] == "no" and not out["offline"]["causal"]
    assert (out["offline"]["look_ahead_capped"] and
            out["offline"]["look_ahead_max_ms"] >= 490)
    # caught off the grid
    assert out["blocks"]["verdict"] == "no" and not out["blocks"]["causal"]
    assert out["blocks"]["look_ahead_max_ms"] >= 190
    assert out["drifty"]["verdict"] == "no" and out["drifty"]["causal"]
    assert (not out["drifty"]["holds"] and
            out["drifty"]["drift_mean_db"] < scorestream.DRIFT_MEAN_DB)
    # a fresh start is the better one for it
    assert out["drifty"]["cold_start_db"] > 0
    # the muter loses the end of every stretch in both renders: lost speech, but
    # not to the context
    assert out["muter"]["dropped_seconds"] == 0
    # the gater keeps the stretch when it starts fresh and silences it after a
    # long context
    assert (out["gater"]["verdict"] == "no" and out["gater"]["causal"] and
            not out["gater"]["holds"])
    assert (out["gater"]["dropped_seconds"] == N_ITEMS and
            "drops speech" in out["gater"]["reasons"][0])
    for o in out.values():
        assert o["n_items"] == N_ITEMS and len(o["items"]) == N_ITEMS
        assert all(
            len(it["look_ahead_by_splice_ms"]) == 3 and it["output_sha256"]
            for it in o["items"])
        assert not o["path_switch"]
    assert out["echo"]["rule"]["version"] == scorestream.RULE_VERSION


def test_a_release_that_switches_inference_paths_is_disclosed(probe_set):
    rows = probe_set["rows"]
    probe = probe_set["probe"]
    root = probe_set["root"]
    results = probe_set["results"]
    meta_path = os.path.join(results, "echo.meta.json")
    board = json.load(open(meta_path, encoding="utf-8"))
    # items 2.5 s, cut 1.4 s
    json.dump(dict(board, path_switch_seconds=2.0),
              open(meta_path, "w", encoding="utf-8"))
    out = scorestream.measure("echo", rows, probe, root, results)
    assert out["path_switch"] and all(it["path_switch"] for it in out["items"])
    assert "different inference paths" in scorestream.describe(out)
    # both beyond: same path
    json.dump(dict(board, path_switch_seconds=1.0),
              open(meta_path, "w", encoding="utf-8"))
    out = scorestream.measure("echo", rows, probe, root, results)
    assert not out["path_switch"]


def test_measure_refuses_unscored_audio_and_incomplete_or_damaged_probe_outputs(
    probe_set,):
    rows = probe_set["rows"]
    probe = probe_set["probe"]
    root = probe_set["root"]
    results = probe_set["results"]
    whole = os.path.join(common.OUT_ROOT, "echo", rows[0]["id"])
    x = common.load48(whole).numpy()
    _write(whole, x * 0.9)
    with pytest.raises(SystemExit, match="not the audio that was scored"):
        scorestream.measure("echo", rows, probe, root, results)
    _write(whole, x)
    future = os.path.join(root, "out", "echo",
                          probe["items"][1]["futures"][2]["name"])
    y = common.load48(future).numpy()
    _write(future, y[:-SR // 10])  # a future output 100 ms short
    with pytest.raises(SystemExit, match="samples for an input"):
        scorestream.measure("echo", rows, probe, root, results)
    bad = y.copy()
    bad[10] = np.nan
    _write(future, bad)
    with pytest.raises(SystemExit, match="not finite"):
        scorestream.measure("echo", rows, probe, root, results)
    os.remove(future)
    with pytest.raises(SystemExit, match="must be complete"):
        scorestream.measure("echo", rows, probe, root, results)
    with pytest.raises(SystemExit, match="score nobody before"):
        scorestream.measure("nobody", rows, probe, root, results)


def test_probe_renders_must_come_from_the_boards_environment_and_this_probe(
    probe_set,):
    root = probe_set["root"]
    results = probe_set["results"]
    probe = probe_set["probe"]
    out = os.path.join(root, "out", "echo")
    environment = {"system": "echo", "wrapper_sha256": "echo" * 4}
    assert scorestream.environment_problem("echo", results, root, probe) is None
    _receipt_only(out, dict(environment, inputs={"probe": "another probe"}))
    assert "other probe inputs" in scorestream.environment_problem(
        "echo", results, root, probe)
    _receipt_only(
        out,
        dict(
            environment,
            wrapper_sha256="edited",
            inputs={"probe": probe["identity"]},
        ),
    )
    assert "another environment" in scorestream.environment_problem(
        "echo", results, root, probe)
    edited = [n for n in os.listdir(out) if n.startswith(".receipt-")][0]
    os.rename(
        os.path.join(out, edited),
        os.path.join(out, ".receipt-otherhost-1.json"),
    )
    # a second shard
    receipt.write(out, dict(environment, inputs={"probe": probe["identity"]}))
    assert "another environment" in scorestream.environment_problem(
        "echo", results, root, probe)
    _receipt_only(out, dict(environment, inputs={"probe": probe["identity"]}))
    assert scorestream.environment_problem("echo", results, root, probe) is None
    for name in os.listdir(out):
        if name.startswith(".receipt-"):
            os.remove(os.path.join(out, name))
    assert "no receipt" in scorestream.environment_problem(
        "echo", results, root, probe)
    assert "no stamp" in scorestream.environment_problem(
        "nobody", results, root, probe)
    stamp_path = os.path.join(results, "ours.meta.json")
    stamp = json.load(open(stamp_path, encoding="utf-8"))
    json.dump(
        {
            k: v for k, v in stamp.items() if k != "render_environment_sha256"
        },
        open(stamp_path, "w", encoding="utf-8"),
    )
    assert "no render environment" in scorestream.environment_problem(
        "ours", results, root, probe)


def test_main_writes_a_bound_sidecar_per_rendered_system(
        probe_set, monkeypatch, capsys):
    root = probe_set["root"]
    results = probe_set["results"]
    probe = probe_set["probe"]
    monkeypatch.setattr("sys.argv", ["scorestream.py", "--probe-dir", root])
    scorestream.main()
    side = json.load(
        open(os.path.join(results, "echo.stream.json"), encoding="utf-8"))
    assert (side["verdict"] == "yes" and side["_set_id"] == "tiny" and
            side["_subset"] == "all")
    assert (side["_rule_version"] == scorestream.RULE_VERSION and
            side["_scorer_sha256"] == scorestream.SCORER_SHA256)
    assert side["_probe_identity"] == probe["identity"]
    with open(os.path.join(results, "echo.meta.json"), encoding="utf-8") as fh:
        stamp = json.load(fh)
    assert side["_render_receipt_sha256"] == stamp["render_receipt_sha256"]
    ours = json.load(
        open(os.path.join(results, "ours.stream.json"), encoding="utf-8"))
    assert ours["verdict"] == "yes" and ours["_render_receipt_sha256"]
    path = os.path.join(results, "offline.stream.json")
    with open(path, encoding="utf-8") as fh:
        offline_side = json.load(fh)
    assert offline_side["verdict"] == "no"
    text = capsys.readouterr().out
    assert "offline" in text and "at least 1100 ms of future input" in text
    assert ("drifts over a long input" in text and "≥1100" in text and
            "drops speech" in text)
    assert (side["_probe_version"] == scorestream.PROBE_VERSION and
            side["rule"]["keep_db"] == scorestream.KEEP_DB)
    # the probe itself is confirmed, not rewritten, when current
    monkeypatch.setattr("sys.argv",
                        ["scorestream.py", "--probe", "--probe-dir", root])
    scorestream.main()
    assert "probe is current" in capsys.readouterr().out
    # a system rendered from another probe stops the run
    _receipt_only(
        os.path.join(root, "out", "echo"),
        {
            "system": "echo",
            "wrapper_sha256": "echo" * 4,
            "inputs": {
                "probe": "stale"
            },
        },
    )
    monkeypatch.setattr("sys.argv", ["scorestream.py", "--probe-dir", root])
    with pytest.raises(SystemExit, match="other probe inputs"):
        scorestream.main()


def test_rule_sentence_quotes_the_constants():
    s = scorestream.rule_sentence()
    assert (f"{scorestream.LOOK_AHEAD_MAX_MS:.0f} ms" in s and "SI-SDR" in s and
            s.startswith("the streaming probe"))
