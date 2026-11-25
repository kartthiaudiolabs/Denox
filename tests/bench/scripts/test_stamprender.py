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
"""A render is stamped only when every check holds.

The renderer succeeded, every receipt carries the identity being stamped for
this system, renderer and set, and the inventory is exact and finite.
"""

# pylint: disable=redefined-outer-name

import glob
import json
import sys

import numpy as np
import pytest
import soundfile as sf

import receipt
import stamprender


@pytest.fixture
def render(tmp_path, monkeypatch):
    """A complete three-item render with one receipt.

    Returns its paths and digest.
    """
    root = tmp_path / "set"
    inp, out, res = root / "mix", tmp_path / "out", tmp_path / "res"
    inp.mkdir(parents=True)
    out.mkdir()
    (root / "checksums.json").write_text(json.dumps({"set_id": "tiny"}))
    meta = tmp_path / "meta"
    meta.mkdir()
    json.dump({
        "display": "Sys",
        "mechanism": "masking"
    }, open(meta / "sys.json", "w", encoding="utf-8"))
    monkeypatch.setattr(receipt, "META_DIR", str(meta))
    wrapper = tmp_path / "runsys.py"
    wrapper.write_text("import argparse\n")
    for i in range(3):
        x = np.full(48000 + 100 * i, 0.1, dtype="float32")
        sf.write(inp / f"item_{i}.wav", x, 48000, subtype="FLOAT")
        sf.write(out / f"item_{i}.wav", x * 0.5, 48000, subtype="FLOAT")
    digest = receipt.begin("sys", str(wrapper), str(out), ["runsys.py"],
                           str(inp), "cpu")
    files = sorted(glob.glob(str(inp / "*.wav")))
    return {
        "inp": inp,
        "out": out,
        "res": res,
        "wrapper": wrapper,
        "digest": digest,
        "files": files,
    }


def _stamp(r, **kw):
    args = dict(
        system="sys",
        renderer=str(r["wrapper"]),
        files=r["files"],
        out_dir=str(r["out"]),
        results=str(r["res"]),
        set_id="tiny",
        probe_digest=r["digest"],
        desc={
            "display": "Sys",
            "mechanism": "masking"
        },
    )
    args.update(kw)
    return stamprender.stamp(**args)


def test_a_complete_receipted_render_is_stamped_with_its_identity(render):
    # a stray file is neither counted nor fatal
    (render["out"] / "render.log").write_text("noise")
    sf.write(
        render["out"] / "item_0.wav",
        np.zeros(48000, "float32"),
        48000,
        subtype="FLOAT",
    )
    meta = _stamp(render, subset="core")
    # silence is counted, not refused
    assert meta["n_files"] == 3 and meta["n_silent"] == 1
    assert meta["render_receipt_sha256"] == render["digest"]
    receipts = receipt.read_all(str(render["out"]))
    assert (
        meta["render_environment_sha256"] == receipts[0]["environment_digest"])
    assert (meta["display"] == "Sys" and meta["set_id"] == "tiny" and
            meta["subset"] == "core")
    assert meta["render_receipt"]["n_receipts"] == 1
    assert json.load(open(render["res"] / "sys.meta.json",
                          encoding="utf-8")) == meta


def test_the_stamp_names_the_exact_bytes_of_its_outputs(render):
    first = _stamp(render)["outputs_sha256"]
    assert len(first) == 64 and _stamp(render)["outputs_sha256"] == first
    # one definition, for a stamp and for a render linked in
    assert stamprender.outputs_digest(render["files"],
                                      str(render["out"])) == first
    # a stray file is not an output: the digest ignores it
    (render["out"] / "render.log").write_text("noise")
    assert _stamp(render)["outputs_sha256"] == first
    x, _ = sf.read(render["out"] / "item_1.wav", dtype="float32")
    x[100] += 1e-6  # one sample, one ulp-scale step
    sf.write(render["out"] / "item_1.wav", x, 48000, subtype="FLOAT")
    assert _stamp(render)["outputs_sha256"] != first


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(probe_digest="0" * 64), "another environment"),
        (dict(system="other"), "rendered as"),
        (dict(set_id="another-build"), "not set another-build"),
    ],
)
def test_receipts_of_another_environment_system_or_set_refuse(
        render, change, message):
    with pytest.raises(SystemExit, match=message):
        _stamp(render, **change)
    path = render["res"] / f"{change.get('system', 'sys')}.meta.json"
    assert not path.exists()


def test_mixed_missing_or_foreign_receipts_refuse(render, tmp_path):
    out = render["out"]
    for p in out.glob(".receipt-*.json"):
        p.rename(out / ".receipt-otherhost-1.json")
    other = tmp_path / "runother.py"
    other.write_text("import argparse\nX = 1\n")
    receipt.begin("sys", str(other), str(out), ["runother.py"],
                  str(render["inp"]), "cpu")
    with pytest.raises(SystemExit, match="identities"):
        _stamp(render)
    for p in out.glob(".receipt-*.json"):
        p.unlink()
    with pytest.raises(SystemExit, match="no render receipt"):
        _stamp(render)
    receipt.begin("sys", str(other), str(out), ["runother.py"],
                  str(render["inp"]), "cpu")
    digest = receipt.read_all(str(out))[0]["digest"]
    with pytest.raises(SystemExit, match="rendered by"):
        _stamp(render, probe_digest=digest)


def test_a_wrong_missing_or_non_finite_output_refuses(render):
    out = render["out"]
    x, _ = sf.read(out / "item_1.wav", dtype="float32")
    sf.write(out / "item_1.wav", x[:-10], 48000, subtype="FLOAT")
    with pytest.raises(SystemExit, match="frames"):
        _stamp(render)
    sf.write(out / "item_1.wav", x, 48000, subtype="PCM_16")
    with pytest.raises(SystemExit, match="PCM_16"):
        _stamp(render)
    y = x.copy()
    y[5] = np.inf
    sf.write(out / "item_1.wav", y, 48000, subtype="FLOAT")
    with pytest.raises(SystemExit, match="NaN or inf"):
        _stamp(render)
    sf.write(out / "item_1.wav", x, 48000, subtype="FLOAT")
    (out / "item_2.wav").unlink()
    with pytest.raises(SystemExit, match="1 outputs missing"):
        _stamp(render)
    # nothing was stamped along the way
    assert not (render["res"] / "sys.meta.json").exists()


def test_the_command_line_refuses_a_failed_renderer_and_stamps_a_good_one(
        render, monkeypatch, capsys):
    argv = [
        "stamprender.py",
        "sys",
        "--wrapper",
        str(render["wrapper"]),
        "--inputs",
        str(render["inp"]),
        "--outputs",
        str(render["out"]),
        "--results",
        str(render["res"]),
        "--set-id",
        "tiny",
        "--probe-digest",
        render["digest"],
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--wrapper-exit", "1"])
    with pytest.raises(SystemExit, match="exited 1"):
        stamprender.main()
    monkeypatch.setattr(sys, "argv", argv + ["--wrapper-exit", "0"])
    stamprender.main()
    assert "stamped sys: 3 files" in capsys.readouterr().out
    with open(render["res"] / "sys.meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["mechanism"] == "masking"


def test_a_linked_output_refuses_a_computed_renders_stamp(render):
    """A render writes files; a folder linked in is stamped by runsystem.py
    --outputs with its source named, never as a computed render."""
    r = render
    target = r["out"] / "item_1.wav"
    target.unlink()
    target.symlink_to(r["inp"] / "item_1.wav")

    with pytest.raises(SystemExit, match="link"):
        _stamp(r)
