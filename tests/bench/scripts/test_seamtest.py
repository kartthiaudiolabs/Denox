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
"""Chunk-seam check: zone geometry, the shift trick, and fakes.

The seam behaviour of each fake is known.
"""

# pylint: disable=redefined-outer-name,unused-argument

import json
import math

import pytest
import soundfile as sf
import torch

import common
import seamtest

SR = common.SR


def speech(seconds, seed=0, level=0.2):
    n = int(seconds * SR)
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    env = 0.3 + 0.7 * (0.5 + 0.5 * torch.sin(2 * math.pi * 2.5 * t)).pow(2)
    return torch.randn(n, generator=g) * env * level


def crossfaded(x, chunk_s, overlap_s, seg_fn):
    """Returns a fake chunked system whose seams sit where the isolator's do.

    The separator's own windowing (model.py, enhance) around ``seg_fn(seg,
    first, last)``.
    """
    chunk, overlap, hop = seamtest.window_samples(chunk_s, overlap_s)
    n = len(x)
    if n <= chunk:
        return seg_fn(x, True, True)
    out, weight = torch.zeros(n), torch.zeros(n)
    for start in range(0, n, hop):
        end = min(start + chunk, n)
        seg = seg_fn(x[start:end], start == 0, end == n)
        w = torch.ones(end - start)
        if overlap > 0 and start > 0:
            w[:overlap] = torch.linspace(0.0, 1.0, overlap)
        if overlap > 0 and end < n:
            w[-overlap:] = torch.linspace(1.0, 0.0, overlap)
        out[start:end] += seg * w
        weight[start:end] += w
        if end == n:
            break
    return out / weight.clamp(min=1e-6)


def edge_damage(overlap_s, gain=0.25):
    """Returns a window whose overlapping edges are ``gain`` quieter.

    The first and last ``overlap`` of the window, except at the clip's own ends:
    after the crossfade every seam zone sits at ``gain``, everything else is
    intact.
    """
    edge = int(overlap_s * SR)

    def fn(seg, first, last):
        y = seg.clone()
        if not first:
            y[:edge] *= gain
        if not last:
            y[-edge:] *= gain
        return y

    return fn


def test_seam_zones_follow_the_separators_windows():
    n = 45 * SR
    # hop 6 s: windows at 0, 6, ..., 36 s reach the end
    zones = seamtest.seam_zones(n, 8.0, 2.0)
    assert len(zones) == 7
    assert zones[0] == (58, 82)  # [6.0 - 0.2, 8.0 + 0.2) s in 100 ms blocks
    assert zones[-1] == (418, 442)
    # the same seams one second earlier
    shifted = seamtest.seam_zones(n, 8.0, 2.0, shift_seconds=1.0)
    assert len(shifted) == 7 and shifted[0] == (48, 72)
    # shorter than one window: no seams
    assert not seamtest.seam_zones(5 * SR, 8.0, 2.0)
    m = seamtest.seam_mask(450, zones)
    assert int(m.sum()) == 7 * 24 and bool(m[58]) and not bool(m[57])
    # overlap capped at half
    assert seamtest.window_samples(8.0, 6.0) == (8 * SR, 4 * SR, 4 * SR)
    # zone-sized stretches away from every seam
    ctrl = seamtest.control_windows(450, zones, 24)
    assert ctrl and all(b1 - b0 == 24 for b0, b1 in ctrl)
    assert all(not m[b0:b1].any() for b0, b1 in ctrl)
    # tiled from the start, short of a seam
    assert ctrl[0] == (0, 24) and ctrl[1] == (24, 48)
    # no stretch that long between seams
    assert not seamtest.control_windows(450, zones, 60)


def test_block_diff_reads_the_floor_for_identical_signals_and_level_otherwise():
    x = speech(2.0)
    d, loud = seamtest.block_diff_db(x, x, x)
    assert float(d.max()) == seamtest.DIFF_FLOOR_DB and bool(loud.all())
    d, _ = seamtest.block_diff_db(x, 0.5 * x, x)
    assert float(d.mean()) == pytest.approx(20 * math.log10(0.5), abs=0.01)


def test_identity_system_reads_the_floor_and_no_seam_penalty():
    x = speech(20.0)
    d = seamtest.measure_item(lambda a, c, o: a, x, x, 4.0, 1.0)
    assert d["n_seam_blocks"] > 0 and d["n_else_blocks"] > 0
    assert d["shift_diff_db"] == seamtest.DIFF_FLOOR_DB
    assert d["shift_diff_seam_mean_db"] == seamtest.DIFF_FLOOR_DB
    assert abs(d["seam_penalty_db"]) < 0.01 and abs(d["seam_dip_db"]) < 0.01
    assert (abs(d["kept_seam_db"]) < 0.01 and
            abs(d["seam_resid_excess_db"]) < 0.5)


def test_edge_damage_is_caught_at_the_seams_and_nowhere_else():
    x = speech(20.0, seed=1)

    def run(a, c, o):
        return crossfaded(a, c, o, edge_damage(o))

    d = seamtest.measure_item(run, x, x, 4.0, 1.0, shift_seconds=1.0)
    # -12 dB at every seam
    assert d["seam_dip_db"] == pytest.approx(20 * math.log10(0.25), abs=0.5)
    # the mean over the widened zone still shows it
    assert d["seam_penalty_db"] < -6
    # nothing lost between the seams
    assert d["kept_else_db"] == pytest.approx(0.0, abs=0.05)
    # position-independent there
    assert d["shift_diff_else_mean_db"] == seamtest.DIFF_FLOOR_DB
    assert d["shift_diff_seam_mean_db"] > d["shift_diff_else_mean_db"] + 40
    # the worst block of every zone: the full -12 dB dip
    assert d["shift_diff_seam_max_db"] > -8
    # and of every control window: nothing
    assert d["shift_diff_else_max_db"] == seamtest.DIFF_FLOOR_DB
    assert d["n_control_windows"] > 0
    assert d["worst_in_seam"]


def test_noise_everywhere_is_position_dependence_but_not_a_seam_penalty():

    # fresh -30 dB noise on every call: the two passes differ everywhere
    def run(a, c, o):
        g = torch.Generator().manual_seed(len(a))
        return a + torch.randn(len(a), generator=g) * float(
            a.pow(2).mean().sqrt()) * 10**(-30 / 20)

    x = speech(20.0, seed=2)
    d = seamtest.measure_item(run, x, x, 4.0, 1.0)
    # two independent draws: +3 dB
    assert d["shift_diff_db"] == pytest.approx(-30 + 3, abs=2)
    assert abs(d["shift_diff_seam_mean_db"] - d["shift_diff_else_mean_db"]) < 2
    # same statistic
    assert abs(d["shift_diff_seam_max_db"] - d["shift_diff_else_max_db"]) < 2
    # both sides fluctuate
    assert abs(d["seam_penalty_db"]) < 0.5 and abs(d["seam_dip_db"]) < 1


def test_reference_free_clip_reports_position_dependence_only():
    x = speech(12.0)
    d = seamtest.measure_item(lambda a, c, o: a, x, None, 4.0, 1.0)
    assert "shift_diff_seam_mean_db" in d and "seam_penalty_db" not in d


def test_short_clip_has_no_seams_but_a_shift_difference():
    x = speech(3.0)
    d = seamtest.measure_item(lambda a, c, o: a, x, x, 8.0, 2.0)
    assert (d["n_seam_blocks"] == 0 and
            d["shift_diff_seam_mean_db"] != d["shift_diff_seam_mean_db"])
    assert d["shift_diff_db"] == seamtest.DIFF_FLOOR_DB
    s = seamtest.aggregate([d])
    assert (s["n_with_seams"] == 0 and
            s["seam_penalty_db"] != s["seam_penalty_db"])


def test_aggregate_averages_items_and_counts_seams():
    a = {
        "n_seam_blocks": 4,
        "shift_diff_db": -50.0,
        "seam_penalty_db": -2.0,
        "worst_in_seam": True,
    }
    b = {"n_seam_blocks": 0, "shift_diff_db": -70.0, "worst_in_seam": False}
    s = seamtest.aggregate([a, b])
    assert s["n_items"] == 2 and s["n_with_seams"] == 1
    assert s["shift_diff_db"] == -60.0 and s["seam_penalty_db"] == -2.0
    assert s["worst_in_seam_frac"] == 1.0  # only items with seams vote


def test_parse_windows_and_label():
    windows = seamtest.parse_windows("8/2,5/1", (8.0, 2.0))
    assert windows == [(8.0, 2.0), (5.0, 1.0)]
    assert seamtest.parse_windows("", (8.0, 2.0)) == [(8.0, 2.0)]
    assert (seamtest.label(8.0, 2.0) == "8/2" and
            seamtest.label(0.5, 0.1) == "0.5/0.1")


@pytest.fixture
def long_set(tmp_path, monkeypatch):
    root = tmp_path / "set"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    rows = []
    for i in range(2):
        x = speech(3.0, seed=10 + i)
        rid = f"longATp4_{i:03d}.wav"
        sf.write(str(root / "mix" / rid), x.numpy(), SR, subtype="FLOAT")
        sf.write(str(root / "ref" / rid), x.numpy(), SR, subtype="FLOAT")
        rows.append({
            "id": rid,
            "condition": "long@+4",
            "mix_path": f"mix/{rid}",
            "clean_path": f"ref/{rid}",
        })
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump({
        "set_id": "tiny",
        "n_items": 2
    }, open(root / "checksums.json", "w", encoding="utf-8"))
    monkeypatch.setattr(common, "EVAL", str(root))
    return root


def test_main_runs_the_tiny_model_through_the_public_path(
        identity_model, long_set, tmp_path, monkeypatch):
    # pylint: disable-next=import-outside-toplevel
    from inworld.voice_isolation import isolator

    vi = isolator.VoiceIsolator(identity_model, device="cpu")
    calls = []

    def runner(a):
        calls.append(a.model)
        return vi, (lambda x, c=None, o=None: vi.isolate(
            x, SR, chunk_seconds=c, overlap_seconds=o))

    monkeypatch.setattr(common, "isolator_runner", runner)
    out = tmp_path / "seams.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "seamtest.py",
            "--model",
            "tiny",
            "--windows",
            "0.5/0.1",
            "--shift-seconds",
            "0.3",
            "--out",
            str(out),
        ],
    )
    seamtest.main()
    r = json.load(open(out, encoding="utf-8"))
    assert calls == ["tiny"] and r["set_id"] == "tiny" and r["model"] == "tiny"
    w = r["windows"]["0.5/0.1"]
    assert w["summary"]["n_items"] == 2 and w["summary"]["n_with_seams"] == 2
    # the tiny model is the identity
    assert abs(w["summary"]["seam_penalty_db"]) < 0.5
    assert w["items"][0]["id"] == "longATp4_000.wav"
