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
"""Runtime check: a fake clock, clip picking and the README line.

Failures are counted, not averaged.
"""

import json
import math

import pytest
import soundfile as sf
import torch

import common
import speedtest

SR = common.SR


class _Clock:
    """A clock the fake systems advance by hand.

    Timings are exact, and the tests take no time.
    """

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_summarize_percentiles_and_empty():
    s = speedtest.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (s["n"] == 5 and s["p50"] == 3.0 and
            s["p95"] == pytest.approx(4.8) and s["max"] == 5.0)
    assert s["mean"] == 3.0
    e = speedtest.summarize([])
    assert e["n"] == 0 and e["p50"] != e["p50"]


def test_check_output_names_every_failure():
    assert speedtest.check_output(torch.zeros(100), 100) is None
    assert speedtest.check_output(torch.zeros(99), 100).startswith("length")
    assert speedtest.check_output(torch.zeros(2, 50), 100).startswith("length")
    y = torch.zeros(100)
    y[3] = float("nan")
    assert speedtest.check_output(y, 100) == "nan"
    assert speedtest.check_output(object(), 100).startswith("not a signal")


def test_cold_start_is_the_first_call_and_warm_passes_are_summarised_per_group(
):
    clock = _Clock()

    # 10 ms of work per second of audio, plus 0.5 s the first time (kernel
    # selection)
    def run(x):
        clock.t += 0.01 * len(x) / SR + (0.5 if not run.warm else 0.0)
        run.warm = True
        return x

    run.warm = False
    clips = [
        ("1s", "a", torch.zeros(SR)),
        ("1s", "b", torch.zeros(SR)),
        ("2s", "c", torch.zeros(2 * SR)),
    ]
    r = speedtest.measure(run, clips, repeat=3, clock=clock)
    assert r["cold"]["clip"] == "a" and r["cold"]["seconds"] == pytest.approx(
        0.51)
    assert r["cold"]["failure"] is None
    g1, g2 = r["groups"]["1s"], r["groups"]["2s"]
    assert (g1["clips"] == ["a", "b"] and len(g1["times"]) == 6 and
            g1["n_failed"] == 0)
    assert g1["summary"]["p50"] == pytest.approx(
        0.01) and g1["realtime_p50"] == pytest.approx(100)
    assert (g2["summary"]["p50"] == pytest.approx(0.02) and
            g2["mean_clip_seconds"] == 2.0)


def test_failures_are_counted_with_their_reason_and_never_averaged():
    clock = _Clock()
    calls = []

    def run(x):
        calls.append(len(x))
        clock.t += 0.1
        if len(x) == 2 * SR:
            return torch.full((len(x),), float("nan"))
        if len(calls) == 3:
            raise RuntimeError("out of memory")
        return x

    clips = [("1s", "a", torch.zeros(SR)), ("2s", "b", torch.zeros(2 * SR))]
    r = speedtest.measure(run, clips, repeat=2, clock=clock)
    g1, g2 = r["groups"]["1s"], r["groups"]["2s"]
    assert g1["n_failed"] == 1 and g1["failures"][0]["reason"].startswith(
        "raised: out of memory")
    assert len(g1["times"]) == 1 and g1["summary"]["p50"] == pytest.approx(0.1)
    assert g2["n_failed"] == 2 and {f["reason"] for f in g2["failures"]
                                   } == {"nan"}
    assert g2["times"] == [] and math.isnan(g2["realtime_p50"])
    with pytest.raises(ValueError):
        speedtest.measure(run, [], repeat=1, clock=clock)


def test_pick_clips_takes_the_nearest_lengths_per_duration():
    rows = [
        ({
            "id": "a"
        }, 1.0),
        ({
            "id": "b"
        }, 2.2),
        ({
            "id": "c"
        }, 5.1),
        ({
            "id": "d"
        }, 9.0),
    ]
    one = speedtest.pick_clips(rows, [1.0, 5.0], 1)
    assert [(g, r["id"]) for g, r, _ in one] == [("1s", "a"), ("5s", "c")]
    two = speedtest.pick_clips(rows, [1.0, 5.0], 2)
    assert [r["id"] for _, r, _ in two] == ["a", "b", "c", "d"]
    assert two[0][0] == "1s" and two[-1][0] == "5s"


def test_clip_seconds_reads_the_headers(tmp_path):
    (tmp_path / "mix").mkdir()
    sf.write(str(tmp_path / "mix" / "x.wav"), torch.zeros(SR // 2).numpy(), SR)
    rows = [{"id": "x.wav", "mix_path": str(tmp_path / "mix" / "x.wav")}]
    assert speedtest.clip_seconds(rows)[0][1] == pytest.approx(0.5)


def test_readme_line_quotes_p50_realtime_and_memory():
    report = {
        "groups": {
            "30s": {
                "mean_clip_seconds": 30.0,
                "summary": {
                    "p50": 7.2
                },
                "realtime_p50": 30 / 7.2,
            },
            "8s": {
                "mean_clip_seconds": 8.0,
                "summary": {
                    "p50": 1.5
                },
                "realtime_p50": 8 / 1.5,
            },
            "45s": {
                "mean_clip_seconds": 45.0,
                "summary": {
                    "p50": float("nan")
                },
                "realtime_p50": float("nan"),
            },
        }
    }
    line = speedtest.readme_line(report, "2.8 GB peak GPU memory")
    assert (
        line == "30 s clip in 7.2 s (4.2× realtime); 8 s clip in 1.5 s (5.3×"
        " realtime); 45s clip: every pass failed; 2.8 GB peak GPU memory")


def test_main_synthetic_with_the_tiny_model_writes_the_report(
        identity_model, tmp_path, monkeypatch):
    # pylint: disable-next=import-outside-toplevel
    from inworld.voice_isolation import isolator

    vi = isolator.VoiceIsolator(identity_model, device="cpu")
    monkeypatch.setattr(
        common,
        "isolator_runner",
        lambda a: (
            vi,
            lambda x, c=None, o=None: vi.isolate(
                x, SR, chunk_seconds=c, overlap_seconds=o),
        ),
    )
    out = tmp_path / "speed.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "speedtest.py",
            "--model",
            "tiny",
            "--synthetic",
            "0.5",
            "--repeat",
            "2",
            "--out",
            str(out),
        ],
    )
    speedtest.main()
    r = json.load(open(out, encoding="utf-8"))
    assert r["synthetic"] and r["set_id"] is None and r["device_type"] == "cpu"
    g = r["groups"]["0.5s"]
    assert (g["summary"]["n"] == 2 and g["n_failed"] == 0 and
            g["realtime_p50"] > 0)
    assert r["cold"]["failure"] is None and "0 s clip in" in r["readme_line"]
    # the model's own window
    assert (r["chunk_seconds"],
            r["overlap_seconds"]) == (identity_model.config.chunk_seconds,
                                      identity_model.config.overlap_seconds)
