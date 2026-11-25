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
"""Measures how much of the ranking is the anchor policy.

It runs a sensitivity sweep over the declared constants.

    anchorsweep.py --results bench/results [--view all] [--draws 2000]
                   [--seed 0] [--json-out FILE] [--markdown-out FILE]

Every declared endpoint in ``anchors.json`` that carries a ``range`` is a policy
choice, not a measurement. This recomputes the composites with those constants
moved inside their ranges:

  one at a time  each ranged constant at the low and at the high end of its
                 range, everything else as declared: which single choice moves
                 the order, and by how much;
  jointly        ``--draws`` seeded draws, every ranged constant uniform in its
                 range at once: per system the 5th to 95th percentile of its
                 composite, how often it ranks first, and for every pair how
                 often one sits above the other and how often the two are inside
                 the tie band.

Raw measurements never change; only the mapping from a raw value to a score
does. The output says which statements about the board hold over the whole
declared policy space and which hold only at the declared point.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors
import ranking
import roster
import scorefiles


def ranged(policy):
    """Returns [(rule index, side, low, high, label)] of the ranged constants.

    There is one entry for every declared constant with a range.
    """
    out = []
    for i, rule in enumerate(policy["rules"]):
        for side in ("floor", "ceiling"):
            span = rule[side].get("range")
            if span is not None:
                label = (f"{'/'.join(rule['conditions'])} {rule['metrics'][0]}"
                         f" {side}")
                out.append((i, side, float(span[0]), float(span[1]), label))
    return out


def with_constants(calibration, policy, values):
    """Returns a copy of ``calibration`` whose points use ``values``.

    ``values`` is {(rule index, side): constant}.
    """
    cal = copy.deepcopy(calibration)
    for (i, side), value in values.items():
        rule = policy["rules"][i]
        for condition in rule["conditions"]:
            for metric in rule["metrics"]:
                key = anchors.row_key(condition, metric)
                if key in cal["points"]:
                    cal["points"][key][side] = float(value)
    for key, point in cal["points"].items():
        if point["direction"] * (point["ceiling"] - point["floor"]) <= 0:
            raise ValueError(
                f"ranges let the endpoints of {key} cross: narrow them")
    return cal


def composites(systems, tiers, calibration, tier="full"):
    """Returns {system: composite} of the ranked systems of ``tier``.

    Every composite is computed under ``calibration``.
    """
    _, comp, order = ranking.rank(systems, tiers, calibration)
    return {n: comp[n] for n in order[tier]}


def sweep(systems, tiers, calibration, policy, draws=2000, seed=0, tier="full"):
    """Returns the sensitivity of the composites to the ranged constants.

    The report holds both the one-at-a-time and the joint sensitivity.
    """
    knobs = ranged(policy)
    base = composites(systems, tiers, calibration, tier)
    names = sorted(base, key=lambda n: -base[n])
    order0 = list(names)
    single = []
    for i, side, low, high, label in knobs:
        declared = policy["rules"][i][side]["value"]
        row = {
            "constant": label,
            "declared": declared,
            "range": [low, high],
            "ends": {},
        }
        for end, value in (("low", low), ("high", high)):
            comp = composites(
                systems,
                tiers,
                with_constants(calibration, policy, {(i, side): value}),
                tier,
            )
            order = sorted(comp, key=lambda n, comp=comp: -comp[n])
            # a swap between two systems tied at the declared point says
            # nothing; a swap of a pair the declared board separates by more
            # than the tie band does
            reversed_pairs = [
                [x, y]
                for x in names
                for y in names
                if base[x] - base[y] > ranking.TIE_BAND and comp[y] > comp[x]
            ]
            row["ends"][end] = {
                "value": value,
                "composites": comp,
                "order": order,
                "order_changed": bool(reversed_pairs),
                "reversed_pairs": reversed_pairs,
                "leader": order[0],
            }
        row["max_shift"] = max(
            abs(row["ends"][e]["composites"][n] - base[n])
            for e in ("low", "high")
            for n in names)
        single.append(row)
    rng = np.random.default_rng(seed)
    samples = {n: [] for n in names}
    first = {n: 0 for n in names}
    above = {a: {b: 0 for b in names if b != a} for a in names}
    tied = {a: {b: 0 for b in names if b != a} for a in names}
    for _ in range(draws):
        values = {
            (i, side): rng.uniform(low, high) for i, side, low, high, _ in knobs
        }
        comp = composites(systems, tiers,
                          with_constants(calibration, policy, values), tier)
        for n in names:
            samples[n].append(comp[n])
        first[max(comp, key=comp.get)] += 1
        for a in names:
            for b in names:
                if a != b:
                    above[a][b] += comp[a] > comp[b]
                    tied[a][b] += abs(comp[a] - comp[b]) <= ranking.TIE_BAND
    joint = {
        n: {
            "declared": base[n],
            "p05": float(np.percentile(samples[n], 5)),
            "p50": float(np.percentile(samples[n], 50)),
            "p95": float(np.percentile(samples[n], 95)),
            "first_fraction": first[n] / draws if draws else float("nan"),
        } for n in names
    }
    pairs = {
        a: {
            b: {
                "above_fraction":
                    (above[a][b] / draws if draws else float("nan")),
                "tie_fraction": tied[a][b] / draws if draws else float("nan"),
            } for b in above[a]
        } for a in names
    }
    return {
        "anchors_version": policy["version"],
        "set_id": calibration["set_id"],
        "subset": calibration["subset"],
        "tier": tier,
        "draws": draws,
        "seed": seed,
        "tie_band": ranking.TIE_BAND,
        "declared_order": order0,
        "single": single,
        "joint": joint,
        "pairs": pairs,
    }


def axis_sensitivity(systems,
                     tiers,
                     calibration,
                     draws=2000,
                     seed=0,
                     tier="full"):
    """Returns how the order depends on the way the axes combine.

    The order with each axis left out in turn (its part, when the composite has
    parts, reads its remaining axes and goes when none remain), and how often
    each system ranks first when the weights of the axes are drawn uniformly
    from the simplex.
    """
    axes, _, order = ranking.rank(systems, tiers, calibration)
    names = order[tier]
    # scored beside the composite, or on another board: no weight to relax
    board_parts = {
        name: list(members)
        for name, members in (ranking.PARTS or {}).items()
        if not ranking.BOARDS or name in ranking.BOARDS[ranking.ranking_board()]
    }
    aside = set(ranking.OUTSIDE) | {
        a for name, members in (ranking.PARTS or {}).items()
        if name not in board_parts for a in members
    }
    axis_names = [
        a for a in ranking.axis_order(ranking.AXES_BY_TIER[tier])
        if a not in aside
    ]
    leave_out = {}
    for axis in axis_names:
        rest = [a for a in axis_names if a != axis]
        parts = {
            name: [a for a in members if a != axis]
            for name, members in board_parts.items()
        }
        parts = {
            name: members for name, members in parts.items() if members
        } or None
        c = {
            n: anchors.composite({a: axes[n][a] for a in rest}, parts)
            for n in names
        }
        leave_out[axis] = {
            "composites": c,
            "order": sorted(c, key=lambda n, c=c: -c[n]),
        }
    rng = np.random.default_rng(seed)
    table = np.array([[axes[n][a] for a in axis_names] for n in names])
    first = np.zeros(len(names))
    for w in rng.dirichlet(np.ones(len(axis_names)), size=draws):
        first[int(np.argmax(table @ w))] += 1
    return {
        "axis_scores": {
            n: {
                a: axes[n][a] for a in axis_names
            } for n in names
        },
        "leave_one_out": leave_out,
        "random_weights_first_fraction": {
            n: float(first[i] / draws) if draws else float("nan")
            for i, n in enumerate(names)
        },
    }


def markdown(report, label=None):
    """Returns the sweep as two markdown tables.

    One is the joint sweep per system, the other the single constants that
    change the order.
    """
    label = label or roster.display_name
    names = report["declared_order"]
    lines = [
        ("Denoising scores of the full-band tier with every declared anchor"
         f" constant drawn inside its stated range ({report['draws']} seeded"
         " draws; anchors"
         f" {report['anchors_version']}). The raw measurements are the same"
         " in every draw; only the scale moves."),
        "",
        ("| system | declared | 5th to 95th percentile | ranked first |"
         " above the next system | tied with it |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, n in enumerate(names):
        j = report["joint"][n]
        nxt = names[k + 1] if k + 1 < len(names) else None
        pair = report["pairs"][n][nxt] if nxt else None
        lines.append(
            f"| {label(n)} | {j['declared']:.3f} | {j['p05']:.3f} to"
            f" {j['p95']:.3f} | {100 * j['first_fraction']:.0f} % | " +
            (f"{100 * pair['above_fraction']:.0f} % |"
             f" {100 * pair['tie_fraction']:.0f} % |" if pair else "- | - |"))
    movers = [
        r for r in report["single"]
        if any(e["order_changed"] for e in r["ends"].values())
    ]
    lines += [
        "",
        ("Single constants whose range alone reverses a pair the declared"
         " board separates by more than the tie band of"
         f" {report['tie_band']:.2f} (everything else as declared):"),
        "",
    ]
    if not movers:
        lines.append(
            "None: no single constant, moved to either end of its range,"
            " reverses such a pair.")
    else:
        lines += [
            "| constant | declared | end of range | order at that end |",
            "|---|---:|---:|---|",
        ]
        for r in movers:
            for end in ("low", "high"):
                e = r["ends"][end]
                if e["order_changed"]:
                    lines.append(f"| {r['constant']} | {r['declared']:g} |"
                                 f" {e['value']:g} | " +
                                 " > ".join(label(n) for n in e["order"]) +
                                 " |")
    ax = report.get("axes")
    if ax:
        lines += [
            "",
            "The other policy choice is how the axes combine (" +
            ranking.composite_rule() +
            "). The order with one axis left out, and how often a system"
            " ranks first when the weights of the axes are drawn uniformly at"
            " random:",
            "",
            "| axis left out | order |",
            "|---|---|",
        ]
        for axis, d in ax["leave_one_out"].items():
            lines.append(f"| {axis} | " + " > ".join(
                f"{label(n)} {d['composites'][n]:.3f}" for n in d["order"]) +
                         " |")
        lines += [
            "",
            "| system | ranked first under random axis weights |",
            "|---|---:|",
        ]
        for n in names:
            lines.append(
                f"| {label(n)} |"
                f" {100 * ax['random_weights_first_fraction'][n]:.0f} % |")
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--results",
        default=os.environ.get("BENCH_RESULTS",
                               os.path.join(os.path.dirname(here), "results")),
    )
    ap.add_argument("--view", default="all", choices=sorted(ranking.AXES_VIEWS))
    ap.add_argument("--tiers",
                    default="",
                    help="JSON file with extra {name: [tier, label]}")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--markdown-out", default="")
    a = ap.parse_args()
    ranking.select_view(a.view)
    tiers = roster.tiers(a.tiers)
    policy = anchors.definitions()
    systems = scorefiles.load(a.results)
    calibration = anchors.load(os.path.join(a.results, anchors.RESOLVED),
                               policy)
    report = sweep(systems, tiers, calibration, policy, a.draws, a.seed)
    report["axes"] = axis_sensitivity(systems, tiers, calibration, a.draws,
                                      a.seed)
    md = markdown(report)
    print(md)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    if a.markdown_out:
        with open(a.markdown_out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")


if __name__ == "__main__":
    main()
