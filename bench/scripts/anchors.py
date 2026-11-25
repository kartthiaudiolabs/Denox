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
"""Versioned, roster-independent calibration for benchmark composites.

``anchors.json`` declares policy. ``anchors.resolved.json`` in a results
directory freezes its set-derived endpoints and deletion cohort. Removing an
INPUT, REF, or competitor score file after calibration cannot move a score.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
from typing import Any

DEFAULT = pathlib.Path(
    os.environ.get("BENCH_ANCHORS",
                   pathlib.Path(__file__).resolve().parents[1] /
                   "anchors.json"))
RESOLVED = "anchors.resolved.json"


def digest(value: Any) -> str:
    """Return the digest of canonical, finite JSON data."""
    data = json.dumps(value,
                      sort_keys=True,
                      separators=(",", ":"),
                      allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def atomic_json(path: str | pathlib.Path, value: Any) -> None:
    """Replace a JSON artifact only after its full serialization succeeds."""
    path = pathlib.Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def definitions(path: str | pathlib.Path = DEFAULT) -> dict:
    """Read and validate declared anchors without loading audio libraries."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(
            data.get("version"), str):
        raise ValueError("unsupported anchor schema or missing anchor version")
    changes = data.get("changelog", [])
    if not changes or changes[-1].get("version") != data["version"]:
        raise ValueError("anchor version needs a changelog entry")
    if len({r["version"] for r in changes}) != len(changes):
        raise ValueError("anchor versions must be unique")
    gap = data.get("minimum_gap")
    if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0:
        raise ValueError("minimum_gap must be finite and positive")
    seen = set()
    for rule in data["rules"]:
        if rule["direction"] not in (-1, 1):
            raise ValueError("anchor direction must be -1 or 1")
        for endpoint in (rule["floor"], rule["ceiling"]):
            if not endpoint.get("rationale") or ("source"
                                                 in endpoint) == ("value"
                                                                  in endpoint):
                raise ValueError(
                    "each anchor needs one value or source and a rationale")
            if "source" in endpoint and endpoint["source"] not in (
                    "INPUT",
                    "REF",
            ):
                raise ValueError(
                    "anchors may only depend on INPUT, REF, or constants")
            for key in ("value", "fallback"):
                if key in endpoint and (isinstance(endpoint[key], bool) or
                                        not isinstance(endpoint[key],
                                                       (int, float)) or
                                        not math.isfinite(endpoint[key])):
                    raise ValueError("anchor constants must be finite numbers")
            verify = endpoint.get("verify")
            if verify is not None:
                if "value" not in endpoint or verify.get("source") not in (
                        "INPUT",
                        "REF",
                ):
                    raise ValueError(
                        "verify checks a declared value against INPUT or REF")
                tolerance = verify.get("tolerance")
                if (isinstance(tolerance, bool) or
                        not isinstance(tolerance, (int, float)) or
                        not math.isfinite(tolerance) or tolerance <= 0):
                    raise ValueError("verify needs a finite positive tolerance")
            span = endpoint.get("range")
            if span is not None:
                if ("value" not in endpoint or len(span) != 2 or not all(
                        isinstance(v, (int, float)) and math.isfinite(v)
                        for v in span) or
                        not span[0] <= endpoint["value"] <= span[1]):
                    raise ValueError(
                        "range must be a finite [low, high] containing the"
                        " declared value")
        for condition in rule["conditions"]:
            for metric in rule["metrics"]:
                key = row_key(condition, metric)
                if key in seen:
                    raise ValueError(f"duplicate anchor definition: {key}")
                seen.add(key)
    return data


def reference_conditions(path: str | pathlib.Path = DEFAULT) -> set[str]:
    """Returns the conditions the REF row must be scored on.

    Every rule with an endpoint measured from REF, or with a declared value
    verified against it.
    """
    return {
        condition for rule in definitions(path)["rules"] if any("REF" in (
            rule[side].get("source"),
            (rule[side].get("verify") or {}).get("source"),
        ) for side in ("floor", "ceiling")) for condition in rule["conditions"]
    }


def stamp(policy: dict | None = None) -> dict:
    """Returns the normalization identity to stamp on newly computed scores."""
    policy = definitions() if policy is None else policy
    return {
        "_anchors_version": policy["version"],
        "_anchors_sha256": digest(policy),
    }


def row_key(condition: str, metric: str) -> str:
    """Return an unambiguous condition/metric identity."""
    return json.dumps([condition, metric], separators=(",", ":"))


def rule_for(policy: dict, condition: str, metric: str) -> dict:
    """Find the declared rule for one sub-metric, failing on undeclared rows."""
    for rule in policy["rules"]:
        if condition in rule["conditions"] and metric in rule["metrics"]:
            return rule
    raise ValueError(f"no anchors declared for {condition}/{metric}")


def _endpoint(
    endpoint: dict,
    systems: dict,
    condition: str,
    metric: str,
    verify: bool = True,
) -> float:
    if "value" in endpoint:
        check = endpoint.get("verify")
        # a declared endpoint the set must agree with
        if check is not None and verify:
            measured = _endpoint({"source": check["source"]}, systems,
                                 condition, metric)
            if abs(measured - endpoint["value"]) > check["tolerance"]:
                raise ValueError(
                    f"declared anchor {endpoint['value']} for"
                    f" {condition}/{metric} is"
                    f" {abs(measured - endpoint['value']):.3f} from"
                    f" {check['source']}'s measured {measured:.3f} (tolerance"
                    f" {check['tolerance']}): fix the set or declare a new"
                    " anchor version")
        return float(endpoint["value"])
    name = endpoint["source"]
    record = systems.get(name, {}).get(condition, {})
    value = record.get(metric)
    count = record.get("metric_counts", {}).get(metric, record.get("n", 0))
    if count < 20 or not isinstance(value, (int, float)):
        raise ValueError(
            f"missing measured anchor {name}: {condition}/{metric}")
    if not math.isfinite(value):
        raise ValueError(
            f"nonfinite measured anchor {name}: {condition}/{metric}; repair"
            " scoring")
    return float(value)


def resolve(
    policy: dict,
    systems: dict,
    axes: dict,
    sealed: bool = False,
    parent: dict | None = None,
) -> dict:
    """Freeze endpoints and the INPUT-defined cohort for one set and subset.

    Args:
        policy: Validated anchor definitions.
        systems: Loaded, level-calibrated INPUT and REF scores.
        axes: Tier-to-axis-to-sub-metric definitions for the selected view.
        sealed: Whether this calibration belongs to sealed data, which is never
            published.
        parent: For a development panel, the frozen whole-set calibration of the
            build it is drawn from. The declared endpoints were verified there
            against every item; a panel inherits that verification instead of
            repeating it on its few items, and records which calibration it
            inherits.

    Returns:
        A self-identifying calibration with no dependence on competitors.
    """
    identities = {systems.get(n, {}).get("_set_id") for n in ("INPUT", "REF")}
    subsets = {
        systems.get(n, {}).get("_subset", "all") for n in ("INPUT", "REF")
    }
    if len(identities) != 1 or None in identities or len(subsets) != 1:
        raise ValueError("INPUT and REF must identify the same set and subset")
    if parent is not None and (parent.get("set_id") != next(iter(identities)) or
                               parent.get("subset") != "all" or
                               parent.get("anchors_sha256") != digest(policy) or
                               parent.get("sealed") != sealed):
        raise ValueError(
            "a panel inherits only the whole-set calibration of its own build, "
            "policy and scope")
    scopes = {bool(systems[n].get("_sealed")) for n in ("INPUT", "REF")}
    if len(scopes) != 1:
        raise ValueError("INPUT and REF disagree on private/public scope")
    # the scorers both rows were read by must be one implementation each; REF
    # has no deletion sidecar (real recordings have no clean reference)
    input_scorers = systems["INPUT"].get("_scorers") or {}
    ref_scorers = systems["REF"].get("_scorers") or {}
    differ = sorted(k for k in set(input_scorers) & set(ref_scorers)
                    if input_scorers[k] != ref_scorers[k])
    if differ:
        raise ValueError(
            "INPUT and REF were scored by different implementations"
            f" ({', '.join(differ)})")
    sealed = sealed or next(iter(scopes))
    points = {}
    for axis in axes.values():
        for rows in axis.values():
            for condition, metric, direction in rows:
                key = row_key(condition, metric)
                if key in points:
                    continue
                rule = rule_for(policy, condition, metric)
                if rule["direction"] != direction:
                    raise ValueError(
                        f"axis/anchor direction mismatch for {key}")
                low = _endpoint(rule["floor"], systems, condition, metric,
                                parent is None)
                high = _endpoint(rule["ceiling"], systems, condition, metric,
                                 parent is None)
                fallbacks = []
                for side in ("floor", "ceiling"):
                    if direction * (high - low) > policy["minimum_gap"]:
                        break
                    if "fallback" in rule[side]:
                        if side == "floor":
                            low = float(rule[side]["fallback"])
                        else:
                            high = float(rule[side]["fallback"])
                        fallbacks.append(side)
                if direction * (high - low) <= policy["minimum_gap"]:
                    raise ValueError(
                        f"degenerate anchors for {key}; declare a new policy")
                points[key] = {
                    "floor": low,
                    "ceiling": high,
                    "direction": direction,
                    "fallbacks": fallbacks,
                }
    items = systems["INPUT"].get("_deletion", {}).get("items", {})
    cohort = {}
    for uid, counts in sorted(items.items()):
        # bool is an int subclass, and a flag is not a count
        if (len(counts) != 2 or counts[0] != 0 or isinstance(counts[1], bool) or
                not isinstance(counts[1], int) or counts[1] <= 0):
            raise ValueError(
                "INPUT deletion cohort must have positive counts and zero"
                " deletions")
        cohort[uid] = counts[1]
    if not cohort:
        raise ValueError(
            "INPUT needs a primary-anchored realx sidecar to freeze the cohort")
    data = {
        "schema_version": 1,
        "anchors_version": policy["version"],
        "anchors_sha256": digest(policy),
        "set_id": next(iter(identities)),
        "subset": next(iter(subsets)),
        "sealed": sealed,
        "points": points,
        "axes_sha256": digest(axes),
        "deletion_cohort": cohort,
        "cohort_sha256": digest(cohort),
        # the implementations whose INPUT and REF scores fixed the endpoints: a
        # board read by another scorer is not on this scale
        "scorers": systems["INPUT"].get("_scorers") or {},
    }
    if parent is not None:
        data["inherits"] = parent["calibration_sha256"]
    data["calibration_sha256"] = digest(data)
    return data


def load(path: str | pathlib.Path, policy: dict | None = None) -> dict:
    """Returns the frozen calibration read from ``path``.

    Refuses version, definition, or content drift.
    """
    policy = definitions() if policy is None else policy
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if data.get("anchors_version") != policy["version"] or data.get(
            "anchors_sha256") != digest(policy):
        raise ValueError(
            "anchor version/definition mismatch; recalibrate under a new"
            " version")
    expected = digest({
        k: v for k, v in data.items() if k != "calibration_sha256"
    })
    if data.get("calibration_sha256") != expected:
        raise ValueError("calibration contents do not match their digest")
    return data


def validate_scores(calibration: dict, systems: dict) -> None:
    """Refuse scores that do not match the calibration.

    These are scores from another set, subset, anchor policy or scorer
    implementation than the calibration's.
    """
    bound = calibration.get("scorers") or {}
    for name, score in systems.items():
        mine = score.get("_scorers") or {}
        differ = [
            k for k, v in bound.items() if v and mine.get(k) and mine[k] != v
        ]
        if differ:
            raise ValueError(
                f"{name}: scored by another implementation than the"
                f" calibration's ({', '.join(differ)}): re-score, or"
                " recalibrate")
        if (score.get("_set_id") != calibration["set_id"] or
                score.get("_subset", "all") != calibration["subset"]):
            raise ValueError(
                f"{name}: score set/subset differs from calibration")
        if (score.get("_anchors_version") != calibration["anchors_version"] or
                score.get("_anchors_sha256") != calibration["anchors_sha256"]):
            raise ValueError(
                f"{name}: anchor version mismatch or unstamped legacy scores")


def normalize(value: float, point: dict) -> float:
    """Returns the fraction of a declared metric gap closed.

    The fraction is saturated to [0, 1].
    """
    if not math.isfinite(value):
        return float("nan")
    fraction = (value - point["floor"]) / (point["ceiling"] - point["floor"])
    return min(1.0, max(0.0, fraction))


def composite(scores: dict, members: dict | None,
              aside: tuple | list = ()) -> float:
    """Returns the composite of one system's axis scores.

    It is nan unless every axis is present.

    Args:
        scores: {axis: score in 0-1}, the axes beside the composite included.
        members: ``parts`` of the registry, or None for the plain mean of the
            axes inside.
        aside: The axes scored beside the composite, which it does not read.
    """
    values = list(scores.values())
    if not values or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return float("nan")
    inside = {a: v for a, v in scores.items() if a not in aside}
    if not inside:
        return float("nan")
    if not members:
        return sum(inside.values()) / len(inside)
    if sorted(a for part in members.values() for a in part) != sorted(inside):
        raise ValueError(
            "the parts of the composite do not hold the scored axes inside it")
    return sum(
        sum(inside[a]
            for a in part) / len(part)
        for part in members.values()) / len(members)


def part_scores(scores: dict, members: dict | None) -> dict:
    """Returns {part: mean of its axes} for one system.

    It is empty without parts; a part is nan where one of its axes is.
    """
    return {
        name: (
            sum(scores[a]
                for a in part) /
            len(part) if all(
                isinstance(scores.get(a), (int,
                                           float)) and math.isfinite(scores[a])
                for a in part) else float("nan")
        ) for name, part in (members or {}).items()
    }


def migrate(results: str | pathlib.Path, previous: str, policy: dict) -> int:
    """Re-stamp the results computed under anchor version ``previous``.

    That covers every score file and item-record header computed under it.

    A score file holds raw measurements; the anchor policy only scales them. A
    policy change therefore needs no re-scoring, but it must be explicit: the
    previous version is recorded in ``_anchors_migrated_from`` and the frozen
    calibration of that version is removed so a new one is resolved. Files
    stamped with any other version are left alone and will be refused.
    """
    identity, count = stamp(policy), 0
    for path in sorted(pathlib.Path(results).glob("*.json")):
        if path.name == RESOLVED:
            continue
        data = json.loads(path.read_text())
        if (not isinstance(data, dict) or
                data.get("_anchors_version") != previous):
            continue
        data.update(identity)
        data["_anchors_migrated_from"] = list(
            data.get("_anchors_migrated_from", [])) + [previous]
        # a score file keeps the form its scorer gave it: one space of indent,
        # and nan where a measurement is missing (the board reads it as missing)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=1))
        temporary.replace(path)
        count += 1
    for path in sorted(pathlib.Path(results).glob("*.items.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        head = json.loads(lines[0]) if lines else {}
        if head.get("anchors_version") != previous:
            continue
        head.update(
            anchors_version=identity["_anchors_version"],
            anchors_sha256=identity["_anchors_sha256"],
            anchors_migrated_from=list(head.get("anchors_migrated_from", [])) +
            [previous],
        )
        lines[0] = json.dumps(head, sort_keys=True) + "\n"
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text("".join(lines))
        temporary.replace(path)
        count += 1
    frozen = pathlib.Path(results) / RESOLVED
    if (frozen.exists() and
            json.loads(frozen.read_text()).get("anchors_version") == previous):
        frozen.unlink()
    print(
        f"Migrated {count} files from anchors {previous} to {policy['version']}"
    )
    return count


def main() -> None:
    """Freeze a results directory's calibration.

    Legacy scores are imported explicitly, and only if requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--anchors", default=str(DEFAULT))
    parser.add_argument(
        "--view",
        choices=("all", "core", "smoke", "gate", "exact"),
        default="all",
    )
    parser.add_argument("--import-legacy", action="store_true")
    parser.add_argument(
        "--migrate-from",
        default="",
        help=("re-stamp scores computed under this earlier anchor version: raw"
              " measurements do not depend on the anchor policy, only their"
              " scaling does"),
    )
    parser.add_argument(
        "--parent",
        default="",
        help=(
            "a development panel's whole-set calibration, whose verification of"
            " the declared endpoints it inherits (default: the parent"
            " directory's)"),
    )
    parser.add_argument("--sealed", action="store_true")
    args = parser.parse_args()
    # pylint: disable-next=import-outside-toplevel
    import common
    # pylint: disable-next=import-outside-toplevel
    import ranking
    # pylint: disable-next=import-outside-toplevel
    import scorefiles
    policy = definitions(args.anchors)
    parent = None
    if args.view in common.PANELS:
        path = (pathlib.Path(args.parent) if args.parent else
                pathlib.Path(args.results).resolve().parent / RESOLVED)
        if not path.exists():
            raise SystemExit(
                f"{path} is missing: freeze the whole set before one of its"
                " panels")
        parent = load(path, policy)
    if args.migrate_from:
        if args.migrate_from not in {
                c["version"] for c in policy["changelog"][:-1]
        }:
            raise ValueError(
                "--migrate-from names no earlier version of this anchor policy")
        migrate(args.results, args.migrate_from, policy)
    ranking.select_view(args.view)
    systems = scorefiles.load(args.results)
    calibration = resolve(policy, systems, ranking.AXES_BY_TIER, args.sealed,
                          parent)
    pending = []
    for suffix in ("metrics", "nisqa", "realx", "kept"):
        for path in sorted(pathlib.Path(args.results).glob(f"*.{suffix}.json")):
            data = json.loads(path.read_text())
            if data.get("_anchors_version") is None and args.import_legacy:
                data.update(stamp(policy))
                pending.append((path, data))
            validate_scores(calibration, {path.name: data})
    destination = pathlib.Path(args.results) / RESOLVED
    if destination.exists():
        old = load(destination, policy)
        if old != calibration:
            raise ValueError(
                "calibration is already frozen; use a new results directory")
    for path, data in pending:
        json.dumps(data, allow_nan=False)
    for path, data in pending:
        atomic_json(path, data)
    atomic_json(destination, calibration)
    print(
        f"Calibration frozen: anchors {policy['version']}, subset {args.view}")


if __name__ == "__main__":
    main()
