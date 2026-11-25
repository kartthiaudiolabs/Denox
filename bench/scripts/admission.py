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
"""Evidence-gated, replayable membership of composite axes.

Panel and listening inputs are private. Decisions retain aggregate evidence
and content-addressed run/study identifiers, not checkpoint codenames, item
IDs, transcripts, or source inventories. This evaluates evidence; it does
not manufacture a listening verdict or automatically edit the live axes.
"""

import argparse
import copy
import json
import math
import os
import pathlib

import numpy as np

import anchors
import compare

# BENCH_AXES selects another registry file (a protected copy of an earlier
# revision, say) so an earlier policy's results stay reproducible while the
# live one is revised; the published file is never overwritten for that
DEFAULT = pathlib.Path(
    os.environ.get("BENCH_AXES",
                   pathlib.Path(__file__).resolve().parents[1] / "axes.json"))


def _measured(report: dict) -> bool:
    """Whether the panel separated the regression, not the null, and listeners
    agreed, every check passing."""
    regression = report.get("regression", {}).get("ci95", [])
    null = report.get("null", {}).get("ci95", [])
    human = (report.get("human") or {}).get("ci95", [])
    return (len(regression) == len(null) == len(human) == 2 and
            regression[0] <= regression[1] < 0 and null[0] <= 0 <= null[1] and
            human[0] <= human[1] < 0 and bool(report.get("panel_run")) and
            bool(report.get("study_id")) and all(
                report.get("checks", {}).get(k) is True
                for k in ("E1", "E2", "E3", "E4")))


def _valid_decision(report: dict) -> bool:
    return _measured(report) and report.get("admitted") is True


def _promotable(report: dict) -> bool:
    """A review of a row already in its axis that passed every check, intact."""
    return _measured(report) and report.get("admitted") is False and _intact(
        report)


def _intact(report: dict) -> bool:
    return report.get("evidence_sha256") == anchors.digest({
        k: v for k, v in report.items() if k != "evidence_sha256"
    })


def _failed_panel(report: dict) -> bool:
    regression = report.get("regression", {}).get("ci95", [])
    null = report.get("null", {}).get("ci95", [])
    return (_intact(report) and report.get("checks", {}).get("E1") is False and
            len(regression) == len(null) == 2 and
            (regression[1] >= 0 or not null[0] <= 0 <= null[1]))


def _is_int(value) -> bool:
    """Returns whether ``value`` is an int; a bool, its subclass, is not."""
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_removal(report: dict, row: dict) -> bool:
    """Returns whether a removal rests on two consecutive failed panels.

    Both read the row removed under this registry, on one build of the set
    other than the previous one (so after a set change), in two panel runs.
    """
    failures = report.get("failures", [])
    if (not _intact(report) or len(failures) != 2 or
            not report.get("previous_set_id") or
            not all(_failed_panel(r) for r in failures)):
        return False
    if not all(
            r.get("axis_sha256") == report["axis_sha256"] and
            _is_int(r.get("sequence")) for r in failures):
        return False
    first, second = failures
    return (first.get("set_id") == second.get("set_id") and
            first.get("set_id") != report["previous_set_id"] and
            second.get("sequence") == first.get("sequence", -2) + 1 and
            all(r.get("panel_run") and r.get("row") == row for r in failures)
            and first["panel_run"] != second["panel_run"])


def _redefine(views: dict, change: dict, prior: dict) -> None:
    """Apply a maintainer's redefinition of what one axis measures to each view.

    Admission adds a test condition on panel and listening evidence. A
    redefinition is another kind of change: the maintainer decides that an axis
    was scoring the wrong quantity and states the rows that replace it. It is
    recorded as such, never dressed up as an admission: it names who decided,
    the public note that shows the board under both definitions, the registry it
    was made against, and the complete rows of every tier.
    """
    rows = change.get("rows")
    if (change.get("axis_sha256") != anchors.digest(prior) or
            not change.get("reason") or
            change.get("decided_by") != "maintainer" or
            not change.get("note") or not isinstance(rows, dict) or not rows):
        raise ValueError(
            "a redefinition names the maintainer, a reason, a public note and"
            " the registry it was made against")
    for tiers in views.values():
        if set(rows) != set(tiers):
            raise ValueError("a redefinition states the rows of every tier")
        for tier, new in rows.items():
            axis = change.get("axis")
            if axis not in tiers[tier]:
                raise ValueError("a redefinition names an existing axis")
            valid = new and all(
                isinstance(r, list) and len(r) == 3 and r[2] in (-1, 1) and all(
                    isinstance(v, str) and v for v in r[:2]) for r in new)
            if not valid or len({tuple(r[:2]) for r in new}) != len(new):
                raise ValueError(
                    "redefined rows are distinct [condition, metric, direction]"
                )
            if [list(r) for r in new] == tiers[tier][axis]:
                raise ValueError("a redefinition changes the axis")
            tiers[tier][axis] = [list(r) for r in new]


def _valid_rows(rows) -> bool:
    return (bool(rows) and all(
        isinstance(r, list) and len(r) == 3 and r[2] in (-1, 1) and all(
            isinstance(v, str) and v
            for v in r[:2])
        for r in rows) and len({tuple(r[:2]) for r in rows}) == len(rows))


def _revise(views: dict, change: dict, prior: dict) -> None:
    """Applies a maintainer's revision of the composite.

    A revision may change several axes at once, add a new one among them, give
    each view rows of its own, and change how the axes combine.

    A redefinition restates one axis with the same rows in every view. A
    revision is the larger decision: what the composite covers changed. It names
    who decided, the public note that shows the board under both definitions and
    the registry it was made against; it states the complete rows of every axis
    it touches, in every view and tier it touches, because a view that lacks a
    condition cannot take the row; a view it leaves out keeps its rows. It may
    state the parts of the composite (see ``parts``), and it may move an axis
    into the composite or beside it without restating any row.
    """
    stated = change.get("axes")
    # a revision that restates no rows must still change what the composite
    # covers: which axes it reads, and in which parts
    moves = (("parts" in change and change["parts"] != (parts(prior) or {})) or
             ("outside" in change and change["outside"] != outside(prior)) or
             ("boards" in change and change["boards"] != boards(prior)) or
             ("assign" in change and change["assign"] != assignment(prior)))
    if (change.get("axis_sha256") != anchors.digest(prior) or
            not change.get("reason") or
            change.get("decided_by") != "maintainer" or
            not change.get("note") or not isinstance(stated, dict) or
            not (stated or moves) or not set(stated) <= set(views)):
        raise ValueError(
            "a revision names the maintainer, a reason, a public note, the"
            " registry it was made against and the axes of the views it"
            " changes, or the axes it moves into or out of the composite")
    for view, stated_tiers in stated.items():
        tiers = views[view]
        if (not stated_tiers or not set(stated_tiers) <= set(tiers) or
                not all(stated_tiers.values())):
            raise ValueError(
                "a revision states the axes of the tiers it changes")
        for tier, axes in stated_tiers.items():
            for axis, rows in axes.items():
                if not _valid_rows(rows):
                    raise ValueError(
                        "revised rows are distinct [condition, metric,"
                        " direction]")
                if [list(r) for r in rows] == tiers[tier].get(axis):
                    raise ValueError("a revision changes every axis it names")
                tiers[tier][axis] = [list(r) for r in rows]
    _check_layout(views, change, prior)


def _check_layout(views: dict, change: dict, prior: dict) -> None:
    """Raises ValueError unless what ``change`` states of how the axes combine
    holds over ``views``: its tie bands, the axes beside the composite, the
    parts, the boards and the assignment rule, each stated or inherited from
    ``prior``. The baseline of a registry is checked the same way."""
    names = [set(axes) for tiers in views.values() for axes in tiers.values()]
    if any(n != names[0] for n in names):
        raise ValueError("every view and tier scores the same axes")
    bands = change.get("tie_bands")
    if bands is not None and (not isinstance(bands, dict) or not bands or any(
            axis not in names[0] or isinstance(band, bool) or
            not isinstance(band, (int, float)) or not 0 < band < 1
            for axis, band in bands.items())):
        raise ValueError(
            "a tie band names a scored axis and a gap between 0 and 1")
    aside = change.get("outside", [])
    if (not isinstance(aside, list) or any(a not in names[0] for a in aside) or
            len(set(aside)) != len(aside) or len(aside) >= len(names[0])):
        raise ValueError(
            "axes outside the composite are scored axes, and not all of them")
    members = change.get("parts")
    if members is None and "outside" in change:
        # the parts inherited from before must still hold every axis that is
        # not beside the composite: an axis in neither has no place on the board
        members = parts(prior)
    if members is not None:
        listed = ([a for part in members.values() for a in part] if isinstance(
            members, dict) else [])
        if not listed or sorted(listed) != sorted(
                a for a in names[0] if a not in aside):
            raise ValueError(
                "the parts of a composite hold every axis inside it exactly"
                " once")
    declared = change.get("boards")
    if declared is not None:
        # each board is the mean of the parts it names; together they name
        # every part exactly once, so no part is counted twice or dropped
        of_parts = list((members if members is not None else
                         (parts(prior) or {})).keys())
        named = ([p for board in declared.values() for p in board]
                 if isinstance(declared, dict) and declared and
                 all(declared.values()) else [])
        if not named or sorted(named) != sorted(of_parts):
            raise ValueError(
                "the boards of a registry hold every part exactly once")
    rule = change.get("assign", assignment(prior))
    if rule is not None:
        # the rule that sends each system to one board: stated or inherited, it
        # must name the boards in force
        _check_assignment(
            rule, declared if declared is not None else boards(prior) or {})


# what an assignment rule may read: a diagnostic of each system's score file
ASSIGNABLE = ("band_above_input_db",)


def _check_assignment(rule, declared: dict) -> None:
    """Raises ValueError unless ``rule`` sends every system to one of the two
    boards in force (see ``assignment``)."""
    names = set(declared)
    if (not isinstance(rule, dict) or
            set(rule) != {"by", "above_db", "board", "otherwise"} or
            rule["by"] not in ASSIGNABLE or
            isinstance(rule["above_db"], bool) or
            not isinstance(rule["above_db"], (int, float)) or
            not math.isfinite(rule["above_db"]) or
            rule["board"] == rule["otherwise"] or
        {rule["board"], rule["otherwise"]} != names):
        raise ValueError(
            "an assignment reads one declared diagnostic against a threshold"
            " and sends each system to one of the two boards in force")


def parts(registry: dict) -> dict | None:
    """Returns how the axes combine: {part: [axes]}, or None.

    The parts come from the latest change that states them; the result is None
    when no change does.

    Without parts the composite is the mean of the axes, so adding an axis that
    rewards removing takes weight from the axes that reward keeping. With parts
    it is the mean of the parts, each the mean of its axes: what a system
    removes, what it keeps and how the result sounds weigh the same however many
    axes read each.
    """
    stated = [c["parts"] for c in registry.get("history", []) if c.get("parts")]
    return {k: list(v) for k, v in stated[-1].items()} if stated else None


def outside(registry: dict) -> list:
    """Returns the axes scored and published beside the composite.

    They come from the latest change that states them; the list is empty when
    every axis is in the composite.

    A row the task contracts keep out of the composite (a room, which the
    denoising task leaves as recorded) can still be measured and shown for every
    system: the axis says what each system does with it, and the composite says
    nothing about it.
    """
    stated = [
        c["outside"] for c in registry.get("history", []) if "outside" in c
    ]
    return list(stated[-1]) if stated else []


def boards(registry: dict) -> dict | None:
    """Returns {board: [parts]}: the scores the registry publishes side by side.

    The boards come from the latest change that states them; None when the
    registry publishes one composite of all its parts.

    Each board is the mean of the parts it names, and together the boards name
    every part exactly once: what a system removes, keeps and how it sounds
    make one board, what it restores another, and no part is read twice.
    """
    stated = [
        c["boards"] for c in registry.get("history", []) if c.get("boards")
    ]
    return {k: list(v) for k, v in stated[-1].items()} if stated else None


def assignment(registry: dict) -> dict | None:
    """Returns the rule that decides which board ranks a system, or None.

    The rule comes from the latest change that states it: a system whose
    diagnostic ``by`` (``band_above_input_db``, the most energy it puts above a
    band-limited input's band) exceeds ``above_db`` is ranked on ``board``,
    every other system on ``otherwise``. A system without that diagnostic is
    ranked on neither. Every system is scored on every board; the rule only
    says which board's score is its composite, and the boards rank the
    full-band tier and the 16 kHz tier on the board that orders the table, so
    a 16 kHz system the rule sends elsewhere is not ranked. None when the
    registry publishes no boards or every system ranks on the first.
    """
    stated = [
        c["assign"] for c in registry.get("history", []) if c.get("assign")
    ]
    return dict(stated[-1]) if stated else None


def tie_bands(registry: dict) -> dict:
    """Returns {axis: the gap under which two systems tie on it}.

    The bands come from the latest change that states them.

    A reporting policy, as the composite's own band is: an axis that stretches a
    narrow range of a metric says so, and the board names the systems inside the
    band instead of ordering them.
    """
    stated = [
        c["tie_bands"]
        for c in registry.get("history", [])
        if c.get("tie_bands")
    ]
    return dict(stated[-1]) if stated else {}


def replay(registry: dict) -> dict:
    """Reproduce axes from the baseline and each versioned, evidenced change."""
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported axis registry schema")
    history = registry.get("history", [])
    if not history or history[0].get("action") != "baseline":
        raise ValueError("axis history must begin with an explicit baseline")
    if len({r["version"] for r in history}) != len(history):
        raise ValueError("every membership change requires a new axis version")
    if registry.get("version") != history[-1]["version"]:
        raise ValueError("axis version does not match the changelog")
    views = copy.deepcopy(registry["baseline"])
    if any(k in history[0]
           for k in ("parts", "outside", "boards", "assign", "tie_bands")):
        _check_layout(views, history[0], dict(registry, history=[]))
    for index, change in enumerate(history[1:], 1):
        report = change.get("evidence", {})
        prior = dict(
            registry,
            version=history[index - 1]["version"],
            history=history[:index],
        )
        if change.get("action") == "redefine":
            _redefine(views, change, prior)
            continue
        if change.get("action") == "revise":
            _revise(views, change, prior)
            continue
        if report.get("axis_sha256") != anchors.digest(prior):
            raise ValueError(
                "membership evidence belongs to another registry history")
        if (change.get("panel_run") != report.get("panel_run") or
                change.get("study_id") != report.get("study_id") or
                not change.get("reason") or not change.get("study_id")):
            raise ValueError(
                "membership changes require named panel and listening evidence")
        proposal = report.get("row", {})
        target = views[proposal["view"]][proposal["tier"]][proposal["axis"]]
        row = [proposal[k] for k in ("condition", "metric", "direction")]
        action = change.get("action")
        if action in ("admit", "replace"):
            if not _valid_decision(report) or not _intact(report):
                raise ValueError(
                    "admission evidence is incomplete or was modified")
            if row in target:
                raise ValueError("row is already in the axis")
            if action == "replace":
                old = proposal.get("replaces")
                if old not in target:
                    raise ValueError(
                        "replacement does not identify an existing row")
                target.remove(old)
            elif proposal.get("replaces"):
                raise ValueError(
                    "a replacement must be recorded as a replacement")
            target.append(row)
        elif action == "promote":
            before_aside, before_parts = outside(prior), parts(prior) or {}
            axis, part = change.get("axis"), change.get("part")
            moved = {
                k: list(v) + ([axis] if k == part else [])
                for k, v in before_parts.items()
            }
            if (not _promotable(report) or row not in target or
                    proposal.get("axis") != axis or axis not in before_aside or
                    part not in before_parts or change.get("outside")
                    != [a for a in before_aside if a != axis] or
                    change.get("parts") != moved or
                    change.get("decided_by") != "maintainer" or
                    not change.get("note")):
                raise ValueError(
                    "a promotion moves one axis scored beside the composite"
                    " into one existing part, on an intact review that passed"
                    " every check, and changes nothing else")
        elif action == "remove":
            if not _valid_removal(report, proposal):
                raise ValueError(
                    "removal needs two consecutive failed panels after a set"
                    " change")
            if row not in target or len(target) == 1:
                raise ValueError("cannot remove an absent row or empty an axis")
            target.remove(row)
        else:
            raise ValueError("unsupported membership change")
    return views


def load(path: str | pathlib.Path = DEFAULT) -> dict:
    """Read a registry and validate its entire membership history."""
    registry = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    replay(registry)
    return registry


def _grouped(rows: list, metadata: dict, condition: str, metric: str,
             direction: int) -> dict:
    """Returns {speech source: mean signed value} of one row of a system."""
    groups = {}
    for record in rows:
        if record["condition"] != condition:
            continue
        v = compare.value(record, metric, metadata)
        if math.isfinite(v):
            groups.setdefault(record["speech_source"], []).append(direction * v)
    return {k: float(np.mean(v)) for k, v in groups.items()}


def _axis_and_row(entry) -> tuple:
    """Returns (axis, row) of a redundancy entry: ``[axis, [condition, metric,
    direction]]`` names the axis the row sits in; a bare row has none."""
    if (len(entry) == 2 and isinstance(entry[0], str) and
            isinstance(entry[1], (list, tuple))):
        return entry[0], list(entry[1])
    return None, list(entry)


def _correlation(systems: dict, proposal: dict, existing: list) -> dict:
    pairs = []
    for metadata, rows in systems.values():
        a = _grouped(rows, metadata, proposal["condition"], proposal["metric"],
                     proposal["direction"])
        b = _grouped(rows, metadata, *existing)
        pairs.extend((a[k], b[k]) for k in sorted(set(a) & set(b)))
    if len(pairs) < 20:
        return {"row": existing, "n_pairs": len(pairs), "abs_correlation": None}
    array = np.asarray(pairs)
    if np.std(array[:, 0]) == 0 or np.std(array[:, 1]) == 0:
        return {"row": existing, "n_pairs": len(pairs), "abs_correlation": None}
    return {
        "row": existing,
        "n_pairs": len(pairs),
        "abs_correlation": float(abs(np.corrcoef(array.T)[0, 1])),
    }


def evaluate(
    panel: dict,
    systems: dict,
    proposal: dict,
    registry: dict,
    policy: dict,
    study: dict | None,
    n_boot: int = 2000,
    seed: int = 0,
    review_existing: bool = False,
    redundancy_rows: list | None = None,
) -> dict:
    """Evaluate E1–E4; missing human or redundancy evidence admits no row.

    ``review_existing`` reviews a row already in the axis (for a removal or a
    promotion) and never admits it. Redundancy (E3) is read against
    ``redundancy_rows`` when given (a promotion: every row the composite reads,
    each as ``[axis, row]`` so that the same measurement in another axis counts)
    and otherwise against the axis's rows, never against the row itself.
    """
    limits = registry["policy"]
    replayed = replay(registry)
    current = replayed[proposal["view"]][proposal["tier"]][proposal["axis"]]
    condition = proposal["condition"]
    metric = proposal["metric"]
    direction = proposal["direction"]
    if [condition, metric, direction] in current and not review_existing:
        raise ValueError("candidate row already belongs to the axis")
    roster = panel.get("roster", {})
    if (not roster.get("sha256") or not roster.get("systems") or
            set(panel["ranked_systems"]) != set(roster["systems"])):
        raise ValueError(
            "coverage must include the entire pinned published roster")
    base, regression, null = (
        systems[panel[k]] for k in ("baseline", "regression", "null"))
    for dataset in systems.values():
        compare.aligned(dataset, base)
    identity = base[0]
    expected = anchors.stamp(policy)
    if (identity.get("anchors_version") != expected["_anchors_version"] or
            identity.get("anchors_sha256") != expected["_anchors_sha256"]):
        raise ValueError(
            "panel anchor version differs from the admission policy")
    regression_result = compare.paired(regression, base, condition, metric,
                                       direction, n_boot, seed)
    null_result = compare.paired(null, base, condition, metric, direction,
                                 n_boot, seed)
    e1 = (regression_result["ci95"][1] < 0 and
          null_result["ci95"][0] <= 0 <= null_result["ci95"][1] and
          regression_result["groups_verified"] and
          null_result["groups_verified"] and
          regression_result["cluster_counts"][0] >= limits["min_source_groups"]
          and all(r["matched_cohort"] and r["n_paired"] >= limits["min_items"]
                  for r in (regression_result, null_result)))
    human = None
    if study is not None:
        if (study.get("set_id") != identity["set_id"] or
                study.get("condition") != condition):
            raise ValueError(
                "listening study does not match the panel set and condition")
        for role, dataset in (("baseline", base), ("regression", regression)):
            model = dataset[0].get("render_identity", {}).get("weights_sha256")
            if not model or study.get(f"{role}_sha256") != model:
                raise ValueError(
                    "listening study checkpoint identity does not match the"
                    " panel")
        candidates = {
            r["id"]: r for r in base[1] if r["condition"] == condition
        }
        ratings = study.get("ratings", [])
        seen = set()
        for rating in ratings:
            if rating.get("item_id") not in candidates or rating.get(
                    "preference") not in (-1, 0, 1):
                raise ValueError(
                    "listening rating has an unknown item or invalid preference"
                )
            key = (rating["item_id"], rating.get("listener"))
            if not key[1] or key in seen:
                raise ValueError(
                    "listening ratings need distinct item/listener pairs")
            seen.add(key)
        if (len({r["item_id"] for r in ratings}) >= limits["min_human_items"]
                and len({r["listener"] for r in ratings}) >= 2):
            values = np.array([r["preference"] for r in ratings])
            groups = [[candidates[r["item_id"]].get(k)
                       for r in ratings]
                      for k in compare.GROUPS]
            groups.append([r["listener"] for r in ratings])
            human = compare.interval(
                lambda w: (float(w @ values / w.sum())
                           if w.sum() else float("nan")),
                groups,
                n_boot,
                seed,
            )
    e2 = human is not None and human["ci95"][1] < 0
    own = [condition, metric, direction]
    compared = []
    for entry in (current if redundancy_rows is None else redundancy_rows):
        axis, row = _axis_and_row(entry)
        # the candidate's own row is left out; the same measurement in another
        # axis of the composite is a duplicate and stays in
        if row == own and axis in (None, proposal.get("axis")):
            continue
        compared.append(row)
    redundancy = [_correlation(systems, proposal, row) for row in compared]
    replacement = proposal.get("replaces")
    e3 = bool(redundancy) and all(r["abs_correlation"] is not None and (
        (r["abs_correlation"] >= limits["max_redundancy"]) if r["row"] ==
        replacement else (r["abs_correlation"] < limits["max_redundancy"]))
                                  for r in redundancy)
    if replacement is not None and replacement not in [
            list(r) for r in current
    ]:
        e3 = False
    coverage = {}
    for name in panel["ranked_systems"]:
        metadata, rows = systems[name]
        selected = [r for r in rows if r["condition"] == condition]
        coverage[anchors.digest(name)] = sum(
            math.isfinite(compare.value(r, metric, metadata)) and
            r.get("groups_verified", False) for r in selected)
    try:
        rule = anchors.rule_for(policy, condition, metric)
        anchored = rule["direction"] == direction
    except ValueError:
        anchored = False
    e4 = (bool(coverage) and
          all(n >= limits["min_items"] for n in coverage.values()) and anchored)
    checks = {"E1": bool(e1), "E2": bool(e2), "E3": bool(e3), "E4": bool(e4)}
    result = {
        "schema_version":
            1,
        "row":
            proposal,
        "set_id":
            identity["set_id"],
        "anchors_version":
            policy["version"],
        "anchors_sha256":
            anchors.digest(policy),
        "axis_version":
            registry["version"],
        "axis_sha256":
            anchors.digest(registry),
        "panel_run":
            anchors.digest({
                "panel": panel,
                "identities": {
                    role: systems[panel[role]][0]
                    for role in ("baseline", "regression", "null")
                },
            }),
        "study_id":
            anchors.digest(study) if study is not None else None,
        "regression":
            regression_result,
        "null":
            null_result,
        "human":
            human,
        "redundancy":
            redundancy,
        "coverage":
            coverage,
        "checks":
            checks,
        "roster_sha256":
            roster["sha256"],
        "sequence":
            panel.get("sequence"),
        "admitted":
            all(checks.values()) and not review_existing,
    }
    result["evidence_sha256"] = anchors.digest(result)
    return result


def record_change(registry: dict, report: dict, version: str,
                  reason: str) -> dict:
    """Stage an admitted membership change.

    The change carries its exact panel/study evidence.
    """
    if (not _valid_decision(report) or
            report.get("axis_sha256") != anchors.digest(registry) or
            report.get("axis_version") != registry["version"]):
        raise ValueError(
            "decision is not admitted or belongs to another axis registry")
    if (not version or version in {r["version"] for r in registry["history"]} or
            not reason):
        raise ValueError("a new unique version and a reason are required")
    result = copy.deepcopy(registry)
    result["version"] = version
    result["history"].append({
        "version": version,
        "action": "replace" if report["row"].get("replaces") else "admit",
        "reason": reason,
        "panel_run": report["panel_run"],
        "study_id": report["study_id"],
        "evidence": report,
    })
    replay(result)
    return result


def record_redefinition(registry: dict, axis: str, rows: dict, version: str,
                        reason: str, note: str) -> dict:
    """Stage a maintainer's redefinition of one axis against the live registry.

    Args:
        registry: The live axis registry.
        axis: The axis whose meaning changes.
        rows: {tier: [[condition, metric, direction], ...]}, the complete rows
            of every tier.
        version: A new, unique membership version.
        reason: Changelog rationale.
        note: Repository path of the public note showing the board under both
            definitions.

    Returns:
        The registry with the change appended, already replayed.
    """
    if not version or version in {r["version"] for r in registry["history"]}:
        raise ValueError("a new unique version is required")
    result = copy.deepcopy(registry)
    result["version"] = version
    result["history"].append({
        "version": version,
        "action": "redefine",
        "axis": axis,
        "rows": rows,
        "reason": reason,
        "decided_by": "maintainer",
        "note": note,
        "panel_run": None,
        "study_id": None,
        "axis_sha256": anchors.digest(registry),
    })
    replay(result)
    return result


def record_revision(
    registry: dict,
    axes: dict,
    version: str,
    reason: str,
    note: str,
    members: dict | None = None,
    bands: dict | None = None,
    aside: list | None = None,
    published: dict | None = None,
    assign: dict | None = None,
) -> dict:
    """Stage a maintainer's revision of the composite against the live registry.

    Args:
        registry: The live axis registry.
        axes: {view: {tier: {axis: [[condition, metric, direction], ...]}}}, the
            complete rows of every axis that changes or is new; ``{}`` when the
            revision only moves axes into or out of the composite (``members``
            and ``aside``).
        version: A new, unique membership version.
        reason: Changelog rationale.
        note: Repository path of the public note showing the board under both
            definitions.
        members: {part: [axes]} when the revision states how the axes combine.
        bands: {axis: tie band} when the revision declares one.
        aside: The axes scored beside the composite, not in it, when the
            revision states them.
        published: {board: [parts]} when the revision states which scores the
            registry publishes side by side (see ``boards``).
        assign: The rule that decides which board ranks a system, when the
            revision states it (see ``assignment``).

    Returns:
        The registry with the change appended, already replayed.
    """
    if not version or version in {r["version"] for r in registry["history"]}:
        raise ValueError("a new unique version is required")
    result = copy.deepcopy(registry)
    result["version"] = version
    change = {
        "version": version,
        "action": "revise",
        "axes": axes,
        "reason": reason,
        "decided_by": "maintainer",
        "note": note,
        "panel_run": None,
        "study_id": None,
        "axis_sha256": anchors.digest(registry),
    }
    if members is not None:
        change["parts"] = members
    if bands is not None:
        change["tie_bands"] = bands
    if aside is not None:
        change["outside"] = aside
    if published is not None:
        change["boards"] = published
    if assign is not None:
        change["assign"] = assign
    result["history"].append(change)
    replay(result)
    return result


def record_promotion(registry: dict, report: dict, part: str, version: str,
                     reason: str, note: str) -> dict:
    """Stage the promotion of an axis scored beside the composite into a part.

    The axis's row was reviewed on a panel and a listening study
    (``evaluate(..., review_existing=True)``, redundancy read against every row
    the composite reads) and passed all four checks. The change carries that
    evidence; the axis leaves ``outside`` and joins ``part``; no row changes.

    Raises:
        ValueError: The review did not pass, was modified or belongs to another
            registry; the axis is not beside the composite; the part does not
            exist; the version is not new; the reason or the note is missing.
    """
    axis = (report.get("row") or {}).get("axis")
    aside, members = outside(registry), parts(registry) or {}
    if (not _promotable(report) or
            report.get("axis_sha256") != anchors.digest(registry) or
            report.get("axis_version") != registry["version"]):
        raise ValueError(
            "the review did not pass every check, was modified, or belongs to"
            " another axis registry")
    if axis not in aside or part not in members:
        raise ValueError("only an axis scored beside the composite is promoted,"
                         " into an existing part")
    if (not version or version in {r["version"] for r in registry["history"]} or
            not reason or not note):
        raise ValueError(
            "a new unique version, a reason and a public note are required")
    result = copy.deepcopy(registry)
    result["version"] = version
    result["history"].append({
        "version": version,
        "action": "promote",
        "axis": axis,
        "part": part,
        "reason": reason,
        "decided_by": "maintainer",
        "note": note,
        "panel_run": report["panel_run"],
        "study_id": report["study_id"],
        "evidence": report,
        "parts": {
            k: list(v) + ([axis] if k == part else [])
            for k, v in members.items()
        },
        "outside": [a for a in aside if a != axis],
    })
    replay(result)
    return result


def record_removal(
    registry: dict,
    failures: list[dict],
    previous_set_id: str,
    version: str,
    reason: str,
) -> dict:
    """Stage a removal backed by two sequential failures.

    The failures come after a set replacement.
    """
    if (len(failures) != 2 or not version or not reason or
            version in {r["version"] for r in registry["history"]}):
        raise ValueError(
            "removal requires two reports, a new version, and a reason")
    report = {
        "axis_sha256": anchors.digest(registry),
        "row": failures[0]["row"],
        "failures": failures,
        "previous_set_id": previous_set_id,
        "study_id": failures[-1].get("study_id"),
        "panel_run": failures[-1].get("panel_run"),
    }
    report["evidence_sha256"] = anchors.digest(report)
    result = copy.deepcopy(registry)
    result["version"] = version
    result["history"].append({
        "version": version,
        "action": "remove",
        "reason": reason,
        "panel_run": report["panel_run"],
        "study_id": report["study_id"],
        "evidence": report,
    })
    replay(result)
    return result


def main() -> None:
    """Produce a reviewable decision.

    Absence of a study writes a rejected decision.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",
                        required=True,
                        help="private panel configuration")
    parser.add_argument(
        "--board",
        required=True,
        help=("the published board (leaderboard.py --json-out), the roster"
              " snapshot used for coverage and redundancy"),
    )
    parser.add_argument("--row", required=True, help="proposed row JSON")
    parser.add_argument("--study", help="human paired-preference study JSON")
    parser.add_argument("--axes", default=str(DEFAULT))
    parser.add_argument("--anchors", default=str(anchors.DEFAULT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--review-existing", action="store_true")
    parser.add_argument("--registry-out",
                        help="stage the admitted registry at this path")
    parser.add_argument("--version",
                        help="new membership version when staging a change")
    parser.add_argument("--reason",
                        help="changelog rationale when staging a change")
    parser.add_argument(
        "--promote-into",
        default="",
        help=("review the row of an axis scored beside the composite and, when"
              " it passes, stage its axis's promotion into this part"))
    parser.add_argument("--note",
                        default="bench/METRICS.md",
                        help="public note of a promotion")
    args = parser.parse_args()
    panel = json.loads(pathlib.Path(args.panel).read_text(encoding="utf-8"))
    board = json.loads(pathlib.Path(args.board).read_text(encoding="utf-8"))
    panel["roster"] = {
        "sha256":
            anchors.digest(board),
        "systems":
            sorted(
                n for n, r in board["systems"].items()
                if r.get("composite") is not None and r.get("tier") in ("full",
                                                                        "nb")),
    }
    names = set(panel["ranked_systems"]) | {
        panel[k] for k in ("baseline", "regression", "null")
    }
    systems = {
        n: compare.load_system(panel["results"], n) for n in sorted(names)
    }
    if any(d[0].get("sealed") for d in systems.values()):
        raise SystemExit(
            "sealed confirmation material cannot be used to select axis"
            " membership")
    proposal = json.loads(pathlib.Path(args.row).read_text(encoding="utf-8"))
    registry = load(args.axes)
    against = None
    if args.promote_into:
        tier = replay(registry)[proposal["view"]][proposal["tier"]]
        against = [[axis, row]
                   for part in (parts(registry) or {}).values()
                   for axis in part
                   for row in tier.get(axis, [])]
    result = evaluate(
        panel,
        systems,
        proposal,
        registry,
        anchors.definitions(args.anchors),
        json.loads(pathlib.Path(args.study).read_text(
            encoding="utf-8")) if args.study else None,
        args.bootstrap,
        review_existing=args.review_existing or bool(args.promote_into),
        redundancy_rows=against,
    )
    anchors.atomic_json(args.out, result)
    if args.registry_out:
        staged = (record_promotion(registry, result, args.promote_into,
                                   args.version, args.reason, args.note)
                  if args.promote_into else record_change(
                      registry, result, args.version, args.reason))
        anchors.atomic_json(args.registry_out, staged)
    print(
        json.dumps({
            "admitted": result["admitted"],
            "checks": result["checks"]
        }))


if __name__ == "__main__":
    main()
