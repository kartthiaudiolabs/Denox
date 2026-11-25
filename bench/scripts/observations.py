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
"""Per-item scoring records with opaque source groups and explicit coverage."""

import ast
import hashlib
import json
import math
import os
import pathlib
import re
from typing import Any

import anchors

# conditions that add nothing to the recording
BEDLESS = ("clean", "nonverbal_clean")
# per-item coherence blocks: a view's lost-block share is derived under its own
# level calibration
BLOCKS = "coh_blocks"


def revive(value: Any) -> Any:
    """The inverse of ``finite``: JSON null back to nan.

    This way pooled reducers see what was measured.
    """
    if value is None:
        return math.nan
    if isinstance(value, dict):
        return {k: revive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [revive(v) for v in value]
    return value


def finite(value: Any) -> Any:
    """Replace nonfinite measurements with JSON null.

    Their missingness is preserved.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def item(row: dict, status: str = "ok") -> dict:
    """Make a record without exporting corpus names or source paths.

    Background groups use the whole recording, not a window offset. A voice
    placed in a room has the room's response as its second source: items that
    share a room are no more independent than items that share a bed. Real
    recordings and the conditions that add nothing to the recording
    (``BEDLESS``) have one independent recording group and no separate draw.
    Missing provenance is marked unverified and cannot support row admission.
    """
    provenance = row.get("provenance", {})
    speech = provenance.get("speech")
    noise = provenance.get("corruptor") or provenance.get("rir")
    foreground = anchors.digest(["foreground", speech or row["id"]])
    real = not row.get("clean_path")
    if real or not noise:
        bed = None
    else:
        bed = anchors.digest(["bed", re.sub(r"@[+-]?[\d.]+s", "", str(noise))])
    return {
        "id":
            row["id"],
        "condition":
            row["condition"],
        "family":
            row.get("family", row["condition"].split("@")[0]),
        "language":
            row.get("language"),
        "speaker":
            anchors.digest(
                ["speaker",
                 row.get("speaker") or speech or row["id"]]),
        "speech_source":
            foreground,
        "bed_source":
            bed,
        "ladder_key":
            anchors.digest([foreground, bed, row.get("family")]),
        "groups_verified":
            (isinstance(speech, str) and bool(speech) and
             not any(token in speech for token in (" + ", "|", "[")) and
             (real or isinstance(noise, str) and bool(noise) and
              not any(token in noise for token in (" + ", "|", "[")) or
              row["condition"] in BEDLESS)),
        "output_status":
            status,
        "metrics": {},
        "measurements": {},
    }


def capture(record: dict, accumulated: dict, before: dict) -> None:
    """Preserve the exact observations the existing condition reducers use."""
    raw = {
        k: values[before.get(k, 0):]
        for k, values in accumulated.items()
        if len(values) > before.get(k, 0)
    }
    record["measurements"] = finite(raw)
    metrics = {}
    aliases = {
        "kept": "kept_mean",
        "kept_nb": "kept_nb_mean",
        "band": "band_kept_db",
        "band_vs_in": "band_vs_in_db",
        "dnsmos": "dnsmos_ovr",
    }
    for key, values in raw.items():
        if key == "terc":
            metrics.update({
                f"kept_q{i + 1}": v for i, v in enumerate(values[0])
            })
        elif key not in ("seg_drop", BLOCKS):
            metrics[aliases.get(key, key)] = values[0]
    # the deletions are read at a calibration: levels.deleted
    if "seg_drop" in raw:
        metrics["n_segments"] = len(raw["seg_drop"])
    record["metrics"] = finite(metrics)


def write(path: str | pathlib.Path, rows: list[dict], metadata: dict) -> None:
    """Atomically write private observations, including a versioned header."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"type": "metadata", "schema_version": 1, **metadata}] + rows
    text = "".join(
        json.dumps(finite(r), sort_keys=True, allow_nan=False) + "\n"
        for r in records)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path: str | pathlib.Path) -> tuple[dict, list[dict]]:
    """Read complete, uniquely identified observations and their header."""
    with pathlib.Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if (not records or records[0].get("type") != "metadata" or
            records[0].get("schema_version") != 1):
        raise ValueError("per-item records need a versioned metadata header")
    metadata, rows = records[0], records[1:]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or len(rows) != metadata.get("n_expected"):
        raise ValueError("per-item records are duplicated or truncated")
    if not metadata.get("set_id") or not metadata.get("system"):
        raise ValueError("per-item records need set and system identities")
    return metadata, rows


def sidecar(
    path: str | pathlib.Path,
    rows: list[dict],
    system: str,
    set_id: str,
    subset: str,
    sealed: bool = False,
    scorer_sha256: str | None = None,
    **header,
) -> None:
    """Writes per-item sidecar observations under one normalization identity.

    ``header`` adds fields such as the level calibration the records were read
    at.
    """
    identity = anchors.stamp()
    write(
        path,
        rows,
        {
            "system": system,
            "set_id": set_id,
            "subset": subset,
            "anchors_version": identity["_anchors_version"],
            "anchors_sha256": identity["_anchors_sha256"],
            "n_expected": len(rows),
            "sealed": sealed,
            "scorer_sha256": scorer_sha256,
            **header,
        },
    )


def code_digest(path: str | pathlib.Path) -> str:
    """Returns the sha256 of a Python file's code.

    Its syntax tree without docstrings, so a comment, a docstring or a
    reformatting never changes it and any change of what runs always does.
    """
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(
                node,
            (
                ast.Module,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ) and body and isinstance(body[0], ast.Expr) and
                isinstance(body[0].value, ast.Constant) and
                isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def implementation(*paths: str) -> str:
    """Fingerprints scorer sources once, before an expensive run starts.

    The code of each file (``code_digest``), so only a change of what runs makes
    earlier scores another scorer's.
    """
    return anchors.digest({pathlib.Path(p).name: code_digest(p) for p in paths})
