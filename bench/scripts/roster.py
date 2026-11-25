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
"""The systems a published board ranks: each one's tier and label.

``full``: eligible for the 0-48 kHz goal. ``nb``: a 16 kHz-native model,
ranked in its own tier on the band-limited readings. ``ref``: a reference row,
shown and never ranked. A system missing here is ignored by the board unless
``leaderboard.py --tiers`` names it.
"""

import json

TIERS = {
    "voice_isolator_v0": ("full", "Inworld Voice Isolator (64M, offline)"),
    "voice_isolator_v0_realtime": (
        "full",
        "Inworld Voice Isolator realtime (58M, look-ahead 20 ms)",
    ),
    "clearvoice_mf2_48k": ("full", "ClearerVoice MossFormer2_SE_48K"),
    "dfn3": ("full", "DeepFilterNet3"),
    "fastenhancer_l_48k": ("full", "FastEnhancer-L 48k"),
    "reuse_nemotron": (
        "full",
        "NVIDIA NemotronLabs RE-USE (9.6M, 2026-10)",
    ),
    "reuse_nemotron_laughter": (
        "full",
        "NVIDIA NemotronLabs RE-USE-laughter (9.6M, 2026-10)",
    ),
    "resemble_dn": ("full", "resemble-enhance (denoise)"),
    "urgent26_bsrnn": (
        "full",
        ("URGENT 2026 BSRNN baseline (official, 48k, shipped peak"
         " normalisation)"),
    ),
    "urgent26_bsrnn_flow": (
        "full",
        ("URGENT 2026 FlowSE baseline (official, 48k, without final peak"
         " normalisation)"),
    ),
    "gap_urgenet": ("full", "GAP-URGENet (URGENT 2026 Track 1, 1st objective)"),
    "unipase": (
        "full",
        "UniPASE (TASLP 2026, 16 kHz core, post-net to 48 kHz)",
    ),
    "smule": ("full", "Smule Renaissance Small (10.4M)"),
    "demucs_vocals": ("ref", "Demucs vocals (separation reference)"),
    "INPUT": ("ref", "unprocessed input"),
    "REF": ("ref", "clean reference (paired items only)"),
}


def tiers(path: str | None = None) -> dict:
    """Returns the registered tiers, with those of a ``--tiers`` file added.

    Args:
        path: A JSON file mapping system names to ``[tier, label]``, or None.
    """
    merged = dict(TIERS)
    if path:
        with open(path, encoding="utf-8") as fh:
            merged.update({k: tuple(v) for k, v in json.load(fh).items()})
    return merged


def display_name(name: str) -> str:
    """Returns a system's label without its parenthesised details."""
    return TIERS.get(name, (None, name))[1].split(" (")[0]
