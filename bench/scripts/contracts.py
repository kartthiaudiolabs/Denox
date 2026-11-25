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
"""Task contracts: what every condition of the set holds a system to.

The set asks three different things of a system, and a number only means
something next to the one it was asked. ``PROFILES`` states them, ``FAMILIES``
assigns every condition family to one, ``CONVENTIONS`` lists the changes a
scorer undoes because nobody can hear them. ``AXES`` states what the board's
axes may score: every denoising row, and the rooms of the restoration profile in
one axis of their own, because most recordings are made in rooms and a board
that leaves them out says nothing about most recordings. Whether that axis is
read by a score or shown beside both is the axis registry's decision
(``admission.outside``). ``check_axes`` refuses any other row, so a voice-policy
or restoration row can only enter an axis as a stated decision, never by
accident. A family that is not listed has no contract yet: ``profile`` raises
instead of guessing.
"""

import common

PROFILES = {
    "denoise": (
        "The voice as recorded stays, its laughs, breaths and sighs with it, at"
        " its level, in its band and with its own room; everything that is not"
        " that voice goes."),
    "voice_policy":
        ("Which voices are content is the question, and the boards' own"
         " answer is the denoise contract: the voice the recording is of stays"
         " and a competing talker goes. These rows read the other answers and"
         " stay outside both scores: a singer who is the target stays, a"
         " song's own singer behind speech goes, both sides of a conversation"
         " stay, and so do the several talkers of a long item, who take turns;"
         " for a system that keeps one speaker the conversation and long rows"
         " are the diagnostics of that choice, not targets."),
    "restoration": (
        "The reference is better than the voice in the input: the dry voice out"
        " of a reverberant one. A system that only removes what was added is"
        " not expected to reach it: the real rooms make the restoration score,"
        " which the denoising score does not read."),
}
FAMILIES = {
    "denoise": (
        "ambience",
        "animals",
        "appliance",
        "babble",
        "cheer",
        "cheer_tail",
        "clean",
        "coroom",
        "events",
        "everyday",
        "expressive_tail",
        "household",
        "music",
        "noise_only",
        "nonverbal",
        "nonverbal_clean",
        "quiet",
        "rain",
        "real_dirty",
        "real_everyday",
        "real_multi",
        "real_noisy",
        "real_wild",
        "soft",
        "srsweep_8k",
        "srsweep_16k",
        "srsweep_24k",
        "srsweep_44.1k",
        "srsweep_48k",
        "stationary",
        "talker",
        "tools",
        "traffic",
        "transition",
        "water",
        "whisper",
        "wild",
        "wind",
    ),
    "voice_policy": ("conversation", "long", "music_vocal", "singer"),
    "restoration": ("reverb_real", "reverb_synth"),
}
# What the board's axes may score: {profile: the axes its rows may feed, None
# for any}. An axis named here belongs to that profile alone.
AXES = {"denoise": None, "restoration": ("room",)}
CONVENTIONS = (
    (
        "delay",
        f"up to {1000 * common.MAX_LAG // common.SR} ms",
        "measured and undone before anything is read",
    ),
    ("polarity", "a flipped sign", "undone with the delay"),
    (
        "gain",
        f"up to {common.MAX_LEVEL_CAL_DB:g} dB",
        ("removed from the retention metrics and scored "
         "once, as the level change on clean speech"),
    ),
)
_PROFILE_OF = {
    family: name for name, families in FAMILIES.items() for family in families
}


def profile(condition: str) -> str:
    """Returns the profile of a condition (``babble@-5`` -> ``denoise``).

    Raises:
        KeyError: The condition has no profile.
    """
    family = condition.split("@", 1)[0]
    # pooled pseudo-conditions of the real recordings (``_deletion``)
    if family.startswith("_"):
        return "denoise"
    return _PROFILE_OF[family]


def check_axes(axes: dict, allowed: dict | None = None) -> None:
    """Refuse an axis table whose rows ask for what ``allowed`` does not state.

    Args:
        axes: {axis: [(condition, metric, direction), ...]} of one tier.
        allowed: {profile: axes or None}; ``AXES`` by default.

    Raises:
        ValueError: A row belongs to a profile the axes do not score, or
            sits in an axis its profile may not feed, or shares an axis reserved
            for another profile.
    """
    allowed = AXES if allowed is None else allowed
    reserved = {
        axis: name for name, named in allowed.items() for axis in named or ()
    }
    for axis, rows in axes.items():
        for condition, metric, _ in rows:
            name = profile(condition)
            if name not in allowed:
                raise ValueError(
                    f"{axis} row {condition}/{metric} is a {name} condition:"
                    f" the axes score {', '.join(allowed)} only")
            if allowed[name] is not None and axis not in allowed[name]:
                raise ValueError(
                    f"{axis} row {condition}/{metric} is a {name} condition:"
                    f" the axes score {name} in"
                    f" {', '.join(allowed[name])} only")
            if reserved.get(axis, name) != name:
                raise ValueError(
                    f"{axis} row {condition}/{metric} is a {name} condition in"
                    f" the axis that scores {reserved[axis]}")


def axes_statement(allowed: dict | None = None) -> str:
    """One sentence saying what the axes score, for the board and its JSON."""
    allowed = AXES if allowed is None else allowed
    return "the axes score " + " and ".join(
        (f"{name} rows" if axes is
         None else f"{name} rows in the {', '.join(axes)} axis alone")
        for name, axes in allowed.items())


def by_profile(conditions) -> dict:
    """{profile: sorted conditions} for the conditions given."""
    out = {name: [] for name in PROFILES}
    for condition in sorted(set(conditions)):
        out[profile(condition)].append(condition)
    return out
