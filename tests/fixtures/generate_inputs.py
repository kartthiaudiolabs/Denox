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
"""Derives parity inputs from the attributed, checked-in speech example."""

import argparse
import pathlib
import wave

SAMPLE_RATE = 48000
DURATIONS = (3, 11)
SOURCE = (pathlib.Path(__file__).resolve().parents[2] / "demo" / "examples" /
          "sample_14.wav")


def main() -> None:
    """Writes the two input WAVs into the requested output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(SOURCE), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(),
                handle.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("The source must be mono PCM16 audio at 48 kHz.")
        source = handle.readframes(handle.getnframes())
    for seconds in DURATIONS:
        size = seconds * SAMPLE_RATE * 2
        frames = source[:size].ljust(size, b"\0")
        with wave.open(str(args.output / f"{seconds}s.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(frames)


if __name__ == "__main__":
    main()
