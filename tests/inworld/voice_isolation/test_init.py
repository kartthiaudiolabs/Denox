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
"""Package boundaries: the public surface and what importing it must not do."""

import importlib
import subprocess
import sys

import pytest


def test_namespace_import_reads_nicely():
    # pylint: disable-next=import-outside-toplevel
    from inworld import voice_isolation

    assert hasattr(voice_isolation, "VoiceIsolator")
    assert hasattr(voice_isolation, "VoiceIsolatorNetwork")
    assert hasattr(voice_isolation, "VoiceIsolatorConfig")
    for name in ("Denoiser", "DenoiserNetwork", "DenoiserConfig", "Isolator",
                 "IsolatorConfig"):
        assert not hasattr(voice_isolation, name)
    for name in ("denoise", "denoise_many"):
        assert not hasattr(voice_isolation.VoiceIsolator, name)
    assert not hasattr(voice_isolation, "TwoStageModel")
    assert not hasattr(voice_isolation.VoiceIsolator, "enhance")
    for name in voice_isolation.__all__:
        assert hasattr(voice_isolation, name), name


@pytest.mark.parametrize(
    "mod",
    [
        "model",
        "scan",
        "streaming",
        "audio",
        "isolator",
        "registry",
        "cli",
        "doctor",
    ],
)
def test_modules_import_without_optional_extras(mod):
    importlib.import_module(f"inworld.voice_isolation.{mod}")


def test_import_does_not_touch_network():
    """Importing the package must not import heavy optional packages.

    Nor may it hit the hub.
    """
    code = (
        "import sys; import inworld.voice_isolation.isolator; bad = [m for m in"
        " ('demucs', 'transformers', 'gradio', 'mamba_ssm', 'huggingface_hub',"
        " 'safetensors') if m in sys.modules];"
        " print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run(  # the exit status is read below
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ":".join(sys.path),
            "PATH": "/usr/bin:/bin"
        },
    )
    assert r.returncode == 0, r.stdout + r.stderr
