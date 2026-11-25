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
"""Shared fixtures: tiny CPU-sized models.

The models exercise every architecture branch: attention and Mamba blocks,
the dual-mode knob with its film, and the plain two-sided model. ``tiny``
hands a test the builders; the model fixtures build a fresh model per test,
so a test that moves the mode knob cannot affect another. ``tiny_ckpt`` saves
the identity model as a release safetensors file, for code that loads a path.
"""

import json
import os
import sys
import types

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "src"),
)

# pylint: disable=wrong-import-position
from inworld.voice_isolation import model

# pylint: enable=wrong-import-position

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

TINY = dict(
    width=16,
    block_types=("transformer", "mamba"),
    num_heads=2,
    n_fft=64,
    hop_length=16,
    band_plan=((6000, 24000),),
    chunk_seconds=1.0,  # one render window for every test clip
    overlap_seconds=0.25,
)
DUAL = dict(dual_mode=True,
            lookahead_frames=0,
            lookahead_frames_choices=(0, 2),
            mode_film=True)
WINDOWED = dict(causal=True,
                lookahead_frames=0,
                lookahead_frames_choices=(0, 2, 4),
                attention_window_frames=8)


def tiny_config(**overrides) -> model.VoiceIsolatorConfig:
    kw = dict(TINY)
    kw.update(overrides)
    return model.VoiceIsolatorConfig(**kw)


def randomize(m: model.VoiceIsolatorNetwork,
              seed: int) -> model.VoiceIsolatorNetwork:
    """Gives the zero-initialised heads and knobs random weights.

    The model then changes its input, so tests can tell its modes apart.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in m.denoise_decoder.heads:
            head[-1].weight.copy_(
                torch.randn(head[-1].weight.shape, generator=g) * 0.05)
        if hasattr(m, "lookahead_embedding"):
            m.lookahead_embedding.weight.copy_(
                torch.randn(m.lookahead_embedding.weight.shape, generator=g) *
                0.1)
        if hasattr(m, "mode_film"):
            m.mode_film.weight.copy_(
                torch.randn(m.mode_film.weight.shape, generator=g) * 0.05)
        # The released decoders' condition is far from its zero init; a
        # random one makes the decoder's controls matter in every test.
        condition = m.denoise_decoder.condition
        condition.weight.copy_(
            torch.randn(condition.weight.shape, generator=g) * 0.2)
        condition.bias.copy_(
            torch.randn(condition.bias.shape, generator=g) * 0.1)
    return m


def build_identity(**overrides) -> model.VoiceIsolatorNetwork:
    """A dual-mode model at initialisation: an exact identity in every mode."""
    torch.manual_seed(0)
    kw = dict(DUAL)
    kw.update(overrides)
    return model.VoiceIsolatorNetwork(tiny_config(**kw)).eval()


def build_dual(seed: int = 1, **overrides) -> model.VoiceIsolatorNetwork:
    """A dual-mode model (streaming positions 0 and 2) that changes audio."""
    return randomize(build_identity(**overrides), seed)


def build_offline(seed: int = 1) -> model.VoiceIsolatorNetwork:
    """A plain two-sided model that changes audio."""
    torch.manual_seed(0)
    return randomize(model.VoiceIsolatorNetwork(tiny_config()).eval(), seed)


def build_windowed(seed: int = 1, **overrides) -> model.VoiceIsolatorNetwork:
    """A causal model with an eight-frame attention window and three
    streaming positions (0, 2 and 4) that changes audio."""
    torch.manual_seed(0)
    kw = dict(WINDOWED)
    kw.update(overrides)
    return randomize(model.VoiceIsolatorNetwork(tiny_config(**kw)).eval(), seed)


@pytest.fixture(scope="session")
def tiny():
    """The builders: ``config``, ``identity``, ``dual``, ``offline`` and
    ``windowed``."""
    return types.SimpleNamespace(config=tiny_config,
                                 identity=build_identity,
                                 dual=build_dual,
                                 offline=build_offline,
                                 windowed=build_windowed)


@pytest.fixture
def dual_model():
    return build_dual()


@pytest.fixture
def offline_model():
    return build_offline()


@pytest.fixture
def windowed_model():
    return build_windowed()


@pytest.fixture
def identity_model():
    return build_identity()


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory, tiny):  # pylint: disable=redefined-outer-name
    """An identity-at-init dual-mode model as a release safetensors file."""
    # pylint: disable-next=import-outside-toplevel
    import safetensors.torch

    m = tiny.identity()
    folder = tmp_path_factory.mktemp("ckpt")
    path = folder / "tiny.safetensors"
    safetensors.torch.save_file(
        {
            k: v.contiguous() for k, v in m.state_dict().items()
        },
        str(path),
        metadata={"config": json.dumps(m.config.to_dict())},
    )
    return str(path)
