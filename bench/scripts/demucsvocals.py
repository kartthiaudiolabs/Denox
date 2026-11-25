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
"""The htdemucs vocals stem, the benchmark's music-separation reference row."""

import hashlib
import os

import torch

from inworld.voice_isolation import audio as audio_lib

STEM_CACHE_ENV = "VOICE_ISOLATION_STEM_CACHE"


class DemucsVocals(torch.nn.Module):
    """Extracts the htdemucs vocals stem at 48 kHz, deterministically.

    Determinism matters for a reference row: a randomly time-shifted stem
    (the demucs default, shifts=1) would make the renders differ between
    runs. An optional content-addressed stem cache directory can be set via
    the VOICE_ISOLATION_STEM_CACHE env var; an entry is keyed by the audio's
    samples, the demucs version and the device type, and holds plain tensors
    only.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        try:
            # pylint: disable-next=import-outside-toplevel
            import demucs.apply
            # pylint: disable-next=import-outside-toplevel
            import demucs.pretrained
        except ImportError as e:
            raise ImportError(
                "The Demucs reference row requires the [demucs] extra: "
                'pip install "inworld-voice-isolation[demucs]"') from e
        self._apply_model = demucs.apply.apply_model
        self.device = torch.device(device)
        self.model = demucs.pretrained.get_model("htdemucs")
        self.model = self.model.to(self.device).eval()
        self.model_sr = int(self.model.samplerate)
        self.vidx = self.model.sources.index("vocals")
        self._cache_salt = (f"htdemucs|demucs {demucs.__version__}|"
                            f"{self.device.type}|{self.model_sr}|"
                            "split overlap 0.25 shifts 0".encode())

    @torch.inference_mode()
    def vocals(self, audio: torch.Tensor) -> torch.Tensor:
        """audio [T] or [1, T] at 48 kHz -> vocals stem [T] at 48 kHz."""
        cache_dir = os.environ.get(STEM_CACHE_ENV, "")
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            data = audio.detach().cpu().float().numpy().tobytes()
            key = hashlib.sha256(self._cache_salt + data).hexdigest()
            cache_path = os.path.join(cache_dir, key + ".pt")
            if os.path.exists(cache_path):
                stem = torch.load(cache_path,
                                  map_location=audio.device,
                                  weights_only=True)
                return stem.squeeze(0)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        mix = audio_lib.resample(audio, audio_lib.SR, self.model_sr)
        if mix.shape[0] == 1:
            mix = mix.repeat(2, 1)
        stems = self._apply_model(
            self.model,
            mix.unsqueeze(0).to(self.device),
            device=self.device,
            split=True,
            overlap=0.25,
            shifts=0,
            progress=False,
        )
        stem = stems[0, self.vidx].mean(0, keepdim=True)
        stem = audio_lib.resample(stem, self.model_sr, audio_lib.SR)
        stem = stem.to(audio.device)
        if cache_path:
            tmp = cache_path + f".tmp{os.getpid()}"
            torch.save(stem.cpu(), tmp)
            os.replace(tmp, cache_path)
        return stem.squeeze(0)
