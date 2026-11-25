# Copyright 2026 Inworld AI & Antigravity
"""Core audio isolation engine for the macOS application."""

import dataclasses
import json
import logging
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from inworld import voice_isolation as vi
from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import registry

# Try importing live session from demo
try:
    import demo.live as live_module
except ImportError:
    live_module = None

logger = logging.getLogger("mac_app.engine")

SAMPLE_RATE = 48000
TMP_DIR = pathlib.Path(tempfile.gettempdir()) / "neo_voice_isolator"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def pcm16(wav: np.ndarray) -> np.ndarray:
    """Clips audio to [-1, 1] and converts to int16."""
    return np.round(np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)


def compute_waveform_peaks(audio: np.ndarray, num_bins: int = 400) -> List[List[float]]:
    """Downsamples audio to a list of [min, max] pairs for fast canvas rendering."""
    if audio.size == 0:
        return []
    flat = audio.flatten()
    chunk_size = max(1, len(flat) // num_bins)
    peaks = []
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i + chunk_size]
        if len(chunk) > 0:
            peaks.append([float(np.min(chunk)), float(np.max(chunk))])
    return peaks


@dataclasses.dataclass
class EngineModels:
    offline: vi.VoiceIsolator
    realtime: vi.VoiceIsolator
    device: str
    offline_lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    realtime_lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    def __post_init__(self):
        # On Apple Silicon Metal GPU (MPS), models must not isolate concurrently
        if "mps" in str(self.device).lower():
            self.realtime_lock = self.offline_lock


_ENGINE_MODELS: Optional[EngineModels] = None
_INIT_LOCK = threading.Lock()


def get_engine_models() -> EngineModels:
    """Loads and caches both Voice Isolator models on Metal GPU (MPS)."""
    global _ENGINE_MODELS
    with _INIT_LOCK:
        if _ENGINE_MODELS is not None:
            return _ENGINE_MODELS

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Initializing models on device: %s", device)

        offline = vi.VoiceIsolator.load(
            registry.REPO_ID,
            device=device,
            mode="offline",
            precision="auto"
        )
        realtime = vi.VoiceIsolator.load(
            registry.REPO_ID,
            device=device,
            mode="streaming",
            precision="auto",
            lookahead_frames=2
        )

        _ENGINE_MODELS = EngineModels(
            offline=offline,
            realtime=realtime,
            device=device
        )
        return _ENGINE_MODELS


def get_system_telemetry() -> Dict[str, Any]:
    """Returns real-time system and model telemetry."""
    device_name = "CPU"
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        device_name = "Apple Metal GPU (MPS)"

    chip_info = platform.processor() or "Apple Silicon"
    # Check macOS machine model if possible
    try:
        model_out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if model_out:
            chip_info = model_out
    except Exception:
        pass

    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    cached_models = []
    if cache_dir.exists():
        for p in cache_dir.glob("models--inworld--voice-isolator*"):
            cached_models.append(p.name)

    return {
        "device": device_name,
        "is_mps": mps_available,
        "chip": chip_info,
        "os": f"macOS {platform.mac_ver()[0]} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "sample_rate": SAMPLE_RATE,
        "precision": "FP32 (MPS Native)",
        "models": {
            "offline": "voice-isolator-v1",
            "realtime": "voice-isolator-v1-realtime",
            "lookahead_ms": 20,
        },
        "cached_models": cached_models
    }


def isolate_file(
    file_path: str,
    model_choice: str = "offline",
    offline_passes: int = 1,
    precision: str = "auto"
) -> Dict[str, Any]:
    """Performs neural voice isolation on a given audio file and extracts background."""
    models = get_engine_models()
    path = pathlib.Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load audio
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1, dtype=np.float32)

    # Resample to 48kHz if needed
    if rate != SAMPLE_RATE:
        mono_resampled = vi.resample(torch.from_numpy(mono), rate, SAMPLE_RATE).numpy()
    else:
        mono_resampled = mono

    duration_sec = len(mono_resampled) / SAMPLE_RATE

    start_time = time.perf_counter()

    if model_choice == "realtime":
        with models.realtime_lock:
            voice = models.realtime.isolate(mono_resampled, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
    else:
        with models.offline_lock:
            # Set passes if requested
            old_passes = models.offline.offline_passes
            try:
                models.offline.offline_passes = max(1, int(offline_passes))
                voice = models.offline.isolate(mono_resampled, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
            finally:
                models.offline.offline_passes = old_passes

    if isinstance(voice, torch.Tensor):
        voice = voice.cpu().numpy()

    elapsed = time.perf_counter() - start_time
    rtf = duration_sec / max(elapsed, 1e-6)

    # Background residual = aligned input - voice
    background = mono_resampled - voice

    # Save stems to temporary studio cache
    stem_id = f"stem_{int(time.time() * 1000)}"
    stem_dir = TMP_DIR / stem_id
    stem_dir.mkdir(parents=True, exist_ok=True)

    orig_path = stem_dir / "original.wav"
    vocal_path = stem_dir / "vocal.wav"
    bg_path = stem_dir / "background.wav"

    sf.write(str(orig_path), pcm16(mono_resampled), SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(vocal_path), pcm16(voice), SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(bg_path), pcm16(background), SAMPLE_RATE, subtype="PCM_16")

    # Generate waveform preview data
    peaks_orig = compute_waveform_peaks(mono_resampled, 400)
    peaks_vocal = compute_waveform_peaks(voice, 400)
    peaks_bg = compute_waveform_peaks(background, 400)

    return {
        "stem_id": stem_id,
        "filename": path.name,
        "duration": round(duration_sec, 2),
        "elapsed": round(elapsed, 2),
        "rtf": round(rtf, 1),
        "sample_rate": SAMPLE_RATE,
        "model_used": model_choice,
        "passes": offline_passes,
        "urls": {
            "original": f"/api/audio/{stem_id}/original.wav",
            "vocal": f"/api/audio/{stem_id}/vocal.wav",
            "background": f"/api/audio/{stem_id}/background.wav",
        },
        "paths": {
            "original": str(orig_path),
            "vocal": str(vocal_path),
            "background": str(bg_path),
        },
        "waveforms": {
            "original": peaks_orig,
            "vocal": peaks_vocal,
            "background": peaks_bg,
        }
    }


class LiveStreamManager:
    """Coordinates real-time streaming audio isolation sessions."""

    def __init__(self):
        self._sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str, max_seconds: float = 60.0):
        models = get_engine_models()
        with self._lock:
            if live_module:
                session = live_module.LiveSession(
                    models.realtime,
                    model_lock=models.realtime_lock,
                    max_seconds=max_seconds
                )
                self._sessions[session_id] = session
                return True
        return False

    def push_chunk(self, session_id: str, pcm_data: np.ndarray, rate: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return None

        update = session.push((rate, pcm_data))
        vis = update.visualization
        return {
            "input_seconds": round(update.input_seconds, 2),
            "output_seconds": round(update.output_seconds, 2),
            "finished": update.finished,
            "limit_reached": update.limit_reached,
            "waveform_input": vis.get("input", {}).get("waveform", []),
            "waveform_output": vis.get("output", {}).get("waveform", []),
            "spectrogram_input": vis.get("input", {}).get("spectrogram", []),
            "spectrogram_output": vis.get("output", {}).get("spectrogram", []),
        }

    def finish_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if not session:
            return None

        audio_in, audio_out = session.finish()
        stem_id = f"take_{int(time.time() * 1000)}"
        take_dir = TMP_DIR / stem_id
        take_dir.mkdir(parents=True, exist_ok=True)

        in_path = take_dir / "mic_raw.wav"
        out_path = take_dir / "clean_voice.wav"

        sf.write(str(in_path), pcm16(audio_in), SAMPLE_RATE, subtype="PCM_16")
        sf.write(str(out_path), pcm16(audio_out), SAMPLE_RATE, subtype="PCM_16")

        return {
            "take_id": stem_id,
            "duration": round(len(audio_out) / SAMPLE_RATE, 2),
            "urls": {
                "raw": f"/api/audio/{stem_id}/mic_raw.wav",
                "clean": f"/api/audio/{stem_id}/clean_voice.wav"
            },
            "waveforms": {
                "raw": compute_waveform_peaks(audio_in, 300),
                "clean": compute_waveform_peaks(audio_out, 300)
            }
        }


STREAM_MANAGER = LiveStreamManager()
