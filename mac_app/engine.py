# Copyright 2026 Inworld AI & Antigravity
"""Core audio isolation engine for the macOS application."""

import dataclasses
import hashlib
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from inworld import voice_isolation as vi
from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import registry
from mac_app import dsp as dsp_module

# Try importing live session from demo
try:
    import demo.live as live_module
except ImportError:
    live_module = None

logger = logging.getLogger("mac_app.engine")

SAMPLE_RATE = 48000
TMP_DIR = pathlib.Path(tempfile.gettempdir()) / "denox_audio"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR = TMP_DIR / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)


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


def build_dsp_config(dsp_params: Optional[Dict[str, Any]] = None) -> dsp_module.DenoxMasteringChainConfig:
    """Builds a DenoxMasteringChainConfig from user parameters."""
    cfg = dsp_module.DenoxMasteringChainConfig()
    if not dsp_params:
        return cfg

    # De-Noiser
    dn = dsp_params.get("denoiser")
    if dn and isinstance(dn, dict):
        cfg.denoiser.enabled = bool(dn.get("enabled", True))
        if "threshold" in dn:
            cfg.denoiser.threshold = float(dn["threshold"])
        elif "isolation_threshold" in dn:
            cfg.denoiser.threshold = float(dn["isolation_threshold"])
        if "mode" in dn:
            cfg.denoiser.mode = str(dn["mode"])
        if "reduction_db" in dn:
            cfg.denoiser.reduction_db = float(dn["reduction_db"])
        if "ambient_bleed" in dn:
            cfg.denoiser.ambient_bleed = float(dn["ambient_bleed"])
    elif "isolation_threshold" in dsp_params:
        cfg.denoiser.threshold = float(dsp_params["isolation_threshold"])
    elif "threshold" in dsp_params:
        cfg.denoiser.threshold = float(dsp_params["threshold"])

    # De-Clicker
    dc = dsp_params.get("declicker")
    if dc and isinstance(dc, dict):
        cfg.declicker.enabled = bool(dc.get("enabled", True))
        if "sensitivity" in dc:
            cfg.declicker.sensitivity = float(dc["sensitivity"])
        if "click_width_samples" in dc:
            cfg.declicker.click_width_samples = int(dc["click_width_samples"])
        if "mode" in dc:
            cfg.declicker.mode = str(dc["mode"])

    # De-Crackler
    dcr = dsp_params.get("decrackler")
    if dcr and isinstance(dcr, dict):
        cfg.decrackler.enabled = bool(dcr.get("enabled", True))
        if "sensitivity" in dcr:
            cfg.decrackler.sensitivity = float(dcr["sensitivity"])
        if "crackle_reduction" in dcr:
            cfg.decrackler.crackle_reduction = float(dcr["crackle_reduction"])
        elif "reduction" in dcr:
            cfg.decrackler.crackle_reduction = float(dcr["reduction"])
        if "frequency_band" in dcr:
            cfg.decrackler.frequency_band = str(dcr["frequency_band"])
        if "density" in dcr:
            cfg.decrackler.density = float(dcr["density"])

    # De-Verb
    dv = dsp_params.get("deverb")
    if dv and isinstance(dv, dict):
        cfg.deverb.enabled = bool(dv.get("enabled", True))
        if "reduction_amount" in dv:
            cfg.deverb.reduction_amount = float(dv["reduction_amount"])
        if "room_size" in dv:
            cfg.deverb.room_size = str(dv["room_size"])
        if "preserve_early_reflections" in dv:
            cfg.deverb.preserve_early_reflections = float(dv["preserve_early_reflections"])
        if "spectral_floor_db" in dv:
            cfg.deverb.spectral_floor_db = float(dv["spectral_floor_db"])

    # De-Esser
    de = dsp_params.get("deesser")
    if de and isinstance(de, dict):
        cfg.deesser.enabled = bool(de.get("enabled", True))
        if "threshold_db" in de:
            cfg.deesser.threshold_db = float(de["threshold_db"])
        if "reduction_amount" in de:
            cfg.deesser.reduction_amount = float(de["reduction_amount"])
        if "frequency_focus_hz" in de:
            cfg.deesser.frequency_focus_hz = float(de["frequency_focus_hz"])
        if "bandwidth_q" in de:
            cfg.deesser.bandwidth_q = float(de["bandwidth_q"])
        if "mode" in de:
            cfg.deesser.mode = str(de["mode"])

    return cfg


def isolate_file(
    file_path: str,
    model_choice: str = "offline",
    offline_passes: int = 1,
    precision: str = "auto",
    dsp_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Performs neural voice isolation followed by the DENOX DSP mastering suite."""
    models = get_engine_models()
    path = resolve_source_audio_path(file_path)

    # Load audio supporting WAV, MP3, FLAC, M4A
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1, dtype=np.float32)
    except Exception:
        wav_t, rate = torchaudio.load(str(path))
        mono_t = audio_lib.to_mono(wav_t)
        mono = mono_t.cpu().numpy().astype(np.float32)

    # Resample to 48kHz if needed
    if rate != SAMPLE_RATE:
        mono_resampled = vi.resample(torch.from_numpy(mono), rate, SAMPLE_RATE).numpy()
    else:
        mono_resampled = mono

    duration_sec = len(mono_resampled) / SAMPLE_RATE
    start_time = time.perf_counter()

    if model_choice == "realtime":
        with models.realtime_lock:
            raw_voice = models.realtime.isolate(mono_resampled, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
    else:
        with models.offline_lock:
            old_passes = models.offline.offline_passes
            try:
                models.offline.offline_passes = max(1, int(offline_passes))
                raw_voice = models.offline.isolate(mono_resampled, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
            finally:
                models.offline.offline_passes = old_passes

    if isinstance(raw_voice, torch.Tensor):
        raw_voice = raw_voice.cpu().numpy()

    elapsed = time.perf_counter() - start_time
    rtf = duration_sec / max(elapsed, 1e-6)

    # Build and execute DSP Mastering Chain
    dsp_cfg = build_dsp_config(dsp_params)
    processor = dsp_module.DenoxMasteringProcessor(SAMPLE_RATE)
    mastered_voice, background, dsp_stats = processor.process_chain(
        mono_resampled, raw_voice, dsp_cfg
    )

    # Save stems to temporary studio cache
    stem_id = f"stem_{int(time.time() * 1000)}"
    stem_dir = TMP_DIR / stem_id
    stem_dir.mkdir(parents=True, exist_ok=True)

    orig_path = stem_dir / "original.wav"
    raw_vocal_path = stem_dir / "raw_neural_voice.wav"
    vocal_path = stem_dir / "vocal.wav"
    bg_path = stem_dir / "background.wav"

    sf.write(str(orig_path), pcm16(mono_resampled), SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(raw_vocal_path), pcm16(raw_voice), SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(vocal_path), pcm16(mastered_voice), SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(bg_path), pcm16(background), SAMPLE_RATE, subtype="PCM_16")

    # Generate waveform preview data
    peaks_orig = compute_waveform_peaks(mono_resampled, 400)
    peaks_vocal = compute_waveform_peaks(mastered_voice, 400)
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
        "dsp_stats": dsp_stats,
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


def remaster_stem(stem_id: str, dsp_params: Dict[str, Any]) -> Dict[str, Any]:
    """Re-applies DSP chain to cached neural isolation output with sub-second latency."""
    stem_dir = TMP_DIR / stem_id
    orig_path = stem_dir / "original.wav"
    raw_vocal_path = stem_dir / "raw_neural_voice.wav"
    vocal_path = stem_dir / "vocal.wav"
    bg_path = stem_dir / "background.wav"

    if not orig_path.exists() or not raw_vocal_path.exists():
        raise FileNotFoundError(f"Stem session {stem_id} not found in cache")

    orig, _ = sf.read(str(orig_path), dtype="float32")
    raw_vocal, _ = sf.read(str(raw_vocal_path), dtype="float32")

    dsp_cfg = build_dsp_config(dsp_params)
    processor = dsp_module.DenoxMasteringProcessor(SAMPLE_RATE)
    mastered_voice, background, dsp_stats = processor.process_chain(orig, raw_vocal, dsp_cfg)

    sf.write(str(vocal_path), pcm16(mastered_voice), SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(bg_path), pcm16(background), SAMPLE_RATE, subtype="PCM_16")

    peaks_vocal = compute_waveform_peaks(mastered_voice, 400)
    peaks_bg = compute_waveform_peaks(background, 400)

    return {
        "stem_id": stem_id,
        "dsp_stats": dsp_stats,
        "urls": {
            "original": f"/api/audio/{stem_id}/original.wav?t={time.time()}",
            "vocal": f"/api/audio/{stem_id}/vocal.wav?t={time.time()}",
            "background": f"/api/audio/{stem_id}/background.wav?t={time.time()}",
        },
        "waveforms": {
            "vocal": peaks_vocal,
            "background": peaks_bg,
        }
    }


def resolve_source_audio_path(file_path: Union[str, pathlib.Path], stem_id: Optional[str] = None) -> pathlib.Path:
    """Robustly locates an audio file across absolute, relative, uploads, demo examples, and stem directories."""
    p = pathlib.Path(file_path).expanduser()
    if p.is_file():
        return p.resolve()
    if (pathlib.Path.cwd() / p).is_file():
        return (pathlib.Path.cwd() / p).resolve()
    if (TMP_DIR / "uploads" / p.name).is_file():
        return (TMP_DIR / "uploads" / p.name).resolve()
    if (pathlib.Path("demo/examples") / p.name).is_file():
        return (pathlib.Path("demo/examples") / p.name).resolve()
    if stem_id and (TMP_DIR / stem_id / "original.wav").is_file():
        return (TMP_DIR / stem_id / "original.wav").resolve()
    if stem_id and (TMP_DIR / stem_id / p.name).is_file():
        return (TMP_DIR / stem_id / p.name).resolve()
    matches = list(TMP_DIR.glob(f"**/{p.name}"))
    if matches and matches[0].is_file():
        return matches[0].resolve()
    raise FileNotFoundError(f"Source audio file not found: {file_path}")


def generate_module_preview(
    file_path: str,
    module_name: str,
    params: Dict[str, Any],
    stem_id: Optional[str] = None,
    source_mode: str = "original",
) -> Dict[str, Any]:
    """Generates audition audio for an individual module with current settings."""
    path = resolve_source_audio_path(file_path, stem_id=stem_id)

    # Check if stem_id provided and we want vocal source
    raw_vocal = None
    if stem_id:
        stem_dir = TMP_DIR / stem_id
        raw_vocal_path = stem_dir / "raw_neural_voice.wav"
        if not raw_vocal_path.exists():
            raw_vocal_path = stem_dir / "vocal.wav"
        if raw_vocal_path.exists():
            try:
                raw_vocal, _ = sf.read(str(raw_vocal_path), dtype="float32")
                if raw_vocal.ndim > 1:
                    raw_vocal = np.mean(raw_vocal, axis=1)
            except Exception:
                raw_vocal = None

    # Load and resample audio supporting multiple formats (WAV, MP3, FLAC, M4A)
    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception:
        wav_t, sr = torchaudio.load(str(path))
        audio = wav_t.numpy()
        if audio.ndim == 2:
            audio = audio.T

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        tensor_audio = torch.from_numpy(audio.astype(np.float32))
        audio = audio_lib.resample(tensor_audio, sr, SAMPLE_RATE).cpu().numpy().astype(np.float32)

    duration_sec = round(len(audio) / SAMPLE_RATE, 2)

    # Check hash for caching
    param_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(f"{path.name}_{module_name}_{source_mode}_{param_str}_{stem_id or ''}".encode()).hexdigest()[:16]
    vocal_cache_key = hashlib.md5(f"{path.name}_raw_vocal_{stem_id or ''}".encode()).hexdigest()[:16]

    out_filename = f"prev_{module_name}_{cache_key}.wav"
    orig_filename = f"prev_orig_{cache_key}.wav"
    vocal_filename = f"prev_vocal_{vocal_cache_key}.wav"
    out_path = PREVIEWS_DIR / out_filename
    orig_path = PREVIEWS_DIR / orig_filename
    vocal_path = PREVIEWS_DIR / vocal_filename

    # Save orig preview if not exists
    if not orig_path.exists():
        sf.write(str(orig_path), pcm16(audio), SAMPLE_RATE, subtype="PCM_16")

    vocal_url = f"/api/preview/audio/{vocal_filename}?t={time.time()}" if vocal_path.exists() else None

    if out_path.exists():
        return {
            "status": "success",
            "preview_url": f"/api/preview/audio/{out_filename}?t={time.time()}",
            "original_url": f"/api/preview/audio/{orig_filename}?t={time.time()}",
            "vocal_url": vocal_url,
            "duration": duration_sec,
            "module": module_name
        }

    # Build specific module config and run it
    cfg = build_dsp_config({module_name: params})

    if module_name == "denoiser":
        denoiser = dsp_module.DeNoiser(SAMPLE_RATE)
        if raw_vocal is None:
            if vocal_path.exists():
                raw_vocal, _ = sf.read(str(vocal_path), dtype="float32")
                if raw_vocal.ndim > 1:
                    raw_vocal = np.mean(raw_vocal, axis=1)
            else:
                models = get_engine_models()
                with models.offline_lock:
                    raw_vocal = models.offline.isolate(audio, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
                    if isinstance(raw_vocal, torch.Tensor):
                        raw_vocal = raw_vocal.cpu().numpy()
                    if raw_vocal.ndim > 1:
                        raw_vocal = np.mean(raw_vocal, axis=1)
                sf.write(str(vocal_path), pcm16(raw_vocal), SAMPLE_RATE, subtype="PCM_16")
        elif not vocal_path.exists():
            sf.write(str(vocal_path), pcm16(raw_vocal), SAMPLE_RATE, subtype="PCM_16")
        
        vocal_url = f"/api/preview/audio/{vocal_filename}?t={time.time()}"
        min_l = min(len(audio), len(raw_vocal))
        processed, _ = denoiser.process(audio[:min_l], raw_vocal[:min_l], cfg.denoiser)

    elif module_name == "declicker":
        declicker = dsp_module.DeClicker(SAMPLE_RATE)
        src = raw_vocal if (source_mode == "vocal" and raw_vocal is not None) else audio
        processed, _ = declicker.process(src, cfg.declicker)

    elif module_name == "decrackler":
        decrackler = dsp_module.DeCrackler(SAMPLE_RATE)
        src = raw_vocal if (source_mode == "vocal" and raw_vocal is not None) else audio
        processed, _ = decrackler.process(src, cfg.decrackler)

    elif module_name == "deverb":
        deverb = dsp_module.DeVerb(SAMPLE_RATE)
        src = raw_vocal if (source_mode == "vocal" and raw_vocal is not None) else audio
        processed = deverb.process(src, cfg.deverb)

    elif module_name == "deesser":
        deesser = dsp_module.DeEsser(SAMPLE_RATE)
        src = raw_vocal if (source_mode == "vocal" and raw_vocal is not None) else audio
        processed = deesser.process(src, cfg.deesser)
    else:
        processed = audio

    sf.write(str(out_path), pcm16(processed), SAMPLE_RATE, subtype="PCM_16")

    return {
        "status": "success",
        "preview_url": f"/api/preview/audio/{out_filename}?t={time.time()}",
        "original_url": f"/api/preview/audio/{orig_filename}?t={time.time()}",
        "vocal_url": vocal_url,
        "duration": duration_sec,
        "module": module_name
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
