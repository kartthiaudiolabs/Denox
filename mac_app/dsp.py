# Copyright 2026 Inworld AI & Antigravity
# DENOX Advanced Audio DSP Suite: De-Clicker, De-Crackler, De-Verb, De-Esser & Thresholding

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


# -----------------------------------------------------------------------------
# 1. THRESHOLD & NEURAL BLENDING
# -----------------------------------------------------------------------------

def apply_isolation_threshold(
    original: np.ndarray,
    isolated: np.ndarray,
    threshold: float = 1.0,
    mode: str = "spectral_gate"
) -> np.ndarray:
    """Applies a 0-100% threshold to control how aggressively noise is removed.

    Args:
        original: Raw input audio (float32 mono, 48kHz).
        isolated: 100% neural isolated vocal audio (float32 mono, 48kHz).
        threshold: 0.0 (0% cleaning = raw audio) to 1.0 (100% cleaning = pure isolated vocal).
        mode: "blend" for linear/power blend, "spectral_gate" for natural room-tone preservation.

    Returns:
        Processed audio with user-selected noise reduction threshold.
    """
    threshold = float(np.clip(threshold, 0.0, 1.0))
    if threshold >= 0.999:
        return isolated
    if threshold <= 0.001:
        return original

    min_len = min(len(original), len(isolated))
    orig = original[:min_len]
    iso = isolated[:min_len]

    if mode == "blend":
        # Perceptual equal-power crossfade between original and isolated
        # threshold 0.0 -> orig, threshold 1.0 -> iso
        angle = threshold * (math.pi / 2.0)
        out = math.cos(angle) * orig + math.sin(angle) * iso
        return np.ascontiguousarray(out, dtype=np.float32)

    # Spectral gating / soft mask mode:
    # Retains a controlled, natural proportion of the noise floor to prevent unnatural "dead room" silence
    # noise residual = original - isolated
    noise_residual = orig - iso
    
    # Attenuation curve: threshold=1.0 -> -60dB (pure vocal), threshold=0.5 -> -12dB, threshold=0.0 -> 0dB
    # dB_attenuation = -60.0 * (threshold ** 1.5)
    attenuation_linear = (1.0 - threshold) ** 1.5
    blended = iso + noise_residual * attenuation_linear
    return np.ascontiguousarray(blended, dtype=np.float32)


# -----------------------------------------------------------------------------
# 2. DE-CLICKER & DE-CRACKLER MODULE
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeClickerConfig:
    enabled: bool = True
    sensitivity: float = 0.70       # 0.0 to 1.0
    click_width_samples: int = 16    # 4 to 64 samples (~0.1ms - 1.3ms at 48kHz)
    decrackle_amount: float = 0.50   # 0.0 to 1.0
    mode: str = "mouth_clicks"      # "mouth_clicks", "digital_artifacts", "vinyl_crackle", "subtle"


DECLICKER_PRESETS: Dict[str, Dict[str, Any]] = {
    "mouth_clicks": {
        "label": "Mouth & Saliva Clicks",
        "sensitivity": 0.75,
        "click_width_samples": 18,
        "decrackle_amount": 0.40,
        "mode": "mouth_clicks"
    },
    "digital_artifacts": {
        "label": "Digital Dropouts & Buffer Clicks",
        "sensitivity": 0.88,
        "click_width_samples": 32,
        "decrackle_amount": 0.30,
        "mode": "digital_artifacts"
    },
    "vinyl_crackle": {
        "label": "Vinyl & Dense Crackle",
        "sensitivity": 0.65,
        "click_width_samples": 12,
        "decrackle_amount": 0.85,
        "mode": "vinyl_crackle"
    },
    "subtle_transparent": {
        "label": "Subtle Transparent Polish",
        "sensitivity": 0.45,
        "click_width_samples": 10,
        "decrackle_amount": 0.25,
        "mode": "subtle"
    }
}


class DeClickerDeCrackler:
    """Detects and repairs impulsive clicks and dense audio crackle."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeClickerConfig) -> Tuple[np.ndarray, int]:
        """Removes clicks and crackle using median prediction and spline interpolation.

        Returns:
            Tuple of (cleaned_audio, num_clicks_repaired)
        """
        if not config.enabled or len(audio) < 128:
            return audio.copy(), 0

        y = audio.astype(np.float64, copy=True)
        n_samples = len(y)

        # 1. High-frequency differential filter (removes smooth speech formants, isolates impulse spikes)
        # Using 2nd difference d[n] = x[n] - 2x[n-1] + x[n-2]
        diff = np.zeros_like(y)
        diff[2:] = y[2:] - 2.0 * y[1:-1] + y[:-2]

        # 2. Local rolling RMS envelope
        window_size = 512
        kernel = np.ones(window_size) / window_size
        local_energy = signal.convolve(diff ** 2, kernel, mode='same')
        local_rms = np.sqrt(np.maximum(local_energy, 1e-9))

        # 3. Dynamic Thresholding based on sensitivity
        # Sensitivity 0.0 -> threshold_factor 8.0 (detects only massive clicks)
        # Sensitivity 1.0 -> threshold_factor 2.2 (detects tiny mouth clicks)
        threshold_factor = 7.5 - (config.sensitivity * 5.0)
        click_mask = np.abs(diff) > (threshold_factor * local_rms)

        # 4. Repair clicks via Cubic Spline Interpolation
        width = max(4, int(config.click_width_samples))
        click_indices = np.where(click_mask)[0]
        repaired_count = 0

        if len(click_indices) > 0:
            i = 0
            while i < len(click_indices):
                idx = click_indices[i]
                start = max(0, idx - width // 2)
                end = min(n_samples, idx + width // 2)

                # Collect consecutive or close clicks into one window
                while i + 1 < len(click_indices) and click_indices[i + 1] <= end + 4:
                    i += 1
                    end = min(n_samples, click_indices[i] + width // 2)

                gap_len = end - start
                if 2 <= gap_len <= 128:
                    # Anchor points before and after the click
                    n_anchors = 6
                    left_anchors = np.arange(max(0, start - n_anchors), start)
                    right_anchors = np.arange(end, min(n_samples, end + n_anchors))

                    if len(left_anchors) >= 2 and len(right_anchors) >= 2:
                        x_anchors = np.concatenate([left_anchors, right_anchors])
                        y_anchors = y[x_anchors]
                        try:
                            cs = CubicSpline(x_anchors, y_anchors, bc_type='natural')
                            y[start:end] = cs(np.arange(start, end))
                            repaired_count += 1
                        except Exception:
                            pass
                i += 1

        # 5. De-Crackling (Adaptive High-Frequency Soft Limiter)
        if config.decrackle_amount > 0.05:
            # High-pass filter above 5kHz to isolate crackle band
            sos = signal.butter(4, 5000, btype='highpass', fs=self.sample_rate, output='sos')
            hf = signal.sosfiltfilt(sos, y)
            hf_rms = np.sqrt(np.mean(hf ** 2) + 1e-9)
            crackle_limit = hf_rms * (4.0 - 2.5 * config.decrackle_amount)
            excess = np.maximum(0, np.abs(hf) - crackle_limit)
            suppression = np.sign(hf) * excess * config.decrackle_amount
            y -= suppression

        return np.ascontiguousarray(y, dtype=np.float32), repaired_count


# -----------------------------------------------------------------------------
# 3. DE-VERB (DEREVERBERATION) MODULE
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeVerbConfig:
    enabled: bool = True
    reduction_amount: float = 0.65       # 0.0 to 1.0 (0% to 100%)
    room_size: str = "medium"           # "small", "medium", "large", "hall"
    preserve_early_reflections: float = 0.50 # 0.0 to 1.0
    spectral_floor_db: float = -18.0    # dB floor to avoid watery artifacts


DEVERB_PRESETS: Dict[str, Dict[str, Any]] = {
    "tight_booth": {
        "label": "Treated Vocal Booth",
        "reduction_amount": 0.85,
        "room_size": "small",
        "preserve_early_reflections": 0.40,
        "spectral_floor_db": -22.0
    },
    "bedroom_studio": {
        "label": "Dry Bedroom Studio",
        "reduction_amount": 0.65,
        "room_size": "medium",
        "preserve_early_reflections": 0.55,
        "spectral_floor_db": -18.0
    },
    "hall_echo_kill": {
        "label": "Cavern & Echo Suppressor",
        "reduction_amount": 0.95,
        "room_size": "large",
        "preserve_early_reflections": 0.20,
        "spectral_floor_db": -26.0
    },
    "subtle_clarity": {
        "label": "Subtle Room Tame",
        "reduction_amount": 0.40,
        "room_size": "small",
        "preserve_early_reflections": 0.70,
        "spectral_floor_db": -14.0
    }
}


class DeVerb:
    """Removes room reverberation and late room reflections using spectral decay modeling."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeVerbConfig) -> np.ndarray:
        """Suppresses reverberant tail energy via multi-band late reverberation subtraction."""
        if not config.enabled or len(audio) < 1024 or config.reduction_amount <= 0.01:
            return audio.copy()

        # STFT parameters
        n_fft = 2048
        hop_length = 512
        win = np.hanning(n_fft).astype(np.float32)

        # Compute STFT
        _, _, stft = signal.stft(
            audio,
            fs=self.sample_rate,
            window=win,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary='zeros'
        )

        magnitude = np.abs(stft)
        phase = np.angle(stft)
        n_freqs, n_frames = magnitude.shape

        # Room size sets the decay time constant gamma and delay D
        room_decay = {
            "small": 0.82,
            "medium": 0.89,
            "large": 0.94,
            "hall": 0.97
        }.get(config.room_size, 0.89)

        # Delay D in frames (preserves direct sound and initial early reflections)
        # 1 frame = 512 / 48000 = ~10.6ms
        delay_frames = max(2, int(2 + config.preserve_early_reflections * 4))

        # Estimate late reverberation energy per frequency bin
        reverb_energy = np.zeros_like(magnitude)
        power = magnitude ** 2

        for t in range(delay_frames, n_frames):
            reverb_energy[:, t] = (
                room_decay * reverb_energy[:, t - 1] +
                (1.0 - room_decay) * power[:, t - delay_frames]
            )

        # Spectral subtraction factor beta
        beta = config.reduction_amount * 1.6
        floor_linear = 10.0 ** (config.spectral_floor_db / 20.0)

        # Clean magnitude estimate
        clean_power = np.maximum(power - beta * reverb_energy, (floor_linear * magnitude) ** 2)
        clean_magnitude = np.sqrt(clean_power)

        # Soft gain smoothing across time frames to eliminate musical noise
        gain = clean_magnitude / (magnitude + 1e-9)
        gain_smooth = signal.medfilt2d(gain, kernel_size=(3, 3))
        final_magnitude = magnitude * gain_smooth

        # Reconstruct waveform via iSTFT
        _, dereverb_audio = signal.istft(
            final_magnitude * np.exp(1j * phase),
            fs=self.sample_rate,
            window=win,
            nperseg=n_fft,
            noverlap=n_fft - hop_length
        )

        # Align length
        dereverb_audio = dereverb_audio[:len(audio)]
        return np.ascontiguousarray(dereverb_audio, dtype=np.float32)


# -----------------------------------------------------------------------------
# 4. DE-ESSER (SIBILANCE REDUCER) MODULE
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeEsserConfig:
    enabled: bool = True
    threshold_db: float = -20.0       # -40.0 dB to 0.0 dB
    reduction_amount: float = 0.70    # 0.0 to 1.0 (0% to 100%)
    frequency_focus_hz: float = 6500.0# 4000 Hz to 10000 Hz
    bandwidth_q: float = 1.4          # 0.5 to 3.0
    mode: str = "split_band"          # "split_band" (attenuates only sibilance) or "wideband"


DEESSER_PRESETS: Dict[str, Dict[str, Any]] = {
    "smooth_broadcast": {
        "label": "Broadcast Warmth",
        "threshold_db": -22.0,
        "reduction_amount": 0.65,
        "frequency_focus_hz": 6500.0,
        "bandwidth_q": 1.4,
        "mode": "split_band"
    },
    "harsh_s_tamer": {
        "label": "Harsh 'S' Piercing Tamer",
        "threshold_db": -26.0,
        "reduction_amount": 0.85,
        "frequency_focus_hz": 7200.0,
        "bandwidth_q": 1.8,
        "mode": "split_band"
    },
    "male_dialogue": {
        "label": "Male Dialogue (Lower S)",
        "threshold_db": -18.0,
        "reduction_amount": 0.60,
        "frequency_focus_hz": 5200.0,
        "bandwidth_q": 1.2,
        "mode": "split_band"
    },
    "female_vocal_airy": {
        "label": "Female Vocal (High S / T)",
        "threshold_db": -24.0,
        "reduction_amount": 0.75,
        "frequency_focus_hz": 8000.0,
        "bandwidth_q": 1.5,
        "mode": "split_band"
    }
}


class DeEsser:
    """Dynamic frequency-selective compressor targeting harsh sibilance ('s', 'sh', 'ch', 't')."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeEsserConfig) -> np.ndarray:
        """Detects sibilant peaks and applies dynamic split-band gain reduction."""
        if not config.enabled or len(audio) < 128 or config.reduction_amount <= 0.01:
            return audio.copy()

        nyquist = self.sample_rate / 2.0
        fc = np.clip(config.frequency_focus_hz, 3000.0, nyquist - 1000.0)
        bw = fc / max(0.5, config.bandwidth_q)
        f_low = max(100.0, fc - bw / 2.0)
        f_high = min(nyquist - 100.0, fc + bw / 2.0)

        # 1. Bandpass filter to isolate the sibilance detection / control band
        sos_band = signal.butter(4, [f_low, f_high], btype='bandpass', fs=self.sample_rate, output='sos')
        sibilant_band = signal.sosfiltfilt(sos_band, audio)

        # 2. Sidechain envelope detector (fast 1ms attack, 35ms release)
        attack_coef = math.exp(-1.0 / (0.001 * self.sample_rate))
        release_coef = math.exp(-1.0 / (0.035 * self.sample_rate))

        rectified = np.abs(sibilant_band)
        envelope = np.zeros_like(rectified)
        curr_env = 0.0

        for i in range(len(rectified)):
            val = rectified[i]
            if val > curr_env:
                curr_env = attack_coef * curr_env + (1.0 - attack_coef) * val
            else:
                curr_env = release_coef * curr_env + (1.0 - release_coef) * val
            envelope[i] = curr_env

        # 3. Dynamic gain reduction curve
        threshold_linear = 10.0 ** (config.threshold_db / 20.0)
        max_reduction_linear = 1.0 - (config.reduction_amount * 0.90)  # max ~20dB reduction

        gain = np.ones_like(envelope)
        over_threshold = envelope > threshold_linear
        if np.any(over_threshold):
            excess_ratio = (envelope[over_threshold] - threshold_linear) / (envelope[over_threshold] + 1e-6)
            gain[over_threshold] = np.maximum(
                max_reduction_linear,
                1.0 - config.reduction_amount * excess_ratio
            )

        # 4. Apply compression
        if config.mode == "split_band":
            # Split-band mode: Subtract compressed sibilant band and re-add attenuated version
            # (zero distortion to low-end vowels / chest resonances!)
            residual_speech = audio - sibilant_band
            output = residual_speech + (sibilant_band * gain)
        else:
            # Wideband mode
            output = audio * gain

        return np.ascontiguousarray(output, dtype=np.float32)


# -----------------------------------------------------------------------------
# 5. DENOX AUDIO MASTERING CHAIN ORCHESTRATOR
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DenoxMasteringChainConfig:
    # Threshold (0.0 - 1.0)
    isolation_threshold: float = 1.0
    threshold_mode: str = "spectral_gate"

    # Modules
    declicker: DeClickerConfig = dataclasses.field(default_factory=DeClickerConfig)
    deverb: DeVerbConfig = dataclasses.field(default_factory=DeVerbConfig)
    deesser: DeEsserConfig = dataclasses.field(default_factory=DeEsserConfig)


# Global Suite Presets
GLOBAL_CHAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "ultra_clean_podcast": {
        "name": "Ultra-Clean Podcast (Max Clarity)",
        "isolation_threshold": 0.92,
        "declicker": { "enabled": True, "preset": "mouth_clicks" },
        "deverb": { "enabled": True, "preset": "tight_booth" },
        "deesser": { "enabled": True, "preset": "smooth_broadcast" }
    },
    "studio_vocal_booth": {
        "name": "Studio Vocal Booth Master",
        "isolation_threshold": 1.00,
        "declicker": { "enabled": True, "preset": "subtle_transparent" },
        "deverb": { "enabled": True, "preset": "bedroom_studio" },
        "deesser": { "enabled": True, "preset": "harsh_s_tamer" }
    },
    "natural_warmth": {
        "name": "Natural Warmth (Retain Room Air)",
        "isolation_threshold": 0.70,
        "declicker": { "enabled": True, "preset": "mouth_clicks" },
        "deverb": { "enabled": False, "preset": "subtle_clarity" },
        "deesser": { "enabled": True, "preset": "smooth_broadcast" }
    },
    "heavy_room_killer": {
        "name": "Echo & Cavern Eliminator",
        "isolation_threshold": 1.00,
        "declicker": { "enabled": True, "preset": "digital_artifacts" },
        "deverb": { "enabled": True, "preset": "hall_echo_kill" },
        "deesser": { "enabled": True, "preset": "harsh_s_tamer" }
    },
    "vintage_restoration": {
        "name": "Vintage Vinyl & Crackle Restoration",
        "isolation_threshold": 0.80,
        "declicker": { "enabled": True, "preset": "vinyl_crackle" },
        "deverb": { "enabled": True, "preset": "subtle_clarity" },
        "deesser": { "enabled": False, "preset": "smooth_broadcast" }
    }
}


class DenoxMasteringProcessor:
    """Runs the complete mastering chain: Isolation Thresholding -> De-Clicker -> De-Verb -> De-Esser."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.declicker = DeClickerDeCrackler(sample_rate)
        self.deverb = DeVerb(sample_rate)
        self.deesser = DeEsser(sample_rate)

    def process_chain(
        self,
        original_audio: np.ndarray,
        isolated_vocal: np.ndarray,
        config: DenoxMasteringChainConfig
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Processes the neural vocal through thresholding and DSP modules.

        Returns:
            Tuple of (mastered_vocal, background_residual, stats)
        """
        # Step 1: Isolation thresholding (0-100%)
        vocal = apply_isolation_threshold(
            original_audio,
            isolated_vocal,
            threshold=config.isolation_threshold,
            mode=config.threshold_mode
        )

        stats: Dict[str, Any] = {
            "threshold_applied": config.isolation_threshold,
            "clicks_repaired": 0,
            "deverb_applied": config.deverb.enabled,
            "deesser_applied": config.deesser.enabled
        }

        # Step 2: De-Clicker & De-Crackler
        if config.declicker.enabled:
            vocal, clicks = self.declicker.process(vocal, config.declicker)
            stats["clicks_repaired"] = clicks

        # Step 3: De-Verb (Dereverberation)
        if config.deverb.enabled:
            vocal = self.deverb.process(vocal, config.deverb)

        # Step 4: De-Esser (Sibilance Compression)
        if config.deesser.enabled:
            vocal = self.deesser.process(vocal, config.deesser)

        # Derived background residual = original - mastered vocal
        min_len = min(len(original_audio), len(vocal))
        background = original_audio[:min_len] - vocal[:min_len]

        return vocal[:min_len], background, stats
