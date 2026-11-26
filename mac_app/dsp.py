# Copyright 2026 Inworld AI & Antigravity
# DENOX Advanced Audio DSP Suite: De-Clicker, De-Crackler, De-Verb, De-Esser & Thresholding

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


# -----------------------------------------------------------------------------
# 1. DE-NOISER MODULE (NEURAL BACKGROUND NOISE SEPARATION & ROOM TONE)
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeNoiserConfig:
    enabled: bool = True
    preset: str = "pure_isolated_vocal"
    threshold: float = 1.0            # 0.0 (0% clean, raw audio) to 1.0 (100% clean, pure isolated vocal)
    mode: str = "spectral_gate"       # "spectral_gate", "blend", "air_preserve"
    reduction_db: float = -60.0       # max attenuation dB (-12dB to -60dB)
    ambient_bleed: float = 0.0        # natural room tone / background bleed (0.0 to 0.3)


DENOISER_PRESETS: Dict[str, Dict[str, Any]] = {
    # --- Category: Podcast & Studio Dialogue ---
    "podcast_hvac_tame": {
        "label": "Studio AC & HVAC Hum Reduction",
        "category": "Podcast & Studio Dialogue",
        "threshold": 0.85,
        "mode": "spectral_gate",
        "ambient_bleed": 0.08
    },
    "pc_fan_laptop_whine": {
        "label": "Computer Fan & Laptop Coil Whine",
        "category": "Podcast & Studio Dialogue",
        "threshold": 0.90,
        "mode": "spectral_gate",
        "ambient_bleed": 0.05
    },
    "whisper_room_air": {
        "label": "Whisper Voice & Delicate Room Air",
        "category": "Podcast & Studio Dialogue",
        "threshold": 0.65,
        "mode": "air_preserve",
        "ambient_bleed": 0.20
    },
    "pure_isolated_vocal": {
        "label": "Zero-Noise Vocal Booth (100% Isolation)",
        "category": "Podcast & Studio Dialogue",
        "threshold": 1.00,
        "mode": "spectral_gate",
        "ambient_bleed": 0.00
    },

    # --- Category: Broadcast, Field & Mobile ---
    "city_traffic_rumble": {
        "label": "Urban Street Traffic & Low Rumble",
        "category": "Broadcast, Field & Mobile",
        "threshold": 0.88,
        "mode": "spectral_gate",
        "ambient_bleed": 0.06
    },
    "outdoor_wind_buffet": {
        "label": "Outdoor Wind & Air Buffet Suppression",
        "category": "Broadcast, Field & Mobile",
        "threshold": 0.92,
        "mode": "spectral_gate",
        "ambient_bleed": 0.04
    },
    "cafeteria_crowd_babble": {
        "label": "Restaurant & Café Crowd Chatter",
        "category": "Broadcast, Field & Mobile",
        "threshold": 0.94,
        "mode": "spectral_gate",
        "ambient_bleed": 0.03
    },
    "smartphone_voice_memo": {
        "label": "Mobile Phone Voice Memo Clean",
        "category": "Broadcast, Field & Mobile",
        "threshold": 0.82,
        "mode": "blend",
        "ambient_bleed": 0.10
    },

    # --- Category: Cinema, Film & Location Sound ---
    "on_location_generator": {
        "label": "Diesel Generator & Inverter Whine",
        "category": "Cinema, Film & Location Sound",
        "threshold": 0.95,
        "mode": "spectral_gate",
        "ambient_bleed": 0.02
    },
    "camera_fan_drone": {
        "label": "Cinema Camera Cooling Fan Whine",
        "category": "Cinema, Film & Location Sound",
        "threshold": 0.89,
        "mode": "spectral_gate",
        "ambient_bleed": 0.05
    },
    "aircraft_overhead_pass": {
        "label": "Jet Aircraft & Overhead Drone Tame",
        "category": "Cinema, Film & Location Sound",
        "threshold": 0.87,
        "mode": "spectral_gate",
        "ambient_bleed": 0.07
    },
    "rain_weather_isolation": {
        "label": "Heavy Rain & Window Spatter Isolation",
        "category": "Cinema, Film & Location Sound",
        "threshold": 0.93,
        "mode": "spectral_gate",
        "ambient_bleed": 0.03
    },

    # --- Category: Live Stage & Music Stems ---
    "stage_drum_spill_bleed": {
        "label": "Live Stage Drum & Cymbal Bleed Kill",
        "category": "Live Stage & Music Stems",
        "threshold": 0.96,
        "mode": "spectral_gate",
        "ambient_bleed": 0.02
    },
    "pa_monitor_feedback_air": {
        "label": "PA Floor Monitor Acoustic Spill",
        "category": "Live Stage & Music Stems",
        "threshold": 0.90,
        "mode": "spectral_gate",
        "ambient_bleed": 0.05
    },
    "acoustic_room_subtle_clean": {
        "label": "Subtle Acoustic Clean (Preserve Live Feel)",
        "category": "Live Stage & Music Stems",
        "threshold": 0.55,
        "mode": "blend",
        "ambient_bleed": 0.25
    },
    "backing_vocal_bleed": {
        "label": "Guitar & Bass Bleed Suppression",
        "category": "Live Stage & Music Stems",
        "threshold": 0.86,
        "mode": "spectral_gate",
        "ambient_bleed": 0.08
    },

    # --- Category: Archival & Industrial Restoration ---
    "mains_hum_50_60hz": {
        "label": "50Hz/60Hz Electrical Mains Hum & Buzz",
        "category": "Archival & Industrial Restoration",
        "threshold": 0.91,
        "mode": "spectral_gate",
        "ambient_bleed": 0.04
    },
    "analog_tape_hiss_wash": {
        "label": "Cassette & Reel-to-Reel Tape Hiss",
        "category": "Archival & Industrial Restoration",
        "threshold": 0.84,
        "mode": "spectral_gate",
        "ambient_bleed": 0.09
    },
    "industrial_plant_drone": {
        "label": "Factory Machinery & Heavy Motor Drone",
        "category": "Archival & Industrial Restoration",
        "threshold": 0.97,
        "mode": "spectral_gate",
        "ambient_bleed": 0.01
    },
    "vintage_radio_static_floor": {
        "label": "Vintage AM/Shortwave Static Floor",
        "category": "Archival & Industrial Restoration",
        "threshold": 0.80,
        "mode": "blend",
        "ambient_bleed": 0.12
    }
}


def true_peak_mastering_limiter(audio: np.ndarray, ceiling: float = 0.96) -> np.ndarray:
    """Transparent soft-knee brickwall limiter to ensure 0 digital clipping and 0 rail-to-rail distortion."""
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling:
        return audio
    threshold = ceiling * 0.85
    excess = np.maximum(0.0, np.abs(audio) - threshold)
    scale = max(ceiling - threshold, 1e-6)
    compressed_excess = scale * np.tanh(excess / scale)
    limited = np.sign(audio) * (np.minimum(np.abs(audio), threshold) + compressed_excess)
    return np.clip(limited, -ceiling, ceiling)


class DeNoiser:
    """Applies multi-mode neural background noise separation, thresholding, and room tone control."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(
        self,
        original: np.ndarray,
        isolated: np.ndarray,
        config: DeNoiserConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Separates and attenuates background noise according to threshold and mode."""
        if not config.enabled:
            min_len = min(len(original), len(isolated))
            return original[:min_len].copy(), {"threshold": 0.0, "mode": "bypassed"}

        threshold = float(np.clip(config.threshold, 0.0, 1.0))
        min_len = min(len(original), len(isolated))
        orig = original[:min_len]
        iso = isolated[:min_len]

        if threshold >= 0.999 and config.ambient_bleed <= 0.001:
            return iso, {"threshold": 1.0, "mode": config.mode}
        if threshold <= 0.001:
            return orig, {"threshold": 0.0, "mode": config.mode}

        noise_residual = orig - iso

        if config.mode == "blend":
            # Coherent in-phase linear crossfade: guarantees unity gain (no +3dB equal-power constructive clipping)
            out = (1.0 - threshold) * orig + threshold * iso
            out = true_peak_mastering_limiter(out, ceiling=0.96)
            return np.ascontiguousarray(out, dtype=np.float32), {"threshold": threshold, "mode": "blend"}

        # Ephraim-Malah Decision-Directed Wiener Masking with High-Frequency Consonant Preservation
        if config.mode in ("spectral_gate", "air_preserve") and len(orig) >= 512:
            try:
                # STFT with 1024-point window and 75% overlap (256-sample hop = 5.3ms at 48kHz)
                nperseg = 1024
                noverlap = 768
                window = "hann"

                f, _, Y_stft = signal.stft(
                    orig,
                    fs=self.sample_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap
                )
                _, _, S_stft = signal.stft(
                    iso,
                    fs=self.sample_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap
                )

                Y_mag = np.abs(Y_stft)
                S_mag = np.abs(S_stft)
                N_stft = Y_stft - S_stft
                N_mag = np.abs(N_stft)

                # Noise Power Spectral Density lambda_N
                lambda_N = np.maximum(N_mag ** 2, 1e-10)
                Y_power = Y_mag ** 2

                # A posteriori SNR: gamma(k, t) = |Y(k, t)|^2 / lambda_N(k, t)
                gamma = Y_power / lambda_N

                # Ephraim-Malah Decision-Directed A Priori SNR estimation
                # xi(k, t) = alpha * (|S_prev|^2 / lambda_N_prev) + (1 - alpha) * max(0, gamma - 1)
                alpha_dd = 0.94
                num_frames = gamma.shape[1]
                xi = np.zeros_like(gamma)
                prev_S_power = np.maximum(S_mag[:, 0] ** 2, 1e-10)

                for t_idx in range(num_frames):
                    inst_snr = np.maximum(0.0, gamma[:, t_idx] - 1.0)
                    if t_idx == 0:
                        xi[:, t_idx] = inst_snr
                    else:
                        prior_est = prev_S_power / lambda_N[:, t_idx - 1]
                        xi[:, t_idx] = alpha_dd * prior_est + (1.0 - alpha_dd) * inst_snr
                    prev_S_power = S_mag[:, t_idx] ** 2

                # Soft-knee Wiener suppression gain: G_wiener = xi / (1 + xi)
                G_wiener = xi / (1.0 + xi + 1e-10)

                # User threshold exponent shaping (higher threshold = steeper suppression curve)
                wiener_exp = 0.6 + (threshold * 1.6)
                G = np.power(np.clip(G_wiener, 0.0, 1.0), wiener_exp)

                # Transient Consonant Preserver: detect high-frequency spectral flux (> 4.5 kHz)
                # Protects unvoiced speech consonants (/s/, /t/, /k/, /ch/) from being muffled
                hf_mask = f >= 4500.0
                if np.any(hf_mask) and num_frames > 2:
                    diff_hf = np.maximum(0.0, Y_mag[hf_mask, 1:] - Y_mag[hf_mask, :-1])
                    flux_hf = np.sum(diff_hf, axis=0)
                    flux_thresh = np.percentile(flux_hf, 70) if len(flux_hf) > 4 else np.median(flux_hf)
                    transient_frames = np.where(flux_hf > (flux_thresh * 1.25))[0] + 1
                    for tf in transient_frames:
                        G[hf_mask, tf] = np.maximum(G[hf_mask, tf], 0.88)

                if config.mode == "air_preserve":
                    # Air preserve: smoothly protect high studio presence above 6.5 kHz
                    air_mask = f >= 6500.0
                    min_air_bleed = max(config.ambient_bleed, ((1.0 - threshold) ** 0.8) * 0.45)
                    G[air_mask, :] = np.maximum(G[air_mask, :], min_air_bleed)

                # Minimum ambient bleed floor
                min_bleed = max(0.0, min(1.0, config.ambient_bleed))
                G = np.maximum(G, min_bleed)
                G = np.clip(G, 0.0, 1.0)

                # Apply Wiener mask to complex STFT
                Y_filtered_stft = Y_stft * G

                # Inverse STFT with Hann overlap-add reconstruction
                _, out_stft = signal.istft(
                    Y_filtered_stft,
                    fs=self.sample_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap
                )

                if len(out_stft) >= min_len:
                    out = out_stft[:min_len]
                else:
                    out = np.pad(out_stft, (0, min_len - len(out_stft)))

                if not np.any(np.isnan(out)) and not np.any(np.isinf(out)):
                    return np.ascontiguousarray(out, dtype=np.float32), {
                        "threshold": threshold,
                        "mode": config.mode,
                        "wiener_snr": True,
                        "consonant_preserver": True
                    }
            except Exception:
                pass

        # Robust time-domain fallback
        if config.mode == "air_preserve":
            attenuation = (1.0 - threshold) ** 1.2
            bleed = max(attenuation, config.ambient_bleed)
            blended = iso + noise_residual * bleed
            return np.ascontiguousarray(blended, dtype=np.float32), {"threshold": threshold, "mode": "air_preserve"}

        # Default fallback: "spectral_gate" soft mask
        attenuation_linear = ((1.0 - threshold) ** 1.5) + config.ambient_bleed
        attenuation_linear = float(np.clip(attenuation_linear, 0.0, 1.0))
        blended = iso + noise_residual * attenuation_linear
        return np.ascontiguousarray(blended, dtype=np.float32), {"threshold": threshold, "mode": "spectral_gate"}


def apply_isolation_threshold(
    original: np.ndarray,
    isolated: np.ndarray,
    threshold: float = 1.0,
    mode: str = "spectral_gate"
) -> np.ndarray:
    """Convenience helper for standalone thresholding."""
    denoiser = DeNoiser()
    cfg = DeNoiserConfig(enabled=True, threshold=threshold, mode=mode)
    out, _ = denoiser.process(original, isolated, cfg)
    return out


# -----------------------------------------------------------------------------
# 2. DE-CLICKER MODULE (DISCRETE TRANSIENT IMPULSES)
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeClickerConfig:
    enabled: bool = True
    sensitivity: float = 0.75        # 0.0 to 1.0 (higher = detects smaller mouth clicks)
    click_width_samples: int = 18    # 4 to 64 samples (~0.1ms to 1.3ms at 48kHz)
    mode: str = "mouth_clicks"       # "mouth_clicks", "digital_artifacts", "mechanical_impact", "optical_pop", "subtle"


DECLICKER_PRESETS: Dict[str, Dict[str, Any]] = {
    # --- Category: Podcast & Voiceover ---
    "mouth_saliva_tame": {
        "label": "Mouth & Saliva Clicks",
        "category": "Podcast & Voiceover",
        "sensitivity": 0.75,
        "click_width_samples": 18,
        "mode": "mouth_clicks"
    },
    "dry_mouth_smacks": {
        "label": "Dry Mouth & Tongue Smacks",
        "category": "Podcast & Voiceover",
        "sensitivity": 0.82,
        "click_width_samples": 24,
        "mode": "mouth_clicks"
    },
    "close_condenser_clicks": {
        "label": "Close Condenser Proximity Clicks",
        "category": "Podcast & Voiceover",
        "sensitivity": 0.70,
        "click_width_samples": 16,
        "mode": "mouth_clicks"
    },
    "whisper_lip_stick": {
        "label": "Whisper & Intimate Lip Stick",
        "category": "Podcast & Voiceover",
        "sensitivity": 0.88,
        "click_width_samples": 12,
        "mode": "subtle"
    },

    # --- Category: Broadcast & Live Stream ---
    "digital_buffer_clicks": {
        "label": "Digital Clock Jitter & Dropouts",
        "category": "Broadcast & Live Stream",
        "sensitivity": 0.90,
        "click_width_samples": 36,
        "mode": "digital_artifacts"
    },
    "wireless_rf_burst_clicks": {
        "label": "Wireless Mic Frequency Pops",
        "category": "Broadcast & Live Stream",
        "sensitivity": 0.84,
        "click_width_samples": 28,
        "mode": "digital_artifacts"
    },
    "bluetooth_packet_ticks": {
        "label": "Bluetooth Audio Packet Ticks",
        "category": "Broadcast & Live Stream",
        "sensitivity": 0.78,
        "click_width_samples": 20,
        "mode": "digital_artifacts"
    },
    "plosive_edge_clicks": {
        "label": "Plosive & Air Blast Edge Clicks",
        "category": "Broadcast & Live Stream",
        "sensitivity": 0.65,
        "click_width_samples": 32,
        "mode": "transient_thump"
    },

    # --- Category: Cinema & Foley ---
    "boom_pole_handling": {
        "label": "Boom Pole Tap & Cable Clicks",
        "category": "Cinema & Foley",
        "sensitivity": 0.72,
        "click_width_samples": 24,
        "mode": "mechanical_impact"
    },
    "foley_impact_soften": {
        "label": "Foley Sharp Mechanical Clicks",
        "category": "Cinema & Foley",
        "sensitivity": 0.60,
        "click_width_samples": 20,
        "mode": "mechanical_impact"
    },
    "props_cutlery_tame": {
        "label": "Props Cutlery & Glass Clinks",
        "category": "Cinema & Foley",
        "sensitivity": 0.75,
        "click_width_samples": 14,
        "mode": "mechanical_impact"
    },
    "switch_relay_pops": {
        "label": "Electrical Switch & Relay Pops",
        "category": "Cinema & Foley",
        "sensitivity": 0.85,
        "click_width_samples": 40,
        "mode": "digital_artifacts"
    },

    # --- Category: Music & Studio Vocals ---
    "lead_vocal_transparent": {
        "label": "Lead Vocal Transparent Clean",
        "category": "Music & Studio Vocals",
        "sensitivity": 0.50,
        "click_width_samples": 12,
        "mode": "subtle"
    },
    "pop_filter_bypass": {
        "label": "Pop Filter Strike Clicks",
        "category": "Music & Studio Vocals",
        "sensitivity": 0.68,
        "click_width_samples": 26,
        "mode": "transient_thump"
    },
    "acoustic_fret_bleed": {
        "label": "Acoustic Guitar Fret Bleed Clicks",
        "category": "Music & Studio Vocals",
        "sensitivity": 0.62,
        "click_width_samples": 14,
        "mode": "subtle"
    },
    "choir_group_mouth_noise": {
        "label": "Choral & Group Mouth Clicks",
        "category": "Music & Studio Vocals",
        "sensitivity": 0.58,
        "click_width_samples": 16,
        "mode": "mouth_clicks"
    },

    # --- Category: Archival & Restoration ---
    "shellac_78rpm_ticks": {
        "label": "78 RPM Shellac Needle Clicks",
        "category": "Archival & Restoration",
        "sensitivity": 0.80,
        "click_width_samples": 16,
        "mode": "optical_pop"
    },
    "vinyl_33rpm_dust_ticks": {
        "label": "33 RPM Vinyl Dust Ticks",
        "category": "Archival & Restoration",
        "sensitivity": 0.76,
        "click_width_samples": 14,
        "mode": "optical_pop"
    },
    "tape_splice_pops": {
        "label": "Tape Splice & Head Pops",
        "category": "Archival & Restoration",
        "sensitivity": 0.86,
        "click_width_samples": 44,
        "mode": "optical_pop"
    },
    "film_optical_track_pops": {
        "label": "35mm Optical Film Sound Pops",
        "category": "Archival & Restoration",
        "sensitivity": 0.82,
        "click_width_samples": 30,
        "mode": "optical_pop"
    }
}


def _solve_ar_lpc(x: np.ndarray, order: int = 16) -> Optional[np.ndarray]:
    """Computes autoregressive linear prediction coefficients via autocorrelation & Levinson-Durbin
    with pole-radius bandwidth expansion to guarantee strict asymptotic filter stability (|z| <= 0.93).
    """
    if len(x) <= order:
        return None
    r = np.correlate(x, x, mode='full')
    mid = len(x) - 1
    r = r[mid:mid + order + 1]
    if r[0] < 1e-12:
        return None
    r = r / r[0]

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    alpha = 1.0

    for i in range(1, order + 1):
        # Bound reflection coefficient to strictly inside the unit circle
        k = -np.sum(a[:i] * r[i:0:-1]) / max(alpha, 1e-12)
        k = np.clip(k, -0.985, 0.985)
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1]
        alpha *= (1.0 - k * k)
        if alpha < 1e-8:
            break

    # Bandwidth expansion: contract all AR polynomial roots radially by gamma
    # Prevents any exponential resonance or rail-to-rail oscillation
    gamma = 0.93
    damping = np.power(gamma, np.arange(1, order + 1))
    return a[1:] * damping


def _ar_interpolate_click_gap(y: np.ndarray, start: int, end: int, order: int = 16) -> np.ndarray:
    """Reconstructs click gap using joint forward-backward Autoregressive LPC prediction
    with monotonic PCHIP anchor blending and dynamic context envelope limiting.
    """
    gap_len = end - start
    context_len = max(64, order * 4)
    n_samples = len(y)

    left_ctx = y[max(0, start - context_len):start]
    right_ctx = y[end:min(n_samples, end + context_len)]

    # Compute surrounding amplitude ceiling to strictly prevent runaway divergence
    max_left = np.max(np.abs(left_ctx[-order:])) if len(left_ctx) >= order else np.max(np.abs(left_ctx)) if len(left_ctx) > 0 else 0.05
    max_right = np.max(np.abs(right_ctx[:order])) if len(right_ctx) >= order else np.max(np.abs(right_ctx)) if len(right_ctx) > 0 else 0.05
    max_env = max(max_left, max_right, 1e-4) * 1.25

    # Linear / monotonic boundary anchor
    anchors_x = [start - 1, end]
    anchors_y = [y[start - 1], y[end]]
    anchor_interp = np.interp(np.arange(start, end), anchors_x, anchors_y)

    if len(left_ctx) < order * 2 or len(right_ctx) < order * 2:
        return np.clip(anchor_interp, -max_env, max_env)

    a_left = _solve_ar_lpc(left_ctx, order)
    a_right = _solve_ar_lpc(right_ctx[::-1], order)

    if a_left is None or a_right is None:
        return np.clip(anchor_interp, -max_env, max_env)

    # Forward prediction into gap with envelope bounding
    pred_f = np.zeros(gap_len, dtype=np.float64)
    hist_f = list(left_ctx[-order:])
    for n in range(gap_len):
        val = -np.sum(a_left * np.array(hist_f[::-1]))
        val = np.clip(val, -max_env, max_env)
        pred_f[n] = val
        hist_f.pop(0)
        hist_f.append(val)

    # Backward prediction into gap with envelope bounding
    pred_b = np.zeros(gap_len, dtype=np.float64)
    hist_b = list(right_ctx[:order])
    for n in range(gap_len):
        val = -np.sum(a_right * np.array(hist_b))
        val = np.clip(val, -max_env, max_env)
        pred_b[n] = val
        hist_b.pop(0)
        hist_b.append(val)

    # Crossfade forward and backward predictions
    weights = np.linspace(0.0, 1.0, gap_len)
    ar_pred = (1.0 - weights) * pred_f + weights * pred_b[::-1]

    # Blend 85% stable AR harmonic prediction with 15% smooth boundary anchor
    reconstructed = 0.85 * ar_pred + 0.15 * anchor_interp
    return np.clip(reconstructed, -max_env, max_env)


class DeClicker:
    """Detects and repairs discrete impulsive clicks, pops, and saliva mouth noise
    using 2nd-order difference detection and Autoregressive (AR) LPC forward-backward restoration.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeClickerConfig) -> Tuple[np.ndarray, int]:
        """Removes isolated transient clicks with harmonic-preserving AR linear prediction interpolation."""
        if not config.enabled or len(audio) < 128 or config.sensitivity <= 0.01:
            return audio.copy(), 0

        y = audio.astype(np.float64, copy=True)
        n_samples = len(y)

        # 1. 2nd-order difference filter to expose discrete impulse discontinuities
        diff = np.zeros_like(y)
        diff[2:] = y[2:] - 2.0 * y[1:-1] + y[:-2]

        # 2. Local rolling RMS envelope
        window_size = 512
        kernel = np.ones(window_size) / window_size
        local_energy = signal.convolve(diff ** 2, kernel, mode='same')
        local_rms = np.sqrt(np.maximum(local_energy, 1e-9))

        # 3. Dynamic Thresholding: clamped lower bound preserves unvoiced speech consonants (/t/, /k/)
        threshold_factor = max(3.5, 7.5 - (config.sensitivity * 4.2))
        click_mask = np.abs(diff) > (threshold_factor * local_rms)

        # 4. Repair clicks via Autoregressive Linear Prediction (Cedar/RX Model)
        width = max(4, int(config.click_width_samples))
        click_indices = np.where(click_mask)[0]
        repaired_count = 0

        if len(click_indices) > 0:
            i = 0
            while i < len(click_indices):
                idx = click_indices[i]
                start = max(0, idx - width // 2)
                end = min(n_samples, idx + width // 2)

                # Group adjacent click flags
                while i + 1 < len(click_indices) and click_indices[i + 1] <= end + 2:
                    i += 1
                    end = min(n_samples, click_indices[i] + width // 2)

                gap_len = end - start
                # Strict upper limit of 48 samples (~1ms) guarantees speech vowels are never erased
                if 2 <= gap_len <= 48:
                    try:
                        y[start:end] = _ar_interpolate_click_gap(y, start, end, order=16)
                        repaired_count += 1
                    except Exception:
                        pass
                i += 1

        return np.ascontiguousarray(y, dtype=np.float32), repaired_count


# -----------------------------------------------------------------------------
# 3. DE-CRACKLER MODULE (CONTINUOUS SURFACE & GRANULAR NOISE)
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeCracklerConfig:
    enabled: bool = True
    sensitivity: float = 0.70         # 0.0 to 1.0 (crackle detection threshold)
    crackle_reduction: float = 0.65   # 0.0 to 1.0 (attenuation depth: 0% to 100%)
    frequency_band: str = "high_frequency"  # "high_frequency", "speech_presence", "full_band", "low_granular"
    density: float = 0.60             # 0.0 to 1.0 (sparse vs dense crackle treatment)


DECRACKLER_PRESETS: Dict[str, Dict[str, Any]] = {
    # --- Category: Vinyl & Phonograph Restoration ---
    "vinyl_surface_gentle": {
        "label": "Vinyl Surface Dust (Gentle)",
        "category": "Vinyl & Phonograph Restoration",
        "sensitivity": 0.60,
        "crackle_reduction": 0.45,
        "frequency_band": "high_frequency",
        "density": 0.40
    },
    "heavy_groove_wear": {
        "label": "Heavy 33 RPM Groove Wear",
        "category": "Vinyl & Phonograph Restoration",
        "sensitivity": 0.80,
        "crackle_reduction": 0.85,
        "frequency_band": "high_frequency",
        "density": 0.80
    },
    "shellac_crackle_dense": {
        "label": "78 RPM Shellac Dense Crackle",
        "category": "Vinyl & Phonograph Restoration",
        "sensitivity": 0.88,
        "crackle_reduction": 0.90,
        "frequency_band": "full_band",
        "density": 0.90
    },
    "acetate_transcription_crackle": {
        "label": "Acetate & Lacquer Disc Crackle",
        "category": "Vinyl & Phonograph Restoration",
        "sensitivity": 0.74,
        "crackle_reduction": 0.70,
        "frequency_band": "full_band",
        "density": 0.65
    },

    # --- Category: Film & Optical Soundtrack ---
    "nitrate_film_grain": {
        "label": "16mm/35mm Nitrate Optical Track",
        "category": "Film & Optical Soundtrack",
        "sensitivity": 0.75,
        "crackle_reduction": 0.75,
        "frequency_band": "speech_presence",
        "density": 0.70
    },
    "celluloid_scratch_continuous": {
        "label": "Celluloid Scratch Continuous Noise",
        "category": "Film & Optical Soundtrack",
        "sensitivity": 0.82,
        "crackle_reduction": 0.80,
        "frequency_band": "high_frequency",
        "density": 0.75
    },
    "projector_lamp_pickup": {
        "label": "Projector Optical Pickup Buzz",
        "category": "Film & Optical Soundtrack",
        "sensitivity": 0.70,
        "crackle_reduction": 0.65,
        "frequency_band": "speech_presence",
        "density": 0.60
    },
    "vintage_newsreel_wash": {
        "label": "Vintage Newsreel Crackle Wash",
        "category": "Film & Optical Soundtrack",
        "sensitivity": 0.85,
        "crackle_reduction": 0.88,
        "frequency_band": "full_band",
        "density": 0.85
    },

    # --- Category: Wardrobe & Lavalier Friction ---
    "silk_cotton_rustle": {
        "label": "Silk & Cotton Shirt Rustle",
        "category": "Wardrobe & Lavalier Friction",
        "sensitivity": 0.78,
        "crackle_reduction": 0.70,
        "frequency_band": "speech_presence",
        "density": 0.60
    },
    "heavy_coat_friction": {
        "label": "Heavy Coat & Wool Friction Crackle",
        "category": "Wardrobe & Lavalier Friction",
        "sensitivity": 0.84,
        "crackle_reduction": 0.82,
        "frequency_band": "low_granular",
        "density": 0.75
    },
    "chest_strap_rub": {
        "label": "Chest Strap & Mic Clip Rubbing",
        "category": "Wardrobe & Lavalier Friction",
        "sensitivity": 0.80,
        "crackle_reduction": 0.78,
        "frequency_band": "low_granular",
        "density": 0.70
    },
    "collar_tie_brush": {
        "label": "Tie & Collar Constant Bristle",
        "category": "Wardrobe & Lavalier Friction",
        "sensitivity": 0.72,
        "crackle_reduction": 0.65,
        "frequency_band": "speech_presence",
        "density": 0.55
    },

    # --- Category: Broadcast & RF Telemetry ---
    "wireless_rf_static": {
        "label": "Wireless RF Static Fringe",
        "category": "Broadcast & RF Telemetry",
        "sensitivity": 0.76,
        "crackle_reduction": 0.72,
        "frequency_band": "high_frequency",
        "density": 0.65
    },
    "scratchy_fader_pot": {
        "label": "Scratchy Fader & Potentiometer",
        "category": "Broadcast & RF Telemetry",
        "sensitivity": 0.82,
        "crackle_reduction": 0.80,
        "frequency_band": "full_band",
        "density": 0.70
    },
    "ground_loop_sputter": {
        "label": "Ground Buzz High-Freq Sputter",
        "category": "Broadcast & RF Telemetry",
        "sensitivity": 0.75,
        "crackle_reduction": 0.68,
        "frequency_band": "high_frequency",
        "density": 0.60
    },
    "am_radio_atmospheric": {
        "label": "AM Radio Atmospheric Static",
        "category": "Broadcast & RF Telemetry",
        "sensitivity": 0.85,
        "crackle_reduction": 0.85,
        "frequency_band": "full_band",
        "density": 0.80
    },

    # --- Category: Studio & Microphones ---
    "condenser_moisture_sputter": {
        "label": "Condenser Capsule Moisture Crackle",
        "category": "Studio & Microphones",
        "sensitivity": 0.74,
        "crackle_reduction": 0.70,
        "frequency_band": "high_frequency",
        "density": 0.55
    },
    "tube_preamp_thermal_noise": {
        "label": "Tube Preamp Thermal Sputter",
        "category": "Studio & Microphones",
        "sensitivity": 0.65,
        "crackle_reduction": 0.55,
        "frequency_band": "speech_presence",
        "density": 0.50
    },
    "phantom_power_residual": {
        "label": "Phantom Power Residual Sputter",
        "category": "Studio & Microphones",
        "sensitivity": 0.70,
        "crackle_reduction": 0.60,
        "frequency_band": "high_frequency",
        "density": 0.45
    },
    "ribbon_mic_air_crackle": {
        "label": "Ribbon Mic Air Crackle",
        "category": "Studio & Microphones",
        "sensitivity": 0.58,
        "crackle_reduction": 0.50,
        "frequency_band": "high_frequency",
        "density": 0.40
    }
}


class DeCrackler:
    """Removes dense, continuous, low-amplitude crackle, vinyl noise, and fabric friction."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeCracklerConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Suppresses granular crackle bursts via multi-band Short-Time Energy Variance limiting."""
        if not config.enabled or len(audio) < 256 or config.crackle_reduction <= 0.01:
            return audio.copy(), {"events_attenuated": 0, "reduction_applied": False}

        y = audio.astype(np.float64, copy=True)
        nyquist = self.sample_rate / 2.0

        # Select processing band
        if config.frequency_band == "high_frequency":
            sos = signal.butter(4, 4500, btype='highpass', fs=self.sample_rate, output='sos')
            band = signal.sosfiltfilt(sos, y)
            residual = y - band
        elif config.frequency_band == "speech_presence":
            sos = signal.butter(4, [1800, 6500], btype='bandpass', fs=self.sample_rate, output='sos')
            band = signal.sosfiltfilt(sos, y)
            residual = y - band
        elif config.frequency_band == "low_granular":
            sos = signal.butter(4, [400, 2500], btype='bandpass', fs=self.sample_rate, output='sos')
            band = signal.sosfiltfilt(sos, y)
            residual = y - band
        else:
            # Full band
            band = y
            residual = np.zeros_like(y)

        # Psychoacoustic Speech Formant Masking
        # Detect low-mid vocal formant energy (150Hz - 2200Hz) to model human auditory masking
        try:
            sos_formant = signal.butter(2, [150, 2200], btype='bandpass', fs=self.sample_rate, output='sos')
            speech_formants = signal.sosfiltfilt(sos_formant, y)
            formant_env = signal.convolve(np.abs(speech_formants), np.ones(512) / 512.0, mode='same')
            formant_rms = np.maximum(formant_env, 1e-6)
            masking_offset = np.clip(formant_rms / 0.15, 0.0, 1.8)
        except Exception:
            masking_offset = np.zeros(len(y))

        # Multi-resolution rolling kurtosis and variance analysis (frame size 64 samples = ~1.33ms)
        frame_len = 64
        kernel_fast = np.ones(frame_len) / frame_len
        mean_band = signal.convolve(band, kernel_fast, mode='same')
        diff = band - mean_band
        diff2 = diff ** 2
        diff4 = diff ** 4

        var_band = signal.convolve(diff2, kernel_fast, mode='same')
        m4_band = signal.convolve(diff4, kernel_fast, mode='same')
        kurt_band = m4_band / np.maximum(var_band ** 2, 1e-12)

        # Long-term noise floor RMS
        kernel_slow = np.ones(1024) / 1024
        lt_var = signal.convolve(diff2, kernel_slow, mode='same')
        lt_rms = np.sqrt(np.maximum(lt_var, 1e-10))

        # Dynamic Kurtosis and Amplitude Thresholds with Formant Masking Offset
        base_kurt_thresh = 4.8 - (config.sensitivity * 2.2)
        dynamic_kurt_thresh = base_kurt_thresh + (masking_offset * 1.5)
        crackle_flags = kurt_band > dynamic_kurt_thresh

        threshold_factor = 3.6 - (config.sensitivity * 2.0)
        crackle_limit = lt_rms * threshold_factor

        abs_band = np.abs(band)
        excess = np.maximum(0.0, abs_band - crackle_limit)
        events_attenuated = int(np.sum(crackle_flags & (excess > 0.0)))

        # Continuous soft-knee shrinkage with hyperbolic tangent knee (avoids hard boolean step clicks)
        reduction_gain = config.crackle_reduction * (0.8 + 0.4 * config.density)
        reduction_gain = float(np.clip(reduction_gain, 0.0, 0.95))

        excess_ratio = np.clip(excess / (crackle_limit + 1e-9), 0.0, 4.0)
        soft_weight = np.tanh(excess_ratio) * reduction_gain
        # Continuous kurtosis confidence weighting
        kurt_weight = np.clip((kurt_band - dynamic_kurt_thresh) / 3.0, 0.0, 1.0)
        total_suppression = np.maximum(soft_weight * 0.35, soft_weight * kurt_weight)
        clean_band = band - (band * total_suppression)
        output = residual + clean_band

        stats = {
            "events_attenuated": events_attenuated,
            "reduction_applied": True,
            "reduction_depth": config.crackle_reduction,
            "kurtosis_detection": True,
            "psychoacoustic_masking": True
        }
        return np.ascontiguousarray(output, dtype=np.float32), stats


# -----------------------------------------------------------------------------
# 4. DE-VERB MODULE (DEREVERBERATION)
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeVerbConfig:
    enabled: bool = True
    reduction_amount: float = 0.85            # 0.0 to 1.0 (0% to 100%)
    room_size: str = "small"                  # "small", "medium", "large", "hall"
    preserve_early_reflections: float = 0.40  # 0.0 to 1.0 (direct voice clarity)
    spectral_floor_db: float = -22.0          # dB floor to avoid watery artifacts


DEVERB_PRESETS: Dict[str, Dict[str, Any]] = {
    # --- Category: Podcast & Home Studio ---
    "tight_vocal_booth": {
        "label": "Treated Vocal Booth (Zero Echo)",
        "category": "Podcast & Home Studio",
        "reduction_amount": 0.85,
        "room_size": "small",
        "preserve_early_reflections": 0.40,
        "spectral_floor_db": -22.0
    },
    "untreated_bedroom_drywall": {
        "label": "Untreated Drywall & Ceiling Tame",
        "category": "Podcast & Home Studio",
        "reduction_amount": 0.68,
        "room_size": "medium",
        "preserve_early_reflections": 0.50,
        "spectral_floor_db": -18.0
    },
    "hardwood_floor_dampener": {
        "label": "Hardwood Floor Reflection Dampener",
        "category": "Podcast & Home Studio",
        "reduction_amount": 0.72,
        "room_size": "medium",
        "preserve_early_reflections": 0.45,
        "spectral_floor_db": -20.0
    },
    "glass_window_slapback": {
        "label": "Glass Window & Monitor Slapback",
        "category": "Podcast & Home Studio",
        "reduction_amount": 0.76,
        "room_size": "small",
        "preserve_early_reflections": 0.35,
        "spectral_floor_db": -20.0
    },

    # --- Category: Film & Location Sound ---
    "concrete_basement_cavern": {
        "label": "Concrete Basement & Garage Reverb",
        "category": "Film & Location Sound",
        "reduction_amount": 0.92,
        "room_size": "large",
        "preserve_early_reflections": 0.25,
        "spectral_floor_db": -25.0
    },
    "high_ceiling_warehouse": {
        "label": "High Ceiling Warehouse & Loft",
        "category": "Film & Location Sound",
        "reduction_amount": 0.88,
        "room_size": "large",
        "preserve_early_reflections": 0.30,
        "spectral_floor_db": -24.0
    },
    "concrete_stairwell_ring": {
        "label": "Concrete Stairwell Heavy Ring",
        "category": "Film & Location Sound",
        "reduction_amount": 0.95,
        "room_size": "hall",
        "preserve_early_reflections": 0.20,
        "spectral_floor_db": -26.0
    },
    "car_interior_boxiness": {
        "label": "Car Cabin Boxy Resonance",
        "category": "Film & Location Sound",
        "reduction_amount": 0.62,
        "room_size": "small",
        "preserve_early_reflections": 0.60,
        "spectral_floor_db": -16.0
    },

    # --- Category: Broadcast & Corporate ---
    "glass_boardroom_echo": {
        "label": "Glass Boardroom Conference Echo",
        "category": "Broadcast & Corporate",
        "reduction_amount": 0.80,
        "room_size": "medium",
        "preserve_early_reflections": 0.38,
        "spectral_floor_db": -22.0
    },
    "open_office_ceiling_bounce": {
        "label": "Open Office Ceiling Bounce",
        "category": "Broadcast & Corporate",
        "reduction_amount": 0.74,
        "room_size": "medium",
        "preserve_early_reflections": 0.45,
        "spectral_floor_db": -19.0
    },
    "lecture_hall_decay": {
        "label": "University Lecture Hall Decay",
        "category": "Broadcast & Corporate",
        "reduction_amount": 0.86,
        "room_size": "large",
        "preserve_early_reflections": 0.28,
        "spectral_floor_db": -23.0
    },
    "auditorium_podium_tail": {
        "label": "Auditorium Podium Long Tail",
        "category": "Broadcast & Corporate",
        "reduction_amount": 0.90,
        "room_size": "hall",
        "preserve_early_reflections": 0.22,
        "spectral_floor_db": -24.0
    },

    # --- Category: Music & Lead Vocals ---
    "lead_vocal_dry_isolation": {
        "label": "Dry Vocal Isolation (Mix Ready)",
        "category": "Music & Lead Vocals",
        "reduction_amount": 0.75,
        "room_size": "small",
        "preserve_early_reflections": 0.50,
        "spectral_floor_db": -20.0
    },
    "over_reverberated_stem_rescue": {
        "label": "Over-Reverberated Vocal Rescue",
        "category": "Music & Lead Vocals",
        "reduction_amount": 0.90,
        "room_size": "large",
        "preserve_early_reflections": 0.30,
        "spectral_floor_db": -24.0
    },
    "plate_reverb_stripper": {
        "label": "Plate Reverb Tail Tamer",
        "category": "Music & Lead Vocals",
        "reduction_amount": 0.70,
        "room_size": "medium",
        "preserve_early_reflections": 0.55,
        "spectral_floor_db": -18.0
    },
    "spring_reverb_resonance": {
        "label": "Spring Reverb Ring Suppressor",
        "category": "Music & Lead Vocals",
        "reduction_amount": 0.82,
        "room_size": "small",
        "preserve_early_reflections": 0.40,
        "spectral_floor_db": -21.0
    },

    # --- Category: Specialized & Extreme ---
    "cathedral_sanctuary_kill": {
        "label": "Cathedral Sanctuary Massive Tail",
        "category": "Specialized & Extreme",
        "reduction_amount": 0.96,
        "room_size": "hall",
        "preserve_early_reflections": 0.15,
        "spectral_floor_db": -28.0
    },
    "tiled_bathroom_flutter": {
        "label": "Tiled Bathroom Fast Flutter",
        "category": "Specialized & Extreme",
        "reduction_amount": 0.84,
        "room_size": "small",
        "preserve_early_reflections": 0.30,
        "spectral_floor_db": -22.0
    },
    "subway_platform_tunnel": {
        "label": "Subway Platform & Tunnel Echo",
        "category": "Specialized & Extreme",
        "reduction_amount": 0.94,
        "room_size": "hall",
        "preserve_early_reflections": 0.18,
        "spectral_floor_db": -26.0
    },
    "subtle_clarity_air_keeper": {
        "label": "Subtle Acoustic Air Keeper",
        "category": "Specialized & Extreme",
        "reduction_amount": 0.38,
        "room_size": "small",
        "preserve_early_reflections": 0.75,
        "spectral_floor_db": -14.0
    }
}


class DeVerb:
    """Removes room reverberation and late reflections using 4-band frequency-dependent Sabine decay modeling
    and Direct Sound Onset Protection.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeVerbConfig) -> np.ndarray:
        """Suppresses reverberant tail energy with multi-band acoustic modeling and transient preservation."""
        if not config.enabled or len(audio) < 1024 or config.reduction_amount <= 0.01:
            return audio.copy()

        n_fft = 2048
        hop_length = 512
        win = np.hanning(n_fft).astype(np.float32)

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

        # 1. Frequency-Dependent Sabine RT60 Acoustic Decay Modeling
        freqs = np.linspace(0.0, self.sample_rate / 2.0, n_freqs)
        
        # Physical RT60 profiles [Sub-Bass, Low-Mid, Mid-High, Air] in seconds
        rt60_profiles = {
            "small": [0.45, 0.35, 0.22, 0.14],
            "medium": [0.75, 0.55, 0.38, 0.20],
            "large": [1.25, 0.90, 0.60, 0.30],
            "hall": [1.90, 1.45, 0.95, 0.45]
        }
        rt60_quad = rt60_profiles.get(config.room_size, rt60_profiles["medium"])

        # Interpolate continuous RT60 curve across frequency bins
        rt60_bins = np.piecewise(
            freqs,
            [freqs < 250.0, (freqs >= 250.0) & (freqs < 1500.0), (freqs >= 1500.0) & (freqs < 5000.0), freqs >= 5000.0],
            [rt60_quad[0], rt60_quad[1], rt60_quad[2], rt60_quad[3]]
        )

        # Decay factor per frame per bin: alpha = exp(-3*ln(10) * hop / (fs * RT60))
        decay_factor = np.exp(-6.907755 * hop_length / (self.sample_rate * np.maximum(rt60_bins, 0.08)))
        decay_factor = decay_factor[:, np.newaxis]  # shape: (n_freqs, 1)

        # Delay frames before late reflection subtraction
        delay_frames = max(2, int(2 + config.preserve_early_reflections * 4))
        reverb_energy = np.zeros_like(magnitude)
        power = magnitude ** 2

        for t in range(delay_frames, n_frames):
            reverb_energy[:, t] = (
                decay_factor[:, 0] * reverb_energy[:, t - 1] +
                (1.0 - decay_factor[:, 0]) * power[:, t - delay_frames]
            )

        # 2. Direct Sound Onset Protection (Spectral Flux Transient Gating)
        spectral_flux = np.zeros(n_frames, dtype=np.float32)
        if n_frames > 1:
            diff = magnitude[:, 1:] - magnitude[:, :-1]
            pos_diff = np.maximum(0.0, diff)
            spectral_flux[1:] = np.mean(pos_diff, axis=0)

        # Identify direct speech onsets
        flux_threshold = np.mean(spectral_flux) + 0.85 * np.std(spectral_flux)
        onset_mask = spectral_flux > flux_threshold

        # 3. Dynamic Spectral Subtraction with Adaptive Floor
        beta = config.reduction_amount * 1.65
        floor_linear = 10.0 ** (config.spectral_floor_db / 20.0)

        clean_power = np.maximum(power - beta * reverb_energy, (floor_linear * magnitude) ** 2)
        clean_magnitude = np.sqrt(clean_power)

        gain = np.clip(clean_magnitude / (magnitude + 1e-9), floor_linear, 1.0)

        # Protect direct sound: freeze gain to 1.0 for 3 frames (~32ms) following vocal onset
        for t in range(n_frames):
            if onset_mask[t]:
                lock_end = min(n_frames, t + 3)
                gain[:, t:lock_end] = np.maximum(gain[:, t:lock_end], 0.95)

        # 4. Temporal Recursive Gain Smoothing (Zero Phase / No Musical Noise)
        gain_smooth = np.zeros_like(gain)
        gain_smooth[:, 0] = gain[:, 0]
        smooth_coeff = 0.72
        for t in range(1, n_frames):
            gain_smooth[:, t] = smooth_coeff * gain_smooth[:, t - 1] + (1.0 - smooth_coeff) * gain[:, t]

        final_magnitude = magnitude * gain_smooth

        _, dereverb_audio = signal.istft(
            final_magnitude * np.exp(1j * phase),
            fs=self.sample_rate,
            window=win,
            nperseg=n_fft,
            noverlap=n_fft - hop_length
        )

        dereverb_audio = dereverb_audio[:len(audio)]
        return np.ascontiguousarray(dereverb_audio, dtype=np.float32)


# -----------------------------------------------------------------------------
# 5. DE-ESSER MODULE (SIBILANCE COMPRESSION)
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DeEsserConfig:
    enabled: bool = True
    threshold_db: float = -22.0        # -36.0 dB to -10.0 dB
    reduction_amount: float = 0.65     # 0.0 to 1.0 (0% to 100%)
    frequency_focus_hz: float = 6500.0 # 4000 Hz to 9500 Hz
    bandwidth_q: float = 1.4           # 0.6 to 2.8
    mode: str = "split_band"           # "split_band" or "wideband"


DEESSER_PRESETS: Dict[str, Dict[str, Any]] = {
    # --- Category: Broadcast & Voiceover ---
    "broadcast_smooth_warmth": {
        "label": "Broadcast Warmth (Classic Radio)",
        "category": "Broadcast & Voiceover",
        "threshold_db": -22.0,
        "reduction_amount": 0.65,
        "frequency_focus_hz": 6500.0,
        "bandwidth_q": 1.4,
        "mode": "split_band"
    },
    "podcast_proximity_sibilance": {
        "label": "Podcast Proximity Sibilance",
        "category": "Broadcast & Voiceover",
        "threshold_db": -24.0,
        "reduction_amount": 0.72,
        "frequency_focus_hz": 6800.0,
        "bandwidth_q": 1.5,
        "mode": "split_band"
    },
    "audiobook_gentle_tamer": {
        "label": "Audiobook Long-Form Gentle Tamer",
        "category": "Broadcast & Voiceover",
        "threshold_db": -20.0,
        "reduction_amount": 0.50,
        "frequency_focus_hz": 6200.0,
        "bandwidth_q": 1.2,
        "mode": "split_band"
    },
    "commercial_voiceover_punch": {
        "label": "Punchy Commercial Voiceover",
        "category": "Broadcast & Voiceover",
        "threshold_db": -26.0,
        "reduction_amount": 0.78,
        "frequency_focus_hz": 7000.0,
        "bandwidth_q": 1.6,
        "mode": "split_band"
    },

    # --- Category: Female Vocals & Airy Dialogue ---
    "female_high_sibilance_8k": {
        "label": "Female High Sibilance (7.5 - 9 kHz)",
        "category": "Female Vocals & Airy Dialogue",
        "threshold_db": -24.0,
        "reduction_amount": 0.75,
        "frequency_focus_hz": 8200.0,
        "bandwidth_q": 1.6,
        "mode": "split_band"
    },
    "female_piercing_t_and_ch": {
        "label": "Female Piercing 'T' & 'CH' Softener",
        "category": "Female Vocals & Airy Dialogue",
        "threshold_db": -25.0,
        "reduction_amount": 0.80,
        "frequency_focus_hz": 7400.0,
        "bandwidth_q": 1.7,
        "mode": "split_band"
    },
    "soprano_air_keeper": {
        "label": "Soprano Air Keeper (Narrow S-Kill)",
        "category": "Female Vocals & Airy Dialogue",
        "threshold_db": -23.0,
        "reduction_amount": 0.68,
        "frequency_focus_hz": 8600.0,
        "bandwidth_q": 2.2,
        "mode": "split_band"
    },
    "bright_condenser_c9_tame": {
        "label": "Bright Condenser Mic Peak Tamer",
        "category": "Female Vocals & Airy Dialogue",
        "threshold_db": -26.0,
        "reduction_amount": 0.82,
        "frequency_focus_hz": 7800.0,
        "bandwidth_q": 1.5,
        "mode": "split_band"
    },

    # --- Category: Male Dialogue & Low Sibilance ---
    "male_dialogue_5k": {
        "label": "Male Dialogue Lower S (5.0 - 6.2 kHz)",
        "category": "Male Dialogue & Low Sibilance",
        "threshold_db": -19.0,
        "reduction_amount": 0.60,
        "frequency_focus_hz": 5400.0,
        "bandwidth_q": 1.3,
        "mode": "split_band"
    },
    "baritone_whistle_dental": {
        "label": "Baritone Whistle & Dental Air",
        "category": "Male Dialogue & Low Sibilance",
        "threshold_db": -21.0,
        "reduction_amount": 0.66,
        "frequency_focus_hz": 5800.0,
        "bandwidth_q": 1.8,
        "mode": "split_band"
    },
    "deep_voice_edge_clean": {
        "label": "Deep Voice Edge & Bite Reduction",
        "category": "Male Dialogue & Low Sibilance",
        "threshold_db": -18.0,
        "reduction_amount": 0.55,
        "frequency_focus_hz": 4800.0,
        "bandwidth_q": 1.2,
        "mode": "split_band"
    },
    "radio_host_crisp_body": {
        "label": "Radio Host Crisp Body & S Control",
        "category": "Male Dialogue & Low Sibilance",
        "threshold_db": -22.0,
        "reduction_amount": 0.64,
        "frequency_focus_hz": 6000.0,
        "bandwidth_q": 1.4,
        "mode": "split_band"
    },

    # --- Category: Cinema & ADR ---
    "adr_boom_dialogue_match": {
        "label": "ADR Boom Match (Natural Consonants)",
        "category": "Cinema & ADR",
        "threshold_db": -23.0,
        "reduction_amount": 0.62,
        "frequency_focus_hz": 6400.0,
        "bandwidth_q": 1.4,
        "mode": "split_band"
    },
    "shouted_dialogue_crunch": {
        "label": "Shouted & Screamed Sibilant Crunch",
        "category": "Cinema & ADR",
        "threshold_db": -28.0,
        "reduction_amount": 0.88,
        "frequency_focus_hz": 6800.0,
        "bandwidth_q": 1.9,
        "mode": "split_band"
    },
    "whispered_dialogue_presence": {
        "label": "Whispered Consonant Presence",
        "category": "Cinema & ADR",
        "threshold_db": -17.0,
        "reduction_amount": 0.45,
        "frequency_focus_hz": 5600.0,
        "bandwidth_q": 1.1,
        "mode": "split_band"
    },
    "dental_palatal_consonants": {
        "label": "Sharp Dental & Palatal Consonants",
        "category": "Cinema & ADR",
        "threshold_db": -25.0,
        "reduction_amount": 0.74,
        "frequency_focus_hz": 7200.0,
        "bandwidth_q": 1.7,
        "mode": "split_band"
    },

    # --- Category: Mix Mastering & Aggressive ---
    "master_bus_high_clamp": {
        "label": "Master Bus High-Band Peak Clamp",
        "category": "Mix Mastering & Aggressive",
        "threshold_db": -20.0,
        "reduction_amount": 0.50,
        "frequency_focus_hz": 7000.0,
        "bandwidth_q": 1.3,
        "mode": "split_band"
    },
    "aggressive_s_crush": {
        "label": "Aggressive S-Crush (-30 dB)",
        "category": "Mix Mastering & Aggressive",
        "threshold_db": -30.0,
        "reduction_amount": 0.92,
        "frequency_focus_hz": 6600.0,
        "bandwidth_q": 2.0,
        "mode": "split_band"
    },
    "wideband_sibilance_duck": {
        "label": "Wideband Sibilance Ducking",
        "category": "Mix Mastering & Aggressive",
        "threshold_db": -24.0,
        "reduction_amount": 0.65,
        "frequency_focus_hz": 6500.0,
        "bandwidth_q": 1.5,
        "mode": "wideband"
    },
    "surgical_notch_q28": {
        "label": "Transparent Surgical Notch (Q=2.8)",
        "category": "Mix Mastering & Aggressive",
        "threshold_db": -26.0,
        "reduction_amount": 0.85,
        "frequency_focus_hz": 7500.0,
        "bandwidth_q": 2.8,
        "mode": "split_band"
    }
}


class DeEsser:
    """Dynamic frequency-selective compressor targeting harsh sibilance ('s', 'sh', 'ch', 't')
    with 3ms Lookahead delay and vocal formant protection.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeEsserConfig) -> np.ndarray:
        """Detects sibilant peaks via 3ms lookahead sidechain and applies dynamic split-band gain reduction."""
        if not config.enabled or len(audio) < 256 or config.reduction_amount <= 0.01:
            return audio.copy()

        nyquist = self.sample_rate / 2.0
        fc = np.clip(config.frequency_focus_hz, 3000.0, nyquist - 1000.0)
        bw = fc / max(0.5, config.bandwidth_q)
        f_low = max(100.0, fc - bw / 2.0)
        f_high = min(nyquist - 100.0, fc + bw / 2.0)

        # 1. Sidechain Sibilant Band Isolation
        sos_band = signal.butter(4, [f_low, f_high], btype='bandpass', fs=self.sample_rate, output='sos')
        sibilant_band = signal.sosfiltfilt(sos_band, audio)

        # 2. Sidechain Envelope Detection with 0.2ms Ultra-Fast Attack & 35ms Release
        attack_coef = math.exp(-1.0 / (0.0002 * self.sample_rate))  # 0.2ms
        release_coef = math.exp(-1.0 / (0.035 * self.sample_rate))  # 35ms

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

        threshold_linear = 10.0 ** (config.threshold_db / 20.0)
        max_reduction_linear = 1.0 - (config.reduction_amount * 0.90)

        # 3. Dynamic Gain Reduction Calculation
        raw_gain = np.ones_like(envelope)
        over_threshold = envelope > threshold_linear
        if np.any(over_threshold):
            excess_ratio = (envelope[over_threshold] - threshold_linear) / (envelope[over_threshold] + 1e-6)
            raw_gain[over_threshold] = np.maximum(
                max_reduction_linear,
                1.0 - config.reduction_amount * excess_ratio
            )

        # 4. 3ms Lookahead Alignment & Ballistics Smoothing (Prevents audio-rate modulation crackle)
        lookahead_samples = int(0.003 * self.sample_rate)  # ~144 samples at 48kHz
        lookahead_gain = np.ones_like(raw_gain)
        if len(raw_gain) > lookahead_samples:
            # Shift gain reduction backwards in time so it attacks BEFORE the peak
            lookahead_gain[:-lookahead_samples] = raw_gain[lookahead_samples:]
            lookahead_gain[-lookahead_samples:] = raw_gain[-1]
        else:
            lookahead_gain = raw_gain

        # One-pole dynamic ballistics filter: 1.5ms attack / 25ms release
        smooth_gain = np.ones_like(lookahead_gain)
        att_alpha = math.exp(-1.0 / (0.0015 * self.sample_rate))
        rel_alpha = math.exp(-1.0 / (0.025 * self.sample_rate))
        curr_g = 1.0
        for n in range(len(lookahead_gain)):
            target_g = lookahead_gain[n]
            if target_g < curr_g:
                curr_g = att_alpha * curr_g + (1.0 - att_alpha) * target_g
            else:
                curr_g = rel_alpha * curr_g + (1.0 - rel_alpha) * target_g
            smooth_gain[n] = curr_g

        # 5. Split-Band / Wideband Application
        if config.mode == "split_band":
            residual_speech = audio - sibilant_band
            output = residual_speech + (sibilant_band * smooth_gain)
        else:
            output = audio * smooth_gain

        return np.ascontiguousarray(output, dtype=np.float32)


# -----------------------------------------------------------------------------
# 6. DENOX AUDIO MASTERING CHAIN ORCHESTRATOR
# -----------------------------------------------------------------------------

DEFAULT_CHAIN_ORDER: List[str] = ["denoiser", "declicker", "decrackler", "deverb", "deesser"]


@dataclasses.dataclass
class DenoxMasteringChainConfig:
    # Dedicated 5-Stage Audio Modules
    denoiser: DeNoiserConfig = dataclasses.field(default_factory=DeNoiserConfig)
    declicker: DeClickerConfig = dataclasses.field(default_factory=DeClickerConfig)
    decrackler: DeCracklerConfig = dataclasses.field(default_factory=DeCracklerConfig)
    deverb: DeVerbConfig = dataclasses.field(default_factory=DeVerbConfig)
    deesser: DeEsserConfig = dataclasses.field(default_factory=DeEsserConfig)
    chain_order: List[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_CHAIN_ORDER))

    # Backwards compatibility helper for isolation threshold
    @property
    def isolation_threshold(self) -> float:
        return self.denoiser.threshold

    @isolation_threshold.setter
    def isolation_threshold(self, val: float):
        self.denoiser.threshold = val

    @property
    def threshold_mode(self) -> str:
        return self.denoiser.mode

    @threshold_mode.setter
    def threshold_mode(self, val: str):
        self.denoiser.mode = val


GLOBAL_CHAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # CATEGORY 1: PODCAST, VOICEOVER & AUDIOBOOK
    # =========================================================================
    "ultra_clean_podcast": {
        "name": "Ultra-Clean Podcast (Max Clarity)",
        "category": "Podcast, Voiceover & Audiobook",
        "denoiser": {"enabled": True, "preset": "pure_isolated_vocal", "threshold": 0.95},
        "declicker": {"enabled": True, "preset": "mouth_saliva_tame"},
        "decrackler": {"enabled": True, "preset": "silk_cotton_rustle"},
        "deverb": {"enabled": True, "preset": "tight_vocal_booth"},
        "deesser": {"enabled": True, "preset": "podcast_proximity_sibilance"}
    },
    "audiobook_intimate_narration": {
        "name": "Audiobook Intimate Narration",
        "category": "Podcast, Voiceover & Audiobook",
        "denoiser": {"enabled": True, "preset": "whisper_room_air", "threshold": 0.70},
        "declicker": {"enabled": True, "preset": "whisper_lip_stick"},
        "decrackler": {"enabled": False, "preset": "condenser_moisture_sputter"},
        "deverb": {"enabled": True, "preset": "untreated_bedroom_drywall"},
        "deesser": {"enabled": True, "preset": "audiobook_gentle_tamer"}
    },
    "commercial_voiceover_punch": {
        "name": "Commercial Voiceover Punch",
        "category": "Podcast, Voiceover & Audiobook",
        "denoiser": {"enabled": True, "preset": "pure_isolated_vocal", "threshold": 1.00},
        "declicker": {"enabled": True, "preset": "close_condenser_clicks"},
        "decrackler": {"enabled": False, "preset": "collar_tie_brush"},
        "deverb": {"enabled": True, "preset": "tight_vocal_booth"},
        "deesser": {"enabled": True, "preset": "commercial_voiceover_punch"}
    },
    "two_host_podcast_roundtable": {
        "name": "Multi-Mic Roundtable & Cross-Bleed",
        "category": "Podcast, Voiceover & Audiobook",
        "denoiser": {"enabled": True, "preset": "podcast_hvac_tame", "threshold": 0.88},
        "declicker": {"enabled": True, "preset": "dry_mouth_smacks"},
        "decrackler": {"enabled": True, "preset": "chest_strap_rub"},
        "deverb": {"enabled": True, "preset": "hardwood_floor_dampener"},
        "deesser": {"enabled": True, "preset": "radio_host_crisp_body"}
    },

    # =========================================================================
    # CATEGORY 2: BROADCAST, TV & STREAMING
    # =========================================================================
    "broadcast_newsroom_master": {
        "name": "Broadcast Newsroom & Live Stream",
        "category": "Broadcast, TV & Streaming",
        "denoiser": {"enabled": True, "preset": "city_traffic_rumble", "threshold": 0.88},
        "declicker": {"enabled": True, "preset": "digital_buffer_clicks"},
        "decrackler": {"enabled": True, "preset": "wireless_rf_static"},
        "deverb": {"enabled": True, "preset": "glass_boardroom_echo"},
        "deesser": {"enabled": True, "preset": "broadcast_smooth_warmth"}
    },
    "live_sports_stadium_isolate": {
        "name": "Live Sports Commentary & Stadium Noise",
        "category": "Broadcast, TV & Streaming",
        "denoiser": {"enabled": True, "preset": "cafeteria_crowd_babble", "threshold": 0.96},
        "declicker": {"enabled": True, "preset": "wireless_rf_burst_clicks"},
        "decrackler": {"enabled": True, "preset": "am_radio_atmospheric"},
        "deverb": {"enabled": True, "preset": "auditorium_podium_tail"},
        "deesser": {"enabled": True, "preset": "shouted_dialogue_crunch"}
    },
    "field_reporter_wind_traffic": {
        "name": "Field Reporter & Severe Weather",
        "category": "Broadcast, TV & Streaming",
        "denoiser": {"enabled": True, "preset": "outdoor_wind_buffet", "threshold": 0.95},
        "declicker": {"enabled": True, "preset": "plosive_edge_clicks"},
        "decrackler": {"enabled": True, "preset": "heavy_coat_friction"},
        "deverb": {"enabled": True, "preset": "subway_platform_tunnel"},
        "deesser": {"enabled": True, "preset": "adr_boom_dialogue_match"}
    },
    "corporate_conference_hall": {
        "name": "Corporate Townhall & Conference Echo",
        "category": "Broadcast, TV & Streaming",
        "denoiser": {"enabled": True, "preset": "pc_fan_laptop_whine", "threshold": 0.85},
        "declicker": {"enabled": True, "preset": "bluetooth_packet_ticks"},
        "decrackler": {"enabled": True, "preset": "ground_loop_sputter"},
        "deverb": {"enabled": True, "preset": "lecture_hall_decay"},
        "deesser": {"enabled": True, "preset": "dental_palatal_consonants"}
    },

    # =========================================================================
    # CATEGORY 3: FILM, CINEMA & ADR DIALOGUE
    # =========================================================================
    "cinema_dialogue_adr": {
        "name": "Cinema Dialogue & Location Boom",
        "category": "Film, Cinema & ADR Dialogue",
        "denoiser": {"enabled": True, "preset": "on_location_generator", "threshold": 0.95},
        "declicker": {"enabled": True, "preset": "boom_pole_handling"},
        "decrackler": {"enabled": True, "preset": "silk_cotton_rustle"},
        "deverb": {"enabled": True, "preset": "high_ceiling_warehouse"},
        "deesser": {"enabled": True, "preset": "adr_boom_dialogue_match"}
    },
    "car_interior_dialogue_clean": {
        "name": "Car Interior Road & Cabin Noise",
        "category": "Film, Cinema & ADR Dialogue",
        "denoiser": {"enabled": True, "preset": "city_traffic_rumble", "threshold": 0.92},
        "declicker": {"enabled": True, "preset": "foley_impact_soften"},
        "decrackler": {"enabled": True, "preset": "collar_tie_brush"},
        "deverb": {"enabled": True, "preset": "car_interior_boxiness"},
        "deesser": {"enabled": True, "preset": "bright_condenser_c9_tame"}
    },
    "whispered_cinema_intimate": {
        "name": "Whispered Dialogue & Foley Air",
        "category": "Film, Cinema & ADR Dialogue",
        "denoiser": {"enabled": True, "preset": "whisper_room_air", "threshold": 0.65},
        "declicker": {"enabled": True, "preset": "whisper_lip_stick"},
        "decrackler": {"enabled": False, "preset": "condenser_moisture_sputter"},
        "deverb": {"enabled": True, "preset": "subtle_clarity_air_keeper"},
        "deesser": {"enabled": True, "preset": "whispered_dialogue_presence"}
    },
    "heavy_room_killer": {
        "name": "Echo & Concrete Cavern Eliminator",
        "category": "Film, Cinema & ADR Dialogue",
        "denoiser": {"enabled": True, "preset": "pure_isolated_vocal", "threshold": 1.00},
        "declicker": {"enabled": True, "preset": "switch_relay_pops"},
        "decrackler": {"enabled": True, "preset": "scratchy_fader_pot"},
        "deverb": {"enabled": True, "preset": "concrete_stairwell_ring"},
        "deesser": {"enabled": True, "preset": "surgical_notch_q28"}
    },

    # =========================================================================
    # CATEGORY 4: MUSIC PRODUCTION & LEAD VOCALS
    # =========================================================================
    "studio_vocal_booth": {
        "name": "Studio Vocal Booth Master (Mix Ready)",
        "category": "Music Production & Lead Vocals",
        "denoiser": {"enabled": True, "preset": "pure_isolated_vocal", "threshold": 1.00},
        "declicker": {"enabled": True, "preset": "lead_vocal_transparent"},
        "decrackler": {"enabled": False, "preset": "condenser_moisture_sputter"},
        "deverb": {"enabled": True, "preset": "lead_vocal_dry_isolation"},
        "deesser": {"enabled": True, "preset": "master_bus_high_clamp"}
    },
    "live_stage_vocal_drum_bleed": {
        "name": "Live Concert Vocal & Drum Spill Kill",
        "category": "Music Production & Lead Vocals",
        "denoiser": {"enabled": True, "preset": "stage_drum_spill_bleed", "threshold": 0.98},
        "declicker": {"enabled": True, "preset": "pop_filter_bypass"},
        "decrackler": {"enabled": True, "preset": "tube_preamp_thermal_noise"},
        "deverb": {"enabled": True, "preset": "cathedral_sanctuary_kill"},
        "deesser": {"enabled": True, "preset": "aggressive_s_crush"}
    },
    "acoustic_singer_songwriter": {
        "name": "Acoustic Singer-Songwriter Balance",
        "category": "Music Production & Lead Vocals",
        "denoiser": {"enabled": True, "preset": "acoustic_room_subtle_clean", "threshold": 0.75},
        "declicker": {"enabled": True, "preset": "acoustic_fret_bleed"},
        "decrackler": {"enabled": False, "preset": "ribbon_mic_air_crackle"},
        "deverb": {"enabled": True, "preset": "spring_reverb_resonance"},
        "deesser": {"enabled": True, "preset": "soprano_air_keeper"}
    },
    "backing_vocal_ensemble": {
        "name": "Backing Vocal & Choral Group Polish",
        "category": "Music Production & Lead Vocals",
        "denoiser": {"enabled": True, "preset": "backing_vocal_bleed", "threshold": 0.90},
        "declicker": {"enabled": True, "preset": "choir_group_mouth_noise"},
        "decrackler": {"enabled": False, "preset": "condenser_moisture_sputter"},
        "deverb": {"enabled": True, "preset": "plate_reverb_stripper"},
        "deesser": {"enabled": True, "preset": "wideband_sibilance_duck"}
    },

    # =========================================================================
    # CATEGORY 5: ARCHIVAL, VINYL & EXTREME RESTORATION
    # =========================================================================
    "vintage_vinyl_restoration": {
        "name": "Vintage Vinyl & Archival 33/78 RPM",
        "category": "Archival, Vinyl & Extreme Restoration",
        "denoiser": {"enabled": True, "preset": "vintage_radio_static_floor", "threshold": 0.80},
        "declicker": {"enabled": True, "preset": "vinyl_33rpm_dust_ticks"},
        "decrackler": {"enabled": True, "preset": "heavy_groove_wear"},
        "deverb": {"enabled": True, "preset": "subtle_clarity_air_keeper"},
        "deesser": {"enabled": False, "preset": "broadcast_smooth_warmth"}
    },
    "cassette_tape_hiss_restore": {
        "name": "Cassette & Reel-to-Reel Tape Hiss Restoration",
        "category": "Archival, Vinyl & Extreme Restoration",
        "denoiser": {"enabled": True, "preset": "analog_tape_hiss_wash", "threshold": 0.88},
        "declicker": {"enabled": True, "preset": "tape_splice_pops"},
        "decrackler": {"enabled": True, "preset": "phantom_power_residual"},
        "deverb": {"enabled": False, "preset": "tight_vocal_booth"},
        "deesser": {"enabled": True, "preset": "deep_voice_edge_clean"}
    },
    "optical_film_soundtrack_rescue": {
        "name": "16mm/35mm Optical Film Soundtrack Rescue",
        "category": "Archival, Vinyl & Extreme Restoration",
        "denoiser": {"enabled": True, "preset": "industrial_plant_drone", "threshold": 0.92},
        "declicker": {"enabled": True, "preset": "film_optical_track_pops"},
        "decrackler": {"enabled": True, "preset": "nitrate_film_grain"},
        "deverb": {"enabled": True, "preset": "tiled_bathroom_flutter"},
        "deesser": {"enabled": True, "preset": "female_piercing_t_and_ch"}
    },
    "electrical_hum_buzz_eliminator": {
        "name": "50Hz/60Hz Ground Loop & Severe Electrical Buzz",
        "category": "Archival, Vinyl & Extreme Restoration",
        "denoiser": {"enabled": True, "preset": "mains_hum_50_60hz", "threshold": 0.96},
        "declicker": {"enabled": True, "preset": "switch_relay_pops"},
        "decrackler": {"enabled": True, "preset": "ground_loop_sputter"},
        "deverb": {"enabled": False, "preset": "tight_vocal_booth"},
        "deesser": {"enabled": True, "preset": "baritone_whistle_dental"}
    }
}


class DenoxMasteringProcessor:
    """Runs the modular mastering chain with arbitrary sequence altering:
    Supports custom chain ordering, enabling, bypassing, adding, and removing modules.
    Default: De-Noiser -> De-Clicker -> De-Crackler -> De-Verb -> De-Esser.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.denoiser = DeNoiser(sample_rate)
        self.declicker = DeClicker(sample_rate)
        self.decrackler = DeCrackler(sample_rate)
        self.deverb = DeVerb(sample_rate)
        self.deesser = DeEsser(sample_rate)

    def process_chain(
        self,
        original_audio: np.ndarray,
        isolated_vocal: np.ndarray,
        config: DenoxMasteringChainConfig
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Processes audio dynamically following user-specified module chain order."""
        # Sanitize and validate active chain order
        requested_order = [str(m).lower() for m in (config.chain_order or DEFAULT_CHAIN_ORDER) if str(m).lower() in DEFAULT_CHAIN_ORDER]
        if not requested_order:
            requested_order = list(DEFAULT_CHAIN_ORDER)

        # Initial signal starts as isolated neural stem if available, else original raw audio
        current_audio = isolated_vocal.copy() if isolated_vocal is not None and len(isolated_vocal) > 0 else original_audio.copy()

        stats: Dict[str, Any] = {
            "chain_order": requested_order,
            "threshold_applied": config.denoiser.threshold if "denoiser" in requested_order and config.denoiser.enabled else 0.0,
            "denoiser_applied": "denoiser" in requested_order and config.denoiser.enabled,
            "denoiser_mode": "bypassed",
            "clicks_repaired": 0,
            "crackle_events": 0,
            "decrackler_applied": "decrackler" in requested_order and config.decrackler.enabled,
            "deverb_applied": "deverb" in requested_order and config.deverb.enabled,
            "deesser_applied": "deesser" in requested_order and config.deesser.enabled,
            "executed_modules": []
        }

        # Sequentially route audio through only the active modules in exact chain order
        for mod_name in requested_order:
            if mod_name == "denoiser" and config.denoiser.enabled:
                # If denoiser is first in the active chain, it blends original with isolated vocal.
                # If placed later, it blends current intermediate audio with isolated vocal.
                input_mix = original_audio if len(stats["executed_modules"]) == 0 else current_audio
                current_audio, denoise_stats = self.denoiser.process(
                    input_mix,
                    isolated_vocal if isolated_vocal is not None else original_audio,
                    config.denoiser
                )
                stats["denoiser_mode"] = denoise_stats.get("mode", "spectral_gate")
                stats["executed_modules"].append("denoiser")

            elif mod_name == "declicker" and config.declicker.enabled:
                current_audio, clicks = self.declicker.process(current_audio, config.declicker)
                stats["clicks_repaired"] = clicks
                stats["executed_modules"].append("declicker")

            elif mod_name == "decrackler" and config.decrackler.enabled:
                current_audio, crackle_stats = self.decrackler.process(current_audio, config.decrackler)
                stats["crackle_events"] = crackle_stats.get("events_attenuated", 0)
                stats["executed_modules"].append("decrackler")

            elif mod_name == "deverb" and config.deverb.enabled:
                current_audio = self.deverb.process(current_audio, config.deverb)
                stats["executed_modules"].append("deverb")

            elif mod_name == "deesser" and config.deesser.enabled:
                current_audio = self.deesser.process(current_audio, config.deesser)
                stats["executed_modules"].append("deesser")

        min_len = min(len(original_audio), len(current_audio))
        # Studio True-Peak Limiter (-0.35 dBFS ceiling): eliminates digital 16-bit rail clipping
        vocal = true_peak_mastering_limiter(current_audio[:min_len], ceiling=0.96)
        raw_background = original_audio[:min_len] - vocal
        background = true_peak_mastering_limiter(raw_background, ceiling=0.96)

        return vocal, background, stats
