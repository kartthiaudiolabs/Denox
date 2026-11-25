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
        angle = threshold * (math.pi / 2.0)
        out = math.cos(angle) * orig + math.sin(angle) * iso
        return np.ascontiguousarray(out, dtype=np.float32)

    noise_residual = orig - iso
    attenuation_linear = (1.0 - threshold) ** 1.5
    blended = iso + noise_residual * attenuation_linear
    return np.ascontiguousarray(blended, dtype=np.float32)


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


class DeClicker:
    """Detects and repairs discrete impulsive clicks, pops, and saliva mouth noise."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeClickerConfig) -> Tuple[np.ndarray, int]:
        """Removes isolated transient clicks using high-frequency differentiation and CubicSpline interpolation."""
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

        # 3. Dynamic Thresholding based on sensitivity
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

                # Group adjacent click flags
                while i + 1 < len(click_indices) and click_indices[i + 1] <= end + 4:
                    i += 1
                    end = min(n_samples, click_indices[i] + width // 2)

                gap_len = end - start
                if 2 <= gap_len <= 128:
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

        # Multi-resolution short-term variance envelope (frame size 128 samples = ~2.6ms)
        frame_len = 128
        band_power = band ** 2
        kernel_fast = np.ones(frame_len) / frame_len
        st_energy = signal.convolve(band_power, kernel_fast, mode='same')

        kernel_slow = np.ones(1024) / 1024
        lt_energy = signal.convolve(band_power, kernel_slow, mode='same')
        lt_rms = np.sqrt(np.maximum(lt_energy, 1e-9))

        # Sensitivity sets the crackle limit threshold relative to local RMS
        threshold_factor = 4.2 - (config.sensitivity * 2.6)
        crackle_limit = lt_rms * threshold_factor

        abs_band = np.abs(band)
        excess = np.maximum(0.0, abs_band - crackle_limit)
        events_attenuated = int(np.sum(excess > 0.0))

        # Soft-knee hyperbolic tangent attenuation to prevent clipping/chatter
        reduction_gain = config.crackle_reduction * (0.8 + 0.4 * config.density)
        suppression = np.sign(band) * excess * np.clip(reduction_gain, 0.0, 0.95)

        clean_band = band - suppression
        output = residual + clean_band

        stats = {
            "events_attenuated": events_attenuated,
            "reduction_applied": True,
            "reduction_depth": config.crackle_reduction
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
    """Removes room reverberation and late reflections using multi-band spectral decay modeling."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, config: DeVerbConfig) -> np.ndarray:
        """Suppresses reverberant tail energy via multi-band late reverberation subtraction."""
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

        room_decay = {
            "small": 0.82,
            "medium": 0.89,
            "large": 0.94,
            "hall": 0.97
        }.get(config.room_size, 0.89)

        delay_frames = max(2, int(2 + config.preserve_early_reflections * 4))
        reverb_energy = np.zeros_like(magnitude)
        power = magnitude ** 2

        for t in range(delay_frames, n_frames):
            reverb_energy[:, t] = (
                room_decay * reverb_energy[:, t - 1] +
                (1.0 - room_decay) * power[:, t - delay_frames]
            )

        beta = config.reduction_amount * 1.6
        floor_linear = 10.0 ** (config.spectral_floor_db / 20.0)

        clean_power = np.maximum(power - beta * reverb_energy, (floor_linear * magnitude) ** 2)
        clean_magnitude = np.sqrt(clean_power)

        gain = clean_magnitude / (magnitude + 1e-9)
        gain_smooth = signal.medfilt2d(gain, kernel_size=(3, 3))
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

        sos_band = signal.butter(4, [f_low, f_high], btype='bandpass', fs=self.sample_rate, output='sos')
        sibilant_band = signal.sosfiltfilt(sos_band, audio)

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

        threshold_linear = 10.0 ** (config.threshold_db / 20.0)
        max_reduction_linear = 1.0 - (config.reduction_amount * 0.90)

        gain = np.ones_like(envelope)
        over_threshold = envelope > threshold_linear
        if np.any(over_threshold):
            excess_ratio = (envelope[over_threshold] - threshold_linear) / (envelope[over_threshold] + 1e-6)
            gain[over_threshold] = np.maximum(
                max_reduction_linear,
                1.0 - config.reduction_amount * excess_ratio
            )

        if config.mode == "split_band":
            residual_speech = audio - sibilant_band
            output = residual_speech + (sibilant_band * gain)
        else:
            output = audio * gain

        return np.ascontiguousarray(output, dtype=np.float32)


# -----------------------------------------------------------------------------
# 6. DENOX AUDIO MASTERING CHAIN ORCHESTRATOR
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class DenoxMasteringChainConfig:
    isolation_threshold: float = 1.0
    threshold_mode: str = "spectral_gate"

    # Distinct Modules
    declicker: DeClickerConfig = dataclasses.field(default_factory=DeClickerConfig)
    decrackler: DeCracklerConfig = dataclasses.field(default_factory=DeCracklerConfig)
    deverb: DeVerbConfig = dataclasses.field(default_factory=DeVerbConfig)
    deesser: DeEsserConfig = dataclasses.field(default_factory=DeEsserConfig)


GLOBAL_CHAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "ultra_clean_podcast": {
        "name": "Ultra-Clean Podcast (Max Clarity)",
        "isolation_threshold": 0.92,
        "declicker": {"enabled": True, "preset": "mouth_saliva_tame"},
        "decrackler": {"enabled": True, "preset": "silk_cotton_rustle"},
        "deverb": {"enabled": True, "preset": "tight_vocal_booth"},
        "deesser": {"enabled": True, "preset": "podcast_proximity_sibilance"}
    },
    "studio_vocal_booth": {
        "name": "Studio Vocal Booth Master",
        "isolation_threshold": 1.00,
        "declicker": {"enabled": True, "preset": "lead_vocal_transparent"},
        "decrackler": {"enabled": False, "preset": "condenser_moisture_sputter"},
        "deverb": {"enabled": True, "preset": "tight_vocal_booth"},
        "deesser": {"enabled": True, "preset": "broadcast_smooth_warmth"}
    },
    "natural_warmth": {
        "name": "Natural Warmth (Retain Room Air)",
        "isolation_threshold": 0.70,
        "declicker": {"enabled": True, "preset": "mouth_saliva_tame"},
        "decrackler": {"enabled": False, "preset": "vinyl_surface_gentle"},
        "deverb": {"enabled": False, "preset": "subtle_clarity_air_keeper"},
        "deesser": {"enabled": True, "preset": "audiobook_gentle_tamer"}
    },
    "heavy_room_killer": {
        "name": "Echo & Cavern Eliminator",
        "isolation_threshold": 1.00,
        "declicker": {"enabled": True, "preset": "digital_buffer_clicks"},
        "decrackler": {"enabled": True, "preset": "wireless_rf_static"},
        "deverb": {"enabled": True, "preset": "concrete_stairwell_ring"},
        "deesser": {"enabled": True, "preset": "shouted_dialogue_crunch"}
    },
    "vintage_vinyl_restoration": {
        "name": "Vintage Vinyl & Archival Restoration",
        "isolation_threshold": 0.80,
        "declicker": {"enabled": True, "preset": "vinyl_33rpm_dust_ticks"},
        "decrackler": {"enabled": True, "preset": "heavy_groove_wear"},
        "deverb": {"enabled": True, "preset": "subtle_clarity_air_keeper"},
        "deesser": {"enabled": False, "preset": "broadcast_smooth_warmth"}
    },
    "cinema_dialogue_adr": {
        "name": "Cinema Dialogue & On-Location Boom",
        "isolation_threshold": 0.95,
        "declicker": {"enabled": True, "preset": "boom_pole_handling"},
        "decrackler": {"enabled": True, "preset": "silk_cotton_rustle"},
        "deverb": {"enabled": True, "preset": "high_ceiling_warehouse"},
        "deesser": {"enabled": True, "preset": "adr_boom_dialogue_match"}
    },
    "broadcast_newsroom": {
        "name": "Broadcast Newsroom & Live Stream",
        "isolation_threshold": 0.88,
        "declicker": {"enabled": True, "preset": "wireless_rf_burst_clicks"},
        "decrackler": {"enabled": True, "preset": "wireless_rf_static"},
        "deverb": {"enabled": True, "preset": "glass_boardroom_echo"},
        "deesser": {"enabled": True, "preset": "commercial_voiceover_punch"}
    }
}


class DenoxMasteringProcessor:
    """Runs the complete mastering chain:
    Thresholding -> De-Clicker -> De-Crackler -> De-Verb -> De-Esser.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
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
        """Processes the neural vocal through thresholding and all DSP modules."""
        vocal = apply_isolation_threshold(
            original_audio,
            isolated_vocal,
            threshold=config.isolation_threshold,
            mode=config.threshold_mode
        )

        stats: Dict[str, Any] = {
            "threshold_applied": config.isolation_threshold,
            "clicks_repaired": 0,
            "crackle_events": 0,
            "decrackler_applied": config.decrackler.enabled,
            "deverb_applied": config.deverb.enabled,
            "deesser_applied": config.deesser.enabled
        }

        # Step 2: De-Clicker
        if config.declicker.enabled:
            vocal, clicks = self.declicker.process(vocal, config.declicker)
            stats["clicks_repaired"] = clicks

        # Step 3: De-Crackler
        if config.decrackler.enabled:
            vocal, crackle_stats = self.decrackler.process(vocal, config.decrackler)
            stats["crackle_events"] = crackle_stats.get("events_attenuated", 0)

        # Step 4: De-Verb
        if config.deverb.enabled:
            vocal = self.deverb.process(vocal, config.deverb)

        # Step 5: De-Esser
        if config.deesser.enabled:
            vocal = self.deesser.process(vocal, config.deesser)

        min_len = min(len(original_audio), len(vocal))
        background = original_audio[:min_len] - vocal[:min_len]

        return vocal[:min_len], background, stats
