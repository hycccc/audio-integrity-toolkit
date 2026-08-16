"""Spectral-integrity analysis for audio files.

Detects fake-lossless audio (lossy transcodes re-wrapped in a lossless
container) via high-frequency cutoff analysis, plus clipping and
dynamic-range checks. Designed as an acceptance gate for audio dataset
ingestion: cheap enough to run on every file, strict enough to catch the
most common vendor-delivery defects.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
from scipy.signal import welch

# Known low-pass signatures of common lossy encoders. A file whose active
# spectrum stops near one of these cutoffs was almost certainly transcoded
# from that codec, no matter what container it arrives in.
TRANSCODE_CUTOFFS: dict[int, str] = {
    16000: "suspected MP3 128kbps transcode",
    19000: "suspected MP3 192kbps transcode",
    20000: "suspected MP3 320kbps / AAC transcode",
}

# Energy this far below the spectral peak is treated as silence when
# locating the highest active frequency.
FLOOR_DB = 80.0

# A spectrum extending to within this margin of Nyquist is considered
# genuinely full-band.
NYQUIST_MARGIN_HZ = 1500.0

CUTOFF_TOLERANCE_HZ = 500.0

VERDICT_LOSSLESS = "lossless (spectrum extends to Nyquist)"


@dataclass
class Report:
    """Structured result of a single-file integrity analysis."""

    path: str
    sample_rate: int
    channels: int
    duration_s: float
    subtype: str
    peak_dbfs: float
    clipped_samples: int
    rms_dbfs: float
    crest_db: float
    cutoff_hz: float
    nyquist_hz: float
    verdict: str
    lossless: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _spectral_cutoff(mono: np.ndarray, sr: int) -> float:
    """Highest frequency with energy above the relative noise floor."""
    nperseg = min(8192, len(mono))
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg)
    psd_db = 10 * np.log10(psd + 1e-20)
    active = freqs[psd_db > psd_db.max() - FLOOR_DB]
    return float(active.max()) if len(active) else 0.0


def _verdict(cutoff: float, nyquist: float) -> tuple[str, bool]:
    if cutoff > nyquist - NYQUIST_MARGIN_HZ:
        return VERDICT_LOSSLESS, True
    for threshold, message in sorted(TRANSCODE_CUTOFFS.items()):
        if cutoff < threshold + CUTOFF_TOLERANCE_HZ:
            return message, False
    return VERDICT_LOSSLESS, True


def analyze(path: str) -> Report:
    """Analyze one audio file and return a structured integrity report."""
    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    nyquist = sr / 2

    peak = float(np.max(np.abs(mono)))
    peak_db = 20 * np.log10(peak + 1e-12)
    clipped = int(np.sum(np.abs(mono) >= 0.999))

    rms = float(np.sqrt(np.mean(mono**2)))
    rms_db = 20 * np.log10(rms + 1e-12)

    cutoff = _spectral_cutoff(mono, sr)
    verdict, lossless = _verdict(cutoff, nyquist)

    return Report(
        path=path,
        sample_rate=sr,
        channels=data.shape[1],
        duration_s=len(mono) / sr,
        subtype=sf.info(path).subtype,
        peak_dbfs=peak_db,
        clipped_samples=clipped,
        rms_dbfs=rms_db,
        crest_db=peak_db - rms_db,
        cutoff_hz=cutoff,
        nyquist_hz=nyquist,
        verdict=verdict,
        lossless=lossless,
    )
