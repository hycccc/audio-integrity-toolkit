"""Metadata–audio matching: does the file's paperwork match its signal?

At ingestion scale the audio and its metadata arrive from different
systems — and disagree constantly. The defects this module catches are
the ones that actually show up in vendor deliveries:

- **Container lies** — the extension says ``.wav`` but the bytes are
  FLAC (or vice versa). Downstream tools that trust the extension
  misparse or silently re-decode.
- **Header/decode duration mismatch** — the header promises more frames
  than decode produces: a truncated upload or interrupted transfer.
- **Padding** — seconds of leading/trailing silence, typical of sloppy
  cuts, which corrupt any duration- or alignment-based pairing with
  lyrics or annotations downstream.
- **Implausible parameters** — nonstandard sample rates and channel
  layouts that are usually conversion accidents, not artistic choices.

Tag-level checks (does the embedded title match the sidecar metadata?)
belong to a fuzzy-matching layer above this module; here we only verify
what can be decided from the file alone, deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# extension -> soundfile major format expected for it
EXPECTED_FORMAT = {
    ".wav": "WAV", ".flac": "FLAC", ".aiff": "AIFF", ".aif": "AIFF",
    ".ogg": "OGG", ".mp3": "MPEG",
}
STANDARD_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000,
                  88200, 96000, 176400, 192000}
DURATION_TOLERANCE = 0.01      # 1% header-vs-decode disagreement allowed
SILENCE_FLOOR = 1e-4           # |sample| below this counts as silence
MAX_EDGE_SILENCE_S = 2.0


@dataclass
class MetaReport:
    """Structured result of a single-file metadata–audio consistency check."""

    path: str
    extension: str
    container_format: str
    header_duration_s: float
    decoded_duration_s: float
    sample_rate: int
    channels: int
    leading_silence_s: float
    trailing_silence_s: float
    issues: list[str]
    passed: bool

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _edge_silence(mono: np.ndarray, sr: int) -> tuple[float, float]:
    loud = np.abs(mono) > SILENCE_FLOOR
    if not loud.any():
        return len(mono) / sr, len(mono) / sr
    first, last = int(np.argmax(loud)), len(loud) - int(np.argmax(loud[::-1]))
    return first / sr, (len(mono) - last) / sr


def verify(path: str) -> MetaReport:
    """Check one file's metadata against its decoded signal."""
    info = sf.info(path)
    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)

    ext = Path(path).suffix.lower()
    header_dur = info.frames / info.samplerate
    decoded_dur = len(mono) / sr
    lead, trail = _edge_silence(mono, sr)

    issues: list[str] = []
    expected = EXPECTED_FORMAT.get(ext)
    if expected and info.format != expected:
        issues.append(
            f"container mismatch: extension {ext} but actual format {info.format}"
        )
    if header_dur > 0 and abs(header_dur - decoded_dur) / header_dur > DURATION_TOLERANCE:
        issues.append(
            f"duration mismatch: header {header_dur:.2f}s vs decoded {decoded_dur:.2f}s"
            " — truncated or corrupt"
        )
    if sr not in STANDARD_RATES:
        issues.append(f"nonstandard sample rate: {sr} Hz")
    if data.shape[1] not in (1, 2):
        issues.append(f"unusual channel count: {data.shape[1]}")
    if lead > MAX_EDGE_SILENCE_S:
        issues.append(f"leading silence: {lead:.1f}s")
    if trail > MAX_EDGE_SILENCE_S:
        issues.append(f"trailing silence: {trail:.1f}s")

    return MetaReport(
        path=path,
        extension=ext,
        container_format=info.format,
        header_duration_s=round(header_dur, 3),
        decoded_duration_s=round(decoded_dur, 3),
        sample_rate=sr,
        channels=data.shape[1],
        leading_silence_s=round(lead, 3),
        trailing_silence_s=round(trail, 3),
        issues=issues,
        passed=not issues,
    )
