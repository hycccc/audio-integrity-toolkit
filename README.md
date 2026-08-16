# audio-integrity-toolkit

[![CI](https://github.com/hycccc/audio-integrity-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/hycccc/audio-integrity-toolkit/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-1fa88c.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-cc7434.svg)](pyproject.toml)

Acceptance-gate QC for audio datasets: **fake-lossless detection**, clipping, and dynamic-range checks.

![Genuine lossless vs MP3-transcode spectrum](docs/spectrum-comparison.png)

When you source audio at scale, a meaningful share of "lossless" deliveries are lossy transcodes re-wrapped in a FLAC/WAV container. The container lies; the spectrum doesn't. This tool implements the acceptance-gate pattern I use for music-dataset ingestion: cheap enough to run on every file, strict enough to catch the most common vendor-delivery defects before they poison a training set.

## How it works

1. **Spectral cutoff analysis** — compute the Welch PSD of the (downmixed) signal and locate the highest frequency with energy above a relative −80 dB floor. Lossy encoders apply characteristic low-pass filters, so a "FLAC" whose spectrum stops near 16 kHz was almost certainly an MP3 128 kbps once:

   | Observed cutoff | Verdict |
   |---|---|
   | < ~16.5 kHz | suspected MP3 128 kbps transcode |
   | < ~19.5 kHz | suspected MP3 192 kbps transcode |
   | < ~20.5 kHz | suspected MP3 320 kbps / AAC transcode |
   | within 1.5 kHz of Nyquist | genuinely full-band |

2. **Clipping detection** — count samples at ≥ 0.999 full scale.
3. **Dynamic-range estimate** — peak, RMS, and crest factor (peak − RMS), a quick proxy for over-compressed masters.

## Install

```bash
pip install git+https://github.com/hycccc/audio-integrity-toolkit
```

## Usage

```bash
# single file, human-readable
audio-check song.flac

# whole delivery batch, one JSON object per file (pipe into your pipeline)
audio-check /data/vendor_batch_07/ --json > batch_07_qc.jsonl
```

Exit code is non-zero if any file fails the gate, so it drops straight into CI or an ingestion pipeline.

Example output:

```
[FAIL] song.flac
  44100 Hz · 2 ch · 213.40s · PCM_16
  peak -0.10 dBFS · rms -9.83 dBFS · crest 9.73 dB · clipped samples 412
  spectral cutoff 16021 Hz / Nyquist 22050 Hz
  verdict: suspected MP3 128kbps transcode
```

As a library:

```python
from audio_integrity import analyze

report = analyze("song.flac")
if not report.lossless:
    quarantine(report.path, reason=report.verdict)
```

## Limitations — use as a guardrail, not as gold

Spectral cutoff is a heuristic. Some genuine recordings are naturally band-limited (dark analog masters, archival material), and some modern encoders keep more highs than their bitrate suggests. In production I treat this check the way I treat every automated audio metric: **a guardrail that catches obvious defects at scale, with human spot-checks as the gold standard.** Flag, quarantine, and sample-listen — don't silently delete.

## License

MIT
