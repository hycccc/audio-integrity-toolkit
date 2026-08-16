"""Command-line interface: audio-check <files-or-dirs>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import Report, analyze

AUDIO_SUFFIXES = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3", ".m4a"}


def _collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(
                f for f in sorted(p.rglob("*")) if f.suffix.lower() in AUDIO_SUFFIXES
            )
        else:
            files.append(p)
    return files


def _print_human(report: Report) -> None:
    flag = "PASS" if report.lossless else "FAIL"
    print(f"[{flag}] {report.path}")
    print(
        f"  {report.sample_rate} Hz · {report.channels} ch · "
        f"{report.duration_s:.2f}s · {report.subtype}"
    )
    print(
        f"  peak {report.peak_dbfs:.2f} dBFS · rms {report.rms_dbfs:.2f} dBFS · "
        f"crest {report.crest_db:.2f} dB · clipped samples {report.clipped_samples}"
    )
    print(f"  spectral cutoff {report.cutoff_hz:.0f} Hz / Nyquist {report.nyquist_hz:.0f} Hz")
    print(f"  verdict: {report.verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audio-check",
        description="Acceptance-gate integrity checks for audio datasets: "
        "fake-lossless detection, clipping, dynamic range.",
    )
    parser.add_argument("paths", nargs="+", help="audio files or directories")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per file")
    args = parser.parse_args(argv)

    files = _collect(args.paths)
    if not files:
        parser.error("no audio files found")

    failed = 0
    for f in files:
        try:
            report = analyze(str(f))
        except Exception as exc:  # unreadable/corrupt files also fail the gate
            failed += 1
            if args.json:
                print(json.dumps({"path": str(f), "error": str(exc)}))
            else:
                print(f"[FAIL] {f}\n  unreadable: {exc}")
            continue
        if args.json:
            print(json.dumps(report.to_dict()))
        else:
            _print_human(report)
        if not report.lossless:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
