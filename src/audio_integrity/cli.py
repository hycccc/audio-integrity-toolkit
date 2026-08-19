"""Command-line interface.

    audio-check <files-or-dirs>          # spectral integrity gate (default)
    audio-check gate <files-or-dirs>     # same, spelled out
    audio-check dedup <files-or-dirs>    # exact + near duplicate clusters
    audio-check meta <files-or-dirs>     # metadata–audio consistency

The bare form stays the spectral gate so existing pipelines keep working.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import Report, analyze
from .dedup import DEFAULT_THRESHOLD, find_duplicates
from .metadata import verify

AUDIO_SUFFIXES = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3", ".m4a"}
SUBCOMMANDS = {"gate", "dedup", "meta"}


def _collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(
                f for f in sorted(p.rglob("*")) if f.suffix.lower() in AUDIO_SUFFIXES
            )
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
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


def _run_gate(files: list[Path], as_json: bool) -> int:
    failed = 0
    for f in files:
        try:
            report = analyze(str(f))
        except Exception as exc:  # unreadable/corrupt files also fail the gate
            failed += 1
            if as_json:
                print(json.dumps({"path": str(f), "error": str(exc)}))
            else:
                print(f"[FAIL] {f}\n  unreadable: {exc}")
            continue
        if as_json:
            print(json.dumps(report.to_dict()))
        else:
            _print_human(report)
        if not report.lossless:
            failed += 1
    return 1 if failed else 0


def _run_dedup(files: list[Path], as_json: bool, threshold: float, strategy: str) -> int:
    clusters = find_duplicates([str(f) for f in files], threshold=threshold,
                               strategy=strategy)
    if as_json:
        for c in clusters:
            print(json.dumps(c.to_dict()))
    elif not clusters:
        print(f"no duplicates among {len(files)} files (threshold {threshold})")
    else:
        for c in clusters:
            label = "exact" if c.kind == "exact" else f"near {c.similarity:.0%}"
            print(f"[DUP {label}]")
            for p in c.paths:
                print(f"  {p}")
    return 1 if clusters else 0


def _run_meta(files: list[Path], as_json: bool) -> int:
    failed = 0
    for f in files:
        try:
            report = verify(str(f))
        except Exception as exc:
            failed += 1
            if as_json:
                print(json.dumps({"path": str(f), "error": str(exc)}))
            else:
                print(f"[FAIL] {f}\n  unreadable: {exc}")
            continue
        if as_json:
            print(json.dumps(report.to_dict()))
        else:
            flag = "PASS" if report.passed else "FAIL"
            print(f"[{flag}] {report.path}")
            print(
                f"  {report.container_format} ({report.extension}) · "
                f"{report.sample_rate} Hz · {report.channels} ch · "
                f"{report.decoded_duration_s:.2f}s"
            )
            for issue in report.issues:
                print(f"  - {issue}")
        if not report.passed:
            failed += 1
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = "gate"
    if argv and argv[0] in SUBCOMMANDS:
        cmd = argv.pop(0)

    parser = argparse.ArgumentParser(
        prog=f"audio-check {cmd}" if cmd != "gate" else "audio-check",
        description="Acceptance-gate QC for audio datasets: fake-lossless "
        "detection (gate), duplicate clusters (dedup), metadata–audio "
        "consistency (meta).",
    )
    parser.add_argument("paths", nargs="+", help="audio files or directories")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per line")
    if cmd == "dedup":
        parser.add_argument(
            "--threshold", type=float, default=DEFAULT_THRESHOLD,
            help=f"near-duplicate bit-agreement threshold (default {DEFAULT_THRESHOLD})",
        )
        parser.add_argument(
            "--strategy", choices=["auto", "pairwise", "indexed"], default="auto",
            help="near-dup comparison strategy: pairwise = every pair (exact), "
                 "indexed = inverted-index candidates first (library scale), "
                 "auto = indexed past 64 files (default)",
        )
    args = parser.parse_args(argv)

    try:
        files = _collect(args.paths)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not files:
        parser.error("no audio files found")

    if cmd == "dedup":
        return _run_dedup(files, args.json, args.threshold, args.strategy)
    if cmd == "meta":
        return _run_meta(files, args.json)
    return _run_gate(files, args.json)


if __name__ == "__main__":
    sys.exit(main())
