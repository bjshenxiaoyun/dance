#!/usr/bin/env python3
"""Compare teacher vs student dance videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare teacher and student dance videos")
    parser.add_argument("--teacher", required=True, help="Teacher video path")
    parser.add_argument("--student", required=True, help="Student video path")
    parser.add_argument("--out", default=str(ROOT / "output" / "run"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--no-align", action="store_true", help="Disable temporal alignment")
    parser.add_argument("--beats", action="store_true", default=True, help="Snapshot on musical downbeats (default on)")
    parser.add_argument("--no-beats", action="store_true", help="Disable beat-based snapshots")
    parser.add_argument("--meter", type=int, default=4, help="Beats per bar for downbeat estimate (default 4)")
    parser.add_argument("--max-downbeats", type=int, default=20, help="Max downbeat screenshots")
    args = parser.parse_args()

    out_dir = Path(args.out)

    print("Extracting teacher poses...")
    print("Extracting student poses...")
    print("Comparing...")
    print("Writing comparison video (original + pose)...")
    print("Saving key-moment snapshots...")
    if not args.no_beats:
        print("Detecting musical downbeats from teacher audio...")

    summary = run_comparison(
        args.teacher,
        args.student,
        out_dir,
        max_frames=args.max_frames,
        align=not args.no_align,
        use_beats=not args.no_beats,
        meter=args.meter,
        max_downbeats=args.max_downbeats,
    )

    beat_track = summary["beat_track"]
    if beat_track:
        print(
            f"  tempo≈{beat_track['tempo']:.1f} BPM, "
            f"beats={len(beat_track['beat_times'])}, downbeats={len(beat_track['downbeat_times'])}"
        )

    print(f"\nScore: {summary['overall_score']:.1f}/100 | lag={summary['lag_frames']} frames")
    if summary["key_moments"]:
        print("\n关键时间点:")
        for m in summary["key_moments"]:
            print(
                f"  [{m['time_sec']:5.1f}s] {m['joint_cn']}: "
                f"老师 {m['teacher_deg']:.0f}° / 你 {m['student_deg']:.0f}° ({m['diff_deg']:+.0f}°)"
            )

    print(f"\n关键截图: {len(summary['outputs']['moments'])} 张 -> {out_dir / 'moments'}")
    if summary["outputs"]["beats"]:
        print(f"重拍截图: {len(summary['outputs']['beats'])} 张 -> {out_dir / 'beats'}")
        for m in summary["beat_moments"][:8]:
            print(
                f"  [{m['time_sec']:5.2f}s] {m['joint_cn']}: "
                f"老师 {m['teacher_deg']:.0f}° / 你 {m['student_deg']:.0f}° ({m['diff_deg']:+.0f}°)"
            )

    for tip in summary["suggestions"]:
        print(f"- {tip}")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
