#!/usr/bin/env python3
"""Persistent regression tests for the Pipe-Deck #407 diagnostic probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest


PROBE_PATH = Path(__file__).with_name("probe.py")
SPEC = importlib.util.spec_from_file_location("pipe_deck_407_probe", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def write_capture(
    path: Path,
    *,
    frames: int,
    active_start: int,
    active_end: int,
    frequency_hz: float = 997.0,
    left_amplitude: float = 0.25,
    right_amplitude: float | None = None,
) -> None:
    """Write independent stereo-f32 test evidence without probe helpers."""
    if right_amplitude is None:
        right_amplitude = left_amplitude
    with path.open("wb") as output:
        for frame in range(frames):
            if active_start <= frame < active_end:
                phase = 2.0 * math.pi * frequency_hz * (frame - active_start) / 48_000
                left = left_amplitude * math.sin(phase)
                right = right_amplitude * math.sin(phase)
            else:
                left = right = 0.0
            output.write(struct.pack("<ff", left, right))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class ProbeAuthenticationTests(unittest.TestCase):
    def test_capture_rejects_numerical_leakage_in_either_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "target-100.raw"
            write_capture(
                raw,
                frames=96_000,
                active_start=12_000,
                active_end=84_000,
                right_amplitude=1e-12,
            )

            result = probe.analyze_capture(raw, 100)

            self.assertFalse(result["valid"], result)

    def test_marker_rejects_numerical_leakage_in_either_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "marker.raw"
            silence = root / "silence.raw"
            write_capture(
                marker,
                frames=12_000,
                active_start=600,
                active_end=11_400,
                frequency_hz=733.0,
                right_amplitude=1e-12,
            )
            write_capture(
                silence,
                frames=12_000,
                active_start=0,
                active_end=0,
                left_amplitude=0.0,
            )

            with self.assertRaises(probe.ProbeError):
                probe.marker_command(
                    marker,
                    silence,
                    root / "marker.json",
                    root / "marker.txt",
                    "target",
                )

    def test_fixture_active_duration_is_bounded_around_generated_extent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shifted = root / "shifted.raw"
            write_capture(
                shifted,
                frames=320_000,
                active_start=30_000,
                active_end=102_000,
            )
            self.assertTrue(probe.analyze_capture(shifted, 100)["valid"])

            for name, active_frames in (
                ("shortened", 67_201),
                ("overlong-100000", 100_000),
                ("overlong-200000", 200_000),
            ):
                with self.subTest(active_frames=active_frames):
                    raw = root / f"{name}.raw"
                    total_frames = max(96_000, 12_000 + active_frames + 12_000)
                    write_capture(
                        raw,
                        frames=total_frames,
                        active_start=12_000,
                        active_end=12_000 + active_frames,
                    )
                    self.assertFalse(probe.analyze_capture(raw, 100)["valid"])

    def test_capture_gap_limit_bridges_four_frames_but_splits_five(self) -> None:
        four_frame_gap = [1.0] * 10 + [0.0] * 4 + [1.0] * 10
        five_frame_gap = [1.0] * 10 + [0.0] * 5 + [1.0] * 10

        self.assertEqual(probe._active_runs(four_frame_gap, 0.5, 4), [(0, 24)])
        self.assertEqual(
            probe._active_runs(five_frame_gap, 0.5, 4),
            [(0, 10), (15, 25)],
        )


class RatioEvidenceTests(unittest.TestCase):
    def _metric_paths(self, root: Path) -> dict[tuple[str, int], Path]:
        paths: dict[tuple[str, int], Path] = {}
        for destination in ("target", "monitor"):
            for volume, amplitude in ((100, 0.25), (10, 0.025)):
                raw = root / f"{destination}-{volume}.raw"
                metrics = root / f"{destination}-{volume}-metrics.json"
                write_capture(
                    raw,
                    frames=96_000,
                    active_start=12_000,
                    active_end=84_000,
                    left_amplitude=amplitude,
                )
                probe.metrics_command(
                    raw,
                    metrics,
                    root / f"{destination}-{volume}-metrics.txt",
                    volume,
                )
                paths[(destination, volume)] = metrics
        return paths

    def _ratio(self, root: Path, paths: dict[tuple[str, int], Path]) -> None:
        probe.ratio_command(
            paths[("target", 100)],
            paths[("target", 10)],
            paths[("monitor", 100)],
            paths[("monitor", 10)],
            root / "ratios.json",
            root / "ratios.txt",
        )

    def test_ratio_rejects_forged_valid_booleans_without_raw_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: dict[tuple[str, int], Path] = {}
            for destination in ("target", "monitor"):
                for volume, value in ((100, 1.0), (10, 0.1)):
                    path = root / f"{destination}-{volume}-metrics.json"
                    write_json(
                        path,
                        {
                            "valid": True,
                            "authenticated_fixture_run_count": 0,
                            "channel_frequency_hz": [0.0, 0.0],
                            "rms": value,
                            "peak": value,
                        },
                    )
                    paths[(destination, volume)] = path

            with self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)

    def test_ratio_rejects_altered_or_missing_bound_raw_capture(self) -> None:
        for scenario in ("altered", "missing"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self._metric_paths(root)
                raw = root / "target-100.raw"
                if scenario == "altered":
                    with raw.open("ab") as output:
                        output.write(struct.pack("<ff", 0.0, 0.0))
                else:
                    raw.unlink()

                with self.assertRaises(probe.ProbeError):
                    self._ratio(root, paths)

    def test_ratio_rejects_metrics_bound_to_the_wrong_leg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            paths[("target", 100)], paths[("monitor", 100)] = (
                paths[("monitor", 100)],
                paths[("target", 100)],
            )

            with self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)

    def test_ratio_rejects_unauthenticated_raw_even_with_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: dict[tuple[str, int], Path] = {}
            for destination in ("target", "monitor"):
                for volume, value in ((100, 1.0), (10, 0.1)):
                    raw = root / f"{destination}-{volume}.raw"
                    write_capture(
                        raw,
                        frames=96_000,
                        active_start=12_000,
                        active_end=84_000,
                        frequency_hz=733.0,
                    )
                    path = root / f"{destination}-{volume}-metrics.json"
                    write_json(
                        path,
                        {
                            "valid": True,
                            "authenticated_fixture_run_count": 1,
                            "channel_frequency_hz": [997.0, 997.0],
                            "rms": value,
                            "peak": value,
                            "capture": str(raw.resolve()),
                            "capture_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                            "expected_destination": destination,
                            "volume_percent": volume,
                        },
                    )
                    paths[(destination, volume)] = path

            with self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)


if __name__ == "__main__":
    unittest.main()
