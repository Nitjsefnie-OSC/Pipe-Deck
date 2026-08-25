#!/usr/bin/env python3
"""Persistent regression tests for the Pipe-Deck #407 diagnostic probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock


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
    dc_offset: float = 0.0,
    second_harmonic_amplitude: float = 0.0,
    right_dc_offset: float | None = None,
    right_second_harmonic_amplitude: float | None = None,
    first_frame: int = 0,
    append: bool = False,
) -> None:
    """Write independent stereo-f32 test evidence without probe helpers."""
    if right_amplitude is None:
        right_amplitude = left_amplitude
    if right_dc_offset is None:
        right_dc_offset = dc_offset
    if right_second_harmonic_amplitude is None:
        right_second_harmonic_amplitude = second_harmonic_amplitude
    with path.open("ab" if append else "wb") as output:
        for frame in range(first_frame, frames):
            if active_start <= frame < active_end:
                phase = 2.0 * math.pi * frequency_hz * (frame - active_start) / 48_000
                left = (
                    dc_offset
                    + left_amplitude * math.sin(phase)
                    + second_harmonic_amplitude * math.sin(2.0 * phase)
                )
                right = (
                    right_dc_offset
                    + right_amplitude * math.sin(phase)
                    + right_second_harmonic_amplitude * math.sin(2.0 * phase)
                )
            else:
                left = right = 0.0
            output.write(struct.pack("<ff", left, right))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_frame(path: Path, frame: int, left: float, right: float) -> None:
    with path.open("r+b") as output:
        output.seek(frame * 8)
        output.write(struct.pack("<ff", left, right))


def establish_capture_origins(
    root: Path,
    volume: int,
    *,
    post: bool = False,
    capture_name_suffix: str | None = None,
) -> None:
    suffix = str(volume) if capture_name_suffix is None else capture_name_suffix
    target_capture = f"capture-target-{suffix}"
    monitor_capture = f"capture-monitor-{suffix}"
    links = root / f"pw-link-{volume}{'-post' if post else ''}-lI.txt"
    links.write_text(
        "\n".join(
            [
                f"{target_capture}:input_FL",
                "  |<- target:monitor_FL",
                f"{target_capture}:input_FR",
                "  |<- target:monitor_FR",
                f"{monitor_capture}:input_FL",
                "  |<- monitor:monitor_FL",
                f"{monitor_capture}:input_FR",
                "  |<- monitor:monitor_FR",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    probe.verify_links_command(
        links,
        "target",
        "monitor",
        target_capture,
        monitor_capture,
        root / f"links-{volume}{'-post' if post else ''}.json",
        root / f"links-{volume}{'-post' if post else ''}.txt",
    )


def acquire_capture_pair(
    root: Path,
    volume: int,
    *,
    amplitude: float = 0.25,
    capture_name_suffix: str | None = None,
    seal: bool = True,
) -> None:
    for destination in ("target", "monitor"):
        write_capture(
            root / f"{destination}-{volume}.raw",
            frames=768,
            active_start=12_000,
            active_end=84_000,
            left_amplitude=amplitude,
        )
    establish_capture_origins(
        root,
        volume,
        capture_name_suffix=capture_name_suffix,
    )
    for destination in ("target", "monitor"):
        write_capture(
            root / f"{destination}-{volume}.raw",
            frames=96_000,
            active_start=12_000,
            active_end=84_000,
            left_amplitude=amplitude,
            first_frame=768,
            append=True,
        )
    if seal:
        establish_capture_origins(
            root,
            volume,
            post=True,
            capture_name_suffix=capture_name_suffix,
        )


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

    def test_capture_duration_exact_fenceposts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for detected_frames, generated_frames, expected_valid in (
                (71_991, 71_992, False),
                (71_992, 71_993, True),
                (72_008, 72_009, True),
                (72_009, 72_010, False),
            ):
                with self.subTest(detected_frames=detected_frames):
                    raw = root / f"duration-{detected_frames}.raw"
                    write_capture(
                        raw,
                        frames=max(96_000, 24_000 + generated_frames),
                        active_start=12_000,
                        active_end=12_000 + generated_frames,
                    )
                    result = probe.analyze_capture(raw, 100)
                    self.assertEqual(
                        result["active_frames_before_trim"], detected_frames
                    )
                    self.assertEqual(result["valid"], expected_valid, result)

    def test_capture_outside_silence_exact_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for outside_value, expected_valid in (
                (0.000999, True),
                (0.001001, False),
            ):
                with self.subTest(outside_value=outside_value):
                    raw = root / f"outside-{outside_value}.raw"
                    write_capture(
                        raw,
                        frames=96_000,
                        active_start=12_000,
                        active_end=84_000,
                    )
                    write_frame(raw, 6_000, outside_value, outside_value)
                    result = probe.analyze_capture(raw, 100)
                    self.assertEqual(result["valid"], expected_valid, result)

    def test_capture_leading_and_trailing_silence_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leading = root / "leading-one-short.raw"
            write_capture(
                leading,
                frames=96_000,
                active_start=11_998,
                active_end=83_998,
            )
            leading_result = probe.analyze_capture(leading, 100)
            self.assertEqual(leading_result["leading_silence_frames"], 11_999)
            self.assertFalse(leading_result["valid"], leading_result)

            trailing = root / "trailing-one-short.raw"
            write_capture(
                trailing,
                frames=96_000,
                active_start=12_001,
                active_end=84_001,
            )
            trailing_result = probe.analyze_capture(trailing, 100)
            self.assertEqual(trailing_result["trailing_silence_frames"], 11_999)
            self.assertFalse(trailing_result["valid"], trailing_result)

    def test_capture_rejects_dc_and_second_harmonic_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, extras in (
                ("dc", {"dc_offset": 0.20}),
                ("second-harmonic", {"second_harmonic_amplitude": 0.10}),
            ):
                with self.subTest(contamination=name):
                    raw = root / f"capture-{name}.raw"
                    write_capture(
                        raw,
                        frames=96_000,
                        active_start=12_000,
                        active_end=84_000,
                        **extras,
                    )
                    self.assertFalse(probe.analyze_capture(raw, 100)["valid"])

    def test_capture_purity_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, extras, expected_valid in (
                ("dc-inside", {"dc_offset": 0.0009}, True),
                ("dc-outside", {"dc_offset": 0.0011}, False),
                (
                    "residual-inside",
                    {"second_harmonic_amplitude": 0.0475},
                    True,
                ),
                (
                    "residual-outside",
                    {"second_harmonic_amplitude": 0.0525},
                    False,
                ),
            ):
                with self.subTest(case=name):
                    raw = root / f"purity-{name}.raw"
                    write_capture(
                        raw,
                        frames=96_000,
                        active_start=12_000,
                        active_end=84_000,
                        **extras,
                    )
                    result = probe.analyze_capture(raw, 100)
                    self.assertEqual(result["valid"], expected_valid, result)

    def test_capture_purity_authenticates_each_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, extras in (
                ("right-dc", {"right_dc_offset": 0.20}),
                (
                    "right-second-harmonic",
                    {"right_second_harmonic_amplitude": 0.10},
                ),
            ):
                with self.subTest(contamination=name):
                    raw = root / f"capture-{name}.raw"
                    write_capture(
                        raw,
                        frames=96_000,
                        active_start=12_000,
                        active_end=84_000,
                        **extras,
                    )
                    self.assertFalse(probe.analyze_capture(raw, 100)["valid"])

    def test_marker_rejects_dc_and_second_harmonic_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silence = root / "silence.raw"
            write_capture(
                silence,
                frames=12_000,
                active_start=0,
                active_end=0,
                left_amplitude=0.0,
            )
            for name, extras in (
                ("dc", {"dc_offset": 0.20}),
                ("second-harmonic", {"second_harmonic_amplitude": 0.10}),
            ):
                with self.subTest(contamination=name):
                    marker = root / f"marker-{name}.raw"
                    write_capture(
                        marker,
                        frames=12_000,
                        active_start=600,
                        active_end=11_400,
                        frequency_hz=733.0,
                        **extras,
                    )
                    with self.assertRaises(probe.ProbeError):
                        probe.marker_command(
                            marker,
                            silence,
                            root / f"marker-{name}.json",
                            root / f"marker-{name}.txt",
                            "target",
                        )

    def test_marker_purity_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silence = root / "silence.raw"
            write_capture(
                silence,
                frames=12_000,
                active_start=0,
                active_end=0,
                left_amplitude=0.0,
            )
            for name, extras, expected_valid in (
                ("dc-inside", {"dc_offset": 0.0009}, True),
                ("dc-outside", {"dc_offset": 0.0011}, False),
                (
                    "residual-inside",
                    {"second_harmonic_amplitude": 0.0475},
                    True,
                ),
                (
                    "residual-outside",
                    {"second_harmonic_amplitude": 0.0525},
                    False,
                ),
            ):
                with self.subTest(case=name):
                    marker = root / f"marker-boundary-{name}.raw"
                    write_capture(
                        marker,
                        frames=12_000,
                        active_start=600,
                        active_end=11_400,
                        frequency_hz=733.0,
                        **extras,
                    )
                    action = lambda: probe.marker_command(
                        marker,
                        silence,
                        root / f"marker-boundary-{name}.json",
                        root / f"marker-boundary-{name}.txt",
                        "target",
                    )
                    if expected_valid:
                        action()
                    else:
                        with self.assertRaises(probe.ProbeError):
                            action()

    def test_marker_purity_authenticates_each_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silence = root / "silence.raw"
            write_capture(
                silence,
                frames=12_000,
                active_start=0,
                active_end=0,
                left_amplitude=0.0,
            )
            for name, extras in (
                ("right-dc", {"right_dc_offset": 0.20}),
                (
                    "right-second-harmonic",
                    {"right_second_harmonic_amplitude": 0.10},
                ),
            ):
                with self.subTest(contamination=name):
                    marker = root / f"marker-{name}.raw"
                    write_capture(
                        marker,
                        frames=12_000,
                        active_start=600,
                        active_end=11_400,
                        frequency_hz=733.0,
                        **extras,
                    )
                    with self.assertRaises(probe.ProbeError):
                        probe.marker_command(
                            marker,
                            silence,
                            root / f"marker-{name}.json",
                            root / f"marker-{name}.txt",
                            "target",
                        )


class RatioEvidenceTests(unittest.TestCase):
    def _metric_paths(
        self,
        root: Path,
        *,
        swap_destinations_before_metrics: bool = False,
    ) -> dict[tuple[str, int], Path]:
        paths: dict[tuple[str, int], Path] = {}
        for volume, amplitude in ((100, 0.25), (10, 0.025)):
            acquire_capture_pair(root, volume, amplitude=amplitude)
            if swap_destinations_before_metrics:
                target = root / f"target-{volume}.raw"
                monitor = root / f"monitor-{volume}.raw"
                target_bytes = target.read_bytes()
                monitor_bytes = monitor.read_bytes()
                target.write_bytes(monitor_bytes)
                monitor.write_bytes(target_bytes)
            for destination in ("target", "monitor"):
                raw = root / f"{destination}-{volume}.raw"
                metrics = root / f"{destination}-{volume}-metrics.json"
                probe.metrics_command(
                    raw,
                    metrics,
                    root / f"{destination}-{volume}-metrics.txt",
                    volume,
                )
                paths[(destination, volume)] = metrics
        return paths

    def _ratio(
        self,
        root: Path,
        paths: dict[tuple[str, int], Path],
        *,
        json_path: Path | None = None,
        human_path: Path | None = None,
    ) -> None:
        probe.ratio_command(
            paths[("target", 100)],
            paths[("target", 10)],
            paths[("monitor", 100)],
            paths[("monitor", 10)],
            json_path or root / "volume-ratios.json",
            human_path or root / "volume-ratios.txt",
        )

    def test_metrics_requires_post_carrier_link_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquire_capture_pair(root, 100, seal=False)

            with self.assertRaises(probe.ProbeError):
                probe.metrics_command(
                    root / "target-100.raw",
                    root / "target-100-metrics.json",
                    root / "target-100-metrics.txt",
                    100,
                )

    def test_post_carrier_relink_cannot_seal_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquire_capture_pair(root, 100, seal=False)
            relinked = root / "pw-link-100-post-lI.txt"
            relinked.write_text(
                "\n".join(
                    [
                        "capture-target-100:input_FL",
                        "  |<- monitor:monitor_FL",
                        "capture-target-100:input_FR",
                        "  |<- monitor:monitor_FR",
                        "capture-monitor-100:input_FL",
                        "  |<- target:monitor_FL",
                        "capture-monitor-100:input_FR",
                        "  |<- target:monitor_FR",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(probe.ProbeError):
                probe.verify_links_command(
                    relinked,
                    "target",
                    "monitor",
                    "capture-target-100",
                    "capture-monitor-100",
                    root / "links-100-post.json",
                    root / "links-100-post.txt",
                )
            manifest = json.loads(
                (root / "capture-origin-100.json").read_text(encoding="utf-8")
            )
            self.assertIsNot(manifest.get("sealed"), True)

    def test_metrics_requires_capture_time_origin_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "target-100.raw"
            write_capture(
                raw,
                frames=96_000,
                active_start=12_000,
                active_end=84_000,
            )

            with self.assertRaises(probe.ProbeError):
                probe.metrics_command(
                    raw,
                    root / "target-100-metrics.json",
                    root / "target-100-metrics.txt",
                    100,
                )

    def test_ratio_rejects_destination_origin_swapped_before_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaises(probe.ProbeError):
                paths = self._metric_paths(
                    root,
                    swap_destinations_before_metrics=True,
                )
                self._ratio(root, paths)

    def test_metrics_rejects_wrong_volume_origin_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for volume in (100, 10):
                for destination in ("target", "monitor"):
                    write_capture(
                        root / f"{destination}-{volume}.raw",
                        frames=96_000,
                        active_start=12_000,
                        active_end=84_000,
                        left_amplitude=0.25,
                    )
                establish_capture_origins(root, volume)
            target_100 = root / "target-100.raw"
            target_100.write_bytes((root / "target-10.raw").read_bytes())

            with self.assertRaises(probe.ProbeError):
                probe.metrics_command(
                    target_100,
                    root / "target-100-metrics.json",
                    root / "target-100-metrics.txt",
                    100,
                )

    def test_ratio_normal_outputs_preserve_all_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            evidence = [
                *paths.values(),
                *(root / f"{destination}-{volume}.raw"
                  for destination in ("target", "monitor")
                  for volume in (100, 10)),
            ]
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence}

            self._ratio(root, paths)

            after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence}
            self.assertEqual(after, before)
            ratio = json.loads((root / "volume-ratios.json").read_text(encoding="utf-8"))
            self.assertEqual(ratio["overall_classification"], "conforming-scaled")
            self.assertIn("overall_classification: conforming-scaled", (root / "volume-ratios.txt").read_text(encoding="utf-8"))
            self.assertTrue((root / "volume-ratios.commit.json").is_file())
            commit = json.loads(
                (root / "volume-ratios.commit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(commit["publication_id"], ratio["publication_id"])

    def test_ratio_rejects_outputs_in_another_evidence_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source_root = parent / "source"
            victim_root = parent / "victim"
            source_root.mkdir()
            victim_root.mkdir()
            source_paths = self._metric_paths(source_root)
            self._metric_paths(victim_root)
            victim_raw = victim_root / "target-100.raw"
            before = hashlib.sha256(victim_raw.read_bytes()).hexdigest()

            with self.assertRaises(probe.ProbeError):
                self._ratio(
                    source_root,
                    source_paths,
                    json_path=victim_raw,
                    human_path=source_root / "volume-ratios.txt",
                )
            self.assertEqual(hashlib.sha256(victim_raw.read_bytes()).hexdigest(), before)

    def test_ratio_rejects_fresh_outputs_outside_evidence_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source_root = parent / "source"
            foreign_root = parent / "foreign"
            source_root.mkdir()
            foreign_root.mkdir()
            paths = self._metric_paths(source_root)

            with self.assertRaises(probe.ProbeError):
                self._ratio(
                    source_root,
                    paths,
                    json_path=foreign_root / "volume-ratios.json",
                    human_path=foreign_root / "volume-ratios.txt",
                )
            self.assertFalse((foreign_root / "volume-ratios.json").exists())
            self.assertFalse((foreign_root / "volume-ratios.txt").exists())
            self.assertFalse((foreign_root / "volume-ratios.commit.json").exists())

    def test_ratio_failed_second_leaf_publish_has_no_mixed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            json_path = root / "volume-ratios.json"
            human_path = root / "volume-ratios.txt"
            json_path.write_text("old-json\n", encoding="utf-8")
            human_path.write_text("old-human\n", encoding="utf-8")
            original_replace = os.replace

            def fail_second_leaf(source: object, destination: object) -> None:
                if Path(destination) == human_path:
                    raise OSError("injected second-leaf failure")
                original_replace(source, destination)

            with mock.patch.object(
                probe.os, "replace", side_effect=fail_second_leaf
            ), self.assertRaises((probe.ProbeError, OSError)):
                self._ratio(root, paths, json_path=json_path, human_path=human_path)

            self.assertEqual(json_path.read_text(encoding="utf-8"), "old-json\n")
            self.assertEqual(human_path.read_text(encoding="utf-8"), "old-human\n")
            self.assertFalse((root / "volume-ratios.commit.json").exists())

    def test_ratio_second_leaf_failure_has_no_committed_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            original_publish = probe._publish_staged_no_clobber
            publication_count = 0

            def fail_second_leaf(source: Path, destination: Path) -> None:
                nonlocal publication_count
                publication_count += 1
                if publication_count == 2:
                    raise probe.ProbeError("injected second-leaf failure")
                original_publish(source, destination)

            with mock.patch.object(
                probe,
                "_publish_staged_no_clobber",
                side_effect=fail_second_leaf,
            ), self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)

            self.assertTrue((root / "volume-ratios.json").is_file())
            self.assertFalse((root / "volume-ratios.txt").exists())
            self.assertFalse((root / "volume-ratios.commit.json").exists())

    def test_ratio_rehashes_evidence_after_both_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            raw = root / "target-100.raw"
            original_publish = probe._publish_staged_no_clobber

            def publish_and_mutate(source: Path, destination: Path) -> None:
                original_publish(source, destination)
                if Path(destination).name == "volume-ratios.txt":
                    with raw.open("ab") as output:
                        output.write(struct.pack("<ff", 0.0, 0.0))

            with mock.patch.object(
                probe,
                "_publish_staged_no_clobber",
                side_effect=publish_and_mutate,
            ), self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)
            self.assertFalse((root / "volume-ratios.commit.json").exists())

    def test_ratio_rejects_output_path_replacement_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            original_stage = probe._stage_output
            staged = 0

            def stage_and_replace(path: Path, data: bytes) -> Path:
                nonlocal staged
                temporary = original_stage(path, data)
                staged += 1
                if staged == 2:
                    (root / "volume-ratios.txt").write_text("raced\n", encoding="utf-8")
                return temporary

            with mock.patch.object(
                probe, "_stage_output", side_effect=stage_and_replace
            ), self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)

    def test_ratio_rejects_direct_output_aliases(self) -> None:
        for scenario in ("metrics", "raw", "outputs-each-other"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self._metric_paths(root)
                json_path = root / "volume-ratios.json"
                human_path = root / "volume-ratios.txt"
                if scenario == "metrics":
                    json_path = paths[("target", 100)]
                elif scenario == "raw":
                    human_path = root / "target-100.raw"
                else:
                    json_path = human_path = root / "same-output"

                evidence = [
                    *paths.values(),
                    *(root / f"{destination}-{volume}.raw"
                      for destination in ("target", "monitor")
                      for volume in (100, 10)),
                ]
                before = {
                    path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in evidence
                }

                with self.assertRaises(probe.ProbeError):
                    self._ratio(
                        root,
                        paths,
                        json_path=json_path,
                        human_path=human_path,
                    )
                after = {
                    path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in evidence
                }
                self.assertEqual(after, before)

    def test_ratio_rejects_symlink_output_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            human_alias = root / "volume-ratios.txt"
            human_alias.symlink_to(root / "target-100.raw")

            with self.assertRaises(probe.ProbeError):
                self._ratio(root, paths, human_path=human_alias)

    def test_ratio_rejects_hardlink_output_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            human_alias = root / "volume-ratios.txt"
            os.link(root / "target-100.raw", human_alias)

            with self.assertRaises(probe.ProbeError):
                self._ratio(root, paths, human_path=human_alias)

    def test_ratio_rejects_stale_or_missing_analysis_digest(self) -> None:
        for scenario in ("stale", "missing"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self._metric_paths(root)
                metrics = paths[("target", 100)]
                record = json.loads(metrics.read_text(encoding="utf-8"))
                if scenario == "stale":
                    record["authenticated_analysis_sha256"] = "0" * 64
                else:
                    record.pop("authenticated_analysis_sha256")
                write_json(metrics, record)

                with self.assertRaises(probe.ProbeError):
                    self._ratio(root, paths)

    def test_ratio_rejects_stale_stored_analysis_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            metrics = paths[("target", 100)]
            record = json.loads(metrics.read_text(encoding="utf-8"))
            record["rms"] = float(record["rms"]) * 2.0
            write_json(metrics, record)

            with self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)

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

    def test_ratio_rejects_stable_raw_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._metric_paths(root)
            raw = root / "target-100.raw"
            replacement = root / "replacement.raw"
            replacement.write_bytes(raw.read_bytes())
            os.replace(replacement, raw)

            with self.assertRaises(probe.ProbeError):
                self._ratio(root, paths)

    def test_ratio_rejects_changed_leg_and_schema_fields(self) -> None:
        for field, value in (
            ("expected_destination", "monitor"),
            ("volume_percent", 10),
            ("evidence_schema", "changed-schema"),
            ("capture_origin_schema", "changed-origin-schema"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self._metric_paths(root)
                metrics = paths[("target", 100)]
                record = json.loads(metrics.read_text(encoding="utf-8"))
                record[field] = value
                write_json(metrics, record)

                with self.assertRaises(probe.ProbeError):
                    self._ratio(root, paths)

    def test_metrics_rejects_changed_pre_or_post_link_evidence(self) -> None:
        for phase in ("pre", "post"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self._metric_paths(root)
                link_path = root / (
                    "pw-link-100-lI.txt"
                    if phase == "pre"
                    else "pw-link-100-post-lI.txt"
                )
                with link_path.open("ab") as output:
                    output.write(b"\n")

                with self.assertRaises(probe.ProbeError):
                    probe.metrics_command(
                        root / "target-100.raw",
                        root / "target-100-metrics.json",
                        root / "target-100-metrics.txt",
                        100,
                    )

    def test_capture_origin_pilot_authenticates_volume_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for volume in (100, 10):
                acquire_capture_pair(
                    root, volume, capture_name_suffix="shared"
                )

            target_100 = root / "target-100.raw"
            target_100.write_bytes((root / "target-10.raw").read_bytes())
            with self.assertRaises(probe.ProbeError):
                probe.metrics_command(
                    target_100,
                    root / "target-100-metrics.json",
                    root / "target-100-metrics.txt",
                    100,
                )

    def test_capture_origin_rejects_renamed_foreign_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for generation in ("first", "second"):
                acquire_capture_pair(root, 100, capture_name_suffix="shared")
                if generation == "first":
                    for destination in ("target", "monitor"):
                        os.replace(
                            root / f"{destination}-100.raw",
                            root / f"saved-{destination}-100.raw",
                        )

            for destination in ("target", "monitor"):
                os.replace(
                    root / f"saved-{destination}-100.raw",
                    root / f"{destination}-100.raw",
                )
            with self.assertRaises(probe.ProbeError):
                probe.metrics_command(
                    root / "target-100.raw",
                    root / "target-100-metrics.json",
                    root / "target-100-metrics.txt",
                    100,
                )

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
