#!/usr/bin/env python3
"""Dependency-free helpers for the disposable Pipe-Deck #407 probe.

This file is intentionally diagnostic-only.  It is called by the ignored Rust
test and by the later runner workflow; it never changes Pipe Deck behaviour.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
import re
import struct
import tempfile
import wave
from pathlib import Path


SAMPLE_RATE = 48_000
STEREO = 2
FIXTURE_FRAMES = 96_000
ACTIVE_START = 12_000
ACTIVE_END = 84_000
FIXTURE_TRAILING_SILENCE_FRAMES = FIXTURE_FRAMES - ACTIVE_END
TRIM_FRAMES = 4_800
MIN_ACTIVE_FRAMES = 57_600
ACTIVE_THRESHOLD = 0.002
OUTSIDE_THRESHOLD = 1e-3
CAPTURE_FREQUENCY_HZ = 997.0
CAPTURE_FREQUENCY_TOLERANCE_HZ = 10.0
CAPTURE_FREQUENCY_SCORE_MIN = 0.80
CAPTURE_MAX_SILENT_GAP_FRAMES = 4
MIN_CAPTURE_RUN_FRAMES = MIN_ACTIVE_FRAMES + (2 * TRIM_FRAMES)
MARKER_FREQUENCIES = {"target": 733.0, "monitor": 1237.0}
MARKER_MIN_ACTIVE_FRAMES = SAMPLE_RATE // 10
MARKER_MAX_SILENT_GAP_FRAMES = 4
MARKER_FREQUENCY_TOLERANCE_HZ = 10.0
MARKER_FREQUENCY_SCORE_MIN = 0.50


class ProbeError(RuntimeError):
    """A failed diagnostic assertion or invalid input."""


def _frequency_within_tolerance(measured: float, expected: float, tolerance: float) -> bool:
    return abs(measured - expected) <= tolerance + 1e-6


def _expect_probe_error(action: Callable[[], object], description: str) -> None:
    try:
        action()
    except ProbeError:
        return
    raise ProbeError(f"{description} unexpectedly passed")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _pcm16_sample(frame: int, frequency_hz: float, amplitude: float, active_start: int, active_end: int) -> int:
    if not active_start <= frame < active_end:
        return 0
    value = amplitude * math.sin(2.0 * math.pi * frequency_hz * (frame - active_start) / SAMPLE_RATE)
    return max(-32768, min(32767, int(round(value * 32767.0))))


def _write_wav(path: Path, frames: int, frequency_hz: float, amplitude: float, active_start: int, active_end: int) -> dict[str, object]:
    raw = bytearray()
    for frame in range(frames):
        sample = _pcm16_sample(frame, frequency_hz, amplitude, active_start, active_end)
        raw.extend(struct.pack("<hh", sample, sample))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(STEREO)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(raw)

    return {
        "sample_rate": SAMPLE_RATE,
        "channels": STEREO,
        "sample_format": "pcm_s16le",
        "frames": frames,
        "duration_seconds": frames / SAMPLE_RATE,
        "frequency_hz": frequency_hz,
        "amplitude": amplitude,
        "active_start_frame": active_start,
        "active_end_frame_exclusive": active_end,
        "silence_before_frames": active_start,
        "silence_after_frames": frames - active_end,
    }


def make_fixture(path: Path, metadata_path: Path) -> None:
    metadata = _write_wav(path, FIXTURE_FRAMES, 997.0, 0.25, ACTIVE_START, ACTIVE_END)
    metadata["kind"] = "soundboard-volume-fixture"
    _write_json(metadata_path, metadata)


def make_marker(path: Path, metadata_path: Path, frequency_hz: float) -> None:
    frames = SAMPLE_RATE // 2
    active_start = SAMPLE_RATE // 20
    active_end = frames - active_start
    metadata = _write_wav(path, frames, frequency_hz, 0.25, active_start, active_end)
    metadata["kind"] = "routing-marker"
    _write_json(metadata_path, metadata)


def _read_f32_stereo(path: Path) -> list[tuple[float, float]]:
    data = path.read_bytes()
    if not data:
        raise ProbeError(f"capture is empty: {path}")
    if len(data) % 8:
        raise ProbeError(f"capture is truncated to a non-stereo-f32 frame: {path}")
    return list(struct.iter_unpack("<ff", data))


def _frame_peaks(frames: list[tuple[float, float]]) -> list[float]:
    return [max(abs(left), abs(right)) for left, right in frames]


def _active_runs(
    peaks: list[float], threshold: float, max_silent_gap_frames: int
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    current_start: int | None = None
    last_active: int | None = None
    for index, peak in enumerate(peaks):
        if peak > threshold:
            if current_start is None:
                current_start = index
            last_active = index
            continue
        if current_start is not None and last_active is not None:
            if index - last_active - 1 > max_silent_gap_frames:
                runs.append((current_start, last_active + 1))
                current_start = None
                last_active = None
    if current_start is not None and last_active is not None:
        runs.append((current_start, last_active + 1))
    return runs


def _capture_run_frequency_metrics(
    frames: list[tuple[float, float]], start: int, end: int
) -> tuple[list[float], list[float]]:
    trimmed_start = start + TRIM_FRAMES
    trimmed_end = end - TRIM_FRAMES
    if trimmed_end <= trimmed_start:
        return [0.0] * STEREO, [0.0] * STEREO
    frequencies = [
        _estimate_frequency(frames, trimmed_start, trimmed_end, channel)
        for channel in range(STEREO)
    ]
    scores = [
        _frequency_identity_score(
            frames,
            trimmed_start,
            trimmed_end,
            frequency,
            channel,
        )
        for channel, frequency in enumerate(frequencies)
    ]
    return frequencies, scores


def analyze_capture(path: Path, volume_percent: int) -> dict[str, object]:
    frames = _read_f32_stereo(path)
    peaks = _frame_peaks(frames)
    active_runs = _active_runs(peaks, ACTIVE_THRESHOLD, CAPTURE_MAX_SILENT_GAP_FRAMES)
    if not active_runs:
        raise ProbeError(f"capture has no active frames above {ACTIVE_THRESHOLD}: {path}")

    eligible_runs = [
        (start, end)
        for start, end in active_runs
        if end - start >= MIN_CAPTURE_RUN_FRAMES
    ]
    authenticated_runs: list[tuple[int, int, list[float], list[float]]] = []
    for start, end in eligible_runs:
        frequencies, scores = _capture_run_frequency_metrics(frames, start, end)
        if all(
            _frequency_within_tolerance(
                frequency, CAPTURE_FREQUENCY_HZ, CAPTURE_FREQUENCY_TOLERANCE_HZ
            )
            and score >= CAPTURE_FREQUENCY_SCORE_MIN
            for frequency, score in zip(frequencies, scores)
        ):
            authenticated_runs.append((start, end, frequencies, scores))

    if len(authenticated_runs) == 1:
        active_start, active_end_exclusive, channel_frequencies, channel_scores = authenticated_runs[0]
    else:
        active_start, active_end_exclusive = max(
            eligible_runs or active_runs,
            key=lambda run: run[1] - run[0],
        )
        channel_frequencies, channel_scores = _capture_run_frequency_metrics(
            frames, active_start, active_end_exclusive
        )

    run_frames = active_end_exclusive - active_start
    trimmed_start = active_start + TRIM_FRAMES
    trimmed_end_exclusive = active_end_exclusive - TRIM_FRAMES
    if trimmed_end_exclusive > trimmed_start:
        trimmed_samples = frames[trimmed_start:trimmed_end_exclusive]
    else:
        trimmed_samples = []
    active_frames = max(0, trimmed_end_exclusive - trimmed_start)
    sum_squares = sum(sample * sample for pair in trimmed_samples for sample in pair)
    rms = math.sqrt(sum_squares / (len(trimmed_samples) * STEREO)) if trimmed_samples else 0.0
    peak = max(_frame_peaks(trimmed_samples), default=0.0)
    outside_values = peaks[:active_start] + peaks[active_end_exclusive:]
    outside_peak = max(outside_values, default=0.0)
    trailing_silence_peak = max(peaks[-FIXTURE_TRAILING_SILENCE_FRAMES:], default=0.0)
    leading_silence_frames = active_start
    trailing_silence_frames = len(frames) - active_end_exclusive

    minimum_rms = 0.02 if volume_percent == 100 else 0.002
    minimum_peak = 0.05 if volume_percent == 100 else 0.005
    violations: list[str] = []
    if len(frames) < FIXTURE_FRAMES:
        violations.append(f"frames {len(frames)} < generated fixture extent {FIXTURE_FRAMES}")
    if trailing_silence_peak >= OUTSIDE_THRESHOLD:
        violations.append(
            f"trailing_silence_peak {trailing_silence_peak:.9f} >= {OUTSIDE_THRESHOLD}"
        )
    if len(authenticated_runs) != 1:
        violations.append(
            f"authenticated_fixture_runs {len(authenticated_runs)} != 1"
        )
    if run_frames < MIN_CAPTURE_RUN_FRAMES:
        violations.append(
            f"contiguous_fixture_run_frames {run_frames} < {MIN_CAPTURE_RUN_FRAMES}"
        )
    if leading_silence_frames < ACTIVE_START:
        violations.append(
            f"leading_silence_frames {leading_silence_frames} < {ACTIVE_START}"
        )
    if trailing_silence_frames < FIXTURE_TRAILING_SILENCE_FRAMES:
        violations.append(
            f"trailing_silence_frames {trailing_silence_frames} < {FIXTURE_TRAILING_SILENCE_FRAMES}"
        )
    for channel, (frequency, score) in enumerate(zip(channel_frequencies, channel_scores)):
        if not _frequency_within_tolerance(
            frequency, CAPTURE_FREQUENCY_HZ, CAPTURE_FREQUENCY_TOLERANCE_HZ
        ):
            violations.append(
                f"channel_{channel}_frequency_hz {frequency:.3f} differs from "
                f"{CAPTURE_FREQUENCY_HZ:.3f} by more than {CAPTURE_FREQUENCY_TOLERANCE_HZ:.3f} Hz"
            )
        if score < CAPTURE_FREQUENCY_SCORE_MIN:
            violations.append(
                f"channel_{channel}_frequency_identity_score {score:.6f} < "
                f"{CAPTURE_FREQUENCY_SCORE_MIN:.6f}"
            )
    if active_frames < MIN_ACTIVE_FRAMES:
        violations.append(f"active_frames {active_frames} < {MIN_ACTIVE_FRAMES}")
    if rms <= minimum_rms:
        violations.append(f"rms {rms:.9f} <= {minimum_rms}")
    if peak <= minimum_peak:
        violations.append(f"peak {peak:.9f} <= {minimum_peak}")
    if outside_peak >= OUTSIDE_THRESHOLD:
        violations.append(f"outside_peak {outside_peak:.9f} >= {OUTSIDE_THRESHOLD}")

    return {
        "capture": str(path),
        "volume_percent": volume_percent,
        "sample_rate": SAMPLE_RATE,
        "channels": STEREO,
        "frames": len(frames),
        "active_start_frame": active_start,
        "active_end_frame_exclusive": active_end_exclusive,
        "active_frames_before_trim": active_end_exclusive - active_start,
        "active_frames": active_frames,
        "active_duration_seconds": active_frames / SAMPLE_RATE,
        "trim_frames_each_edge": TRIM_FRAMES,
        "fixture_run_count": len(active_runs),
        "authenticated_fixture_run_count": len(authenticated_runs),
        "capture_frequency_hz": CAPTURE_FREQUENCY_HZ,
        "capture_frequency_tolerance_hz": CAPTURE_FREQUENCY_TOLERANCE_HZ,
        "channel_frequency_hz": channel_frequencies,
        "channel_frequency_identity_score": channel_scores,
        "leading_silence_frames": leading_silence_frames,
        "trailing_silence_frames": trailing_silence_frames,
        "rms": rms,
        "peak": peak,
        "outside_peak": outside_peak,
        "trailing_silence_peak": trailing_silence_peak,
        "expected_frames": FIXTURE_FRAMES,
        "required_trailing_silence_frames": FIXTURE_TRAILING_SILENCE_FRAMES,
        "minimum_rms": minimum_rms,
        "minimum_peak": minimum_peak,
        "valid": not violations,
        "violations": violations,
    }


def metrics_command(raw_path: Path, json_path: Path, human_path: Path, volume_percent: int) -> None:
    result = analyze_capture(raw_path, volume_percent)
    _write_json(json_path, result)
    _write_text(
        human_path,
        "\n".join(
            [
                f"capture: {raw_path}",
                f"volume_percent: {volume_percent}",
                f"frames: {result['frames']}",
                f"active_frames: {result['active_frames']}",
                f"active_duration_seconds: {result['active_duration_seconds']:.6f}",
                f"fixture_run_count: {result['fixture_run_count']}",
                f"authenticated_fixture_run_count: {result['authenticated_fixture_run_count']}",
                f"channel_frequency_hz: {result['channel_frequency_hz']}",
                f"channel_frequency_identity_score: {result['channel_frequency_identity_score']}",
                f"rms: {result['rms']:.9f}",
                f"peak: {result['peak']:.9f}",
                f"outside_peak: {result['outside_peak']:.9f}",
                f"trailing_silence_peak: {result['trailing_silence_peak']:.9f}",
                f"valid: {result['valid']}",
                f"violations: {', '.join(result['violations']) or 'none'}",
            ]
        ),
    )
    if not result["valid"]:
        raise ProbeError(f"capture validity failed: {raw_path}: {result['violations']}")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeError(f"expected JSON object in {path}")
    return value


def ratio_command(
    target_100_path: Path,
    target_10_path: Path,
    monitor_100_path: Path,
    monitor_10_path: Path,
    json_path: Path,
    human_path: Path,
) -> None:
    source_paths = {
        "target": (target_100_path, target_10_path),
        "monitor": (monitor_100_path, monitor_10_path),
    }
    destinations: dict[str, dict[str, object]] = {}
    for destination, (full_path, reduced_path) in source_paths.items():
        full = _load_json(full_path)
        reduced = _load_json(reduced_path)
        if full.get("valid") is not True or reduced.get("valid") is not True:
            raise ProbeError(
                f"{destination} metrics are not authenticated capture evidence"
            )
        rms_ratio = float(reduced["rms"]) / float(full["rms"])
        peak_ratio = float(reduced["peak"]) / float(full["peak"])
        ratios = (rms_ratio, peak_ratio)
        if all(0.075 <= ratio <= 0.13 for ratio in ratios):
            classification = "conforming-scaled"
        elif any(ratio >= 0.50 for ratio in ratios):
            classification = "confirmed-full-volume-signature"
        elif any(0.13 < ratio < 0.50 for ratio in ratios):
            classification = "inconclusive-nonconforming"
        else:
            classification = "inconclusive-ratio-out-of-range"
        destinations[destination] = {
            "rms_100": float(full["rms"]),
            "rms_10": float(reduced["rms"]),
            "peak_100": float(full["peak"]),
            "peak_10": float(reduced["peak"]),
            "rms_ratio_10_over_100": rms_ratio,
            "peak_ratio_10_over_100": peak_ratio,
            "ratio_range": [0.075, 0.13],
            "classification": classification,
            "full_volume_metrics": str(full_path),
            "ten_percent_metrics": str(reduced_path),
        }

    if all(item["classification"] == "conforming-scaled" for item in destinations.values()):
        overall = "conforming-scaled"
    elif any(item["classification"] == "confirmed-full-volume-signature" for item in destinations.values()):
        overall = "confirmed-full-volume-signature"
    else:
        overall = "inconclusive-nonconforming"
    result = {"overall_classification": overall, "destinations": destinations}
    _write_json(json_path, result)
    human_lines = [f"overall_classification: {overall}"]
    for destination, item in destinations.items():
        human_lines.extend(
            [
                f"{destination}.rms_10_over_100: {item['rms_ratio_10_over_100']:.9f}",
                f"{destination}.peak_10_over_100: {item['peak_ratio_10_over_100']:.9f}",
                f"{destination}.classification: {item['classification']}",
            ]
        )
    _write_text(human_path, "\n".join(human_lines))
    if overall != "conforming-scaled":
        raise ProbeError(f"volume ratio classification is {overall}; it never passes as scaled volume")


def _longest_active_run(peaks: list[float], threshold: float) -> tuple[int, int]:
    runs = _active_runs(peaks, threshold, MARKER_MAX_SILENT_GAP_FRAMES)
    return max(runs, key=lambda run: run[1] - run[0], default=(0, 0))


def _estimate_frequency(
    frames: list[tuple[float, float]], start: int, end: int, channel: int
) -> float:
    previous_sign: int | None = None
    previous_index: int | None = None
    previous_sample: float | None = None
    crossings: list[float] = []
    for index, frame in enumerate(frames[start:end]):
        sample = frame[channel]
        sign = 1 if sample > 0.0 else -1 if sample < 0.0 else 0
        if sign == 0:
            continue
        if previous_sign is not None and sign != previous_sign:
            assert previous_index is not None
            assert previous_sample is not None
            denominator = abs(previous_sample) + abs(sample)
            fraction = abs(previous_sample) / denominator if denominator else 0.0
            crossings.append(previous_index + fraction)
        previous_sign = sign
        previous_index = index
        previous_sample = sample
    if len(crossings) < 2:
        return 0.0
    mean_position = sum(crossings) / len(crossings)
    mean_crossing = (len(crossings) - 1) / 2.0
    numerator = sum(
        (position - mean_position) * (crossing - mean_crossing)
        for crossing, position in enumerate(crossings)
    )
    denominator = sum((position - mean_position) ** 2 for position in crossings)
    return (numerator / denominator) * SAMPLE_RATE / 2.0 if denominator > 0.0 else 0.0


def _frequency_identity_score(
    frames: list[tuple[float, float]], start: int, end: int, frequency_hz: float, channel: int
) -> float:
    samples = [frame[channel] for frame in frames[start:end]]
    if not samples:
        return 0.0
    mean = sum(samples) / len(samples)
    centered = [sample - mean for sample in samples]
    sample_energy = sum(sample * sample for sample in centered)
    if sample_energy <= 0.0:
        return 0.0
    sine_projection = 0.0
    cosine_projection = 0.0
    basis_energy = 0.0
    for index, sample in enumerate(centered):
        phase = 2.0 * math.pi * frequency_hz * index / SAMPLE_RATE
        sine = math.sin(phase)
        cosine = math.cos(phase)
        sine_projection += sample * sine
        cosine_projection += sample * cosine
        basis_energy += sine * sine + cosine * cosine
    score = math.sqrt(sine_projection**2 + cosine_projection**2) / math.sqrt(
        sample_energy * (basis_energy / 2.0)
    )
    return max(0.0, min(1.0, score))


def marker_command(expected_path: Path, other_path: Path, json_path: Path, human_path: Path, expected_destination: str) -> None:
    expected_frequency = MARKER_FREQUENCIES.get(expected_destination)
    if expected_frequency is None:
        raise ProbeError(f"unknown marker destination: {expected_destination}")
    expected_frames = _read_f32_stereo(expected_path)
    other_frames = _read_f32_stereo(other_path)
    expected_peaks = _frame_peaks(expected_frames)
    other_peaks = _frame_peaks(other_frames)
    expected_peak = max(expected_peaks, default=0.0)
    other_peak = max(other_peaks, default=0.0)
    expected_runs = _active_runs(expected_peaks, ACTIVE_THRESHOLD, MARKER_MAX_SILENT_GAP_FRAMES)
    active_start, active_end_exclusive = max(
        expected_runs,
        key=lambda run: run[1] - run[0],
        default=(0, 0),
    )
    expected_active_frames = active_end_exclusive - active_start
    channel_frequencies = [
        _estimate_frequency(expected_frames, active_start, active_end_exclusive, channel)
        for channel in range(STEREO)
    ]
    channel_scores = [
        _frequency_identity_score(
            expected_frames,
            active_start,
            active_end_exclusive,
            frequency,
            channel,
        )
        for channel, frequency in enumerate(channel_frequencies)
    ]
    authenticated_runs = []
    for start, end in expected_runs:
        if end - start < MARKER_MIN_ACTIVE_FRAMES:
            continue
        frequencies = [
            _estimate_frequency(expected_frames, start, end, channel)
            for channel in range(STEREO)
        ]
        scores = [
            _frequency_identity_score(expected_frames, start, end, frequency, channel)
            for channel, frequency in enumerate(frequencies)
        ]
        if all(
            _frequency_within_tolerance(
                frequency, expected_frequency, MARKER_FREQUENCY_TOLERANCE_HZ
            )
            and score >= MARKER_FREQUENCY_SCORE_MIN
            for frequency, score in zip(frequencies, scores)
        ):
            authenticated_runs.append((start, end))
    expected_outside_values = expected_peaks[:active_start] + expected_peaks[active_end_exclusive:]
    expected_outside_peak = max(expected_outside_values, default=0.0)
    measured_frequency = sum(channel_frequencies) / STEREO
    frequency_score = min(channel_scores, default=0.0)
    violations: list[str] = []
    if len(authenticated_runs) != 1:
        violations.append(f"authenticated_marker_runs {len(authenticated_runs)} != 1")
    if expected_active_frames < MARKER_MIN_ACTIVE_FRAMES:
        violations.append(
            f"expected_active_frames {expected_active_frames} < {MARKER_MIN_ACTIVE_FRAMES}"
        )
    if expected_peak <= 0.05:
        violations.append(f"expected_peak {expected_peak:.9f} <= 0.05")
    for channel, (frequency, score) in enumerate(zip(channel_frequencies, channel_scores)):
        if not _frequency_within_tolerance(
            frequency, expected_frequency, MARKER_FREQUENCY_TOLERANCE_HZ
        ):
            violations.append(
                f"channel_{channel}_measured_frequency_hz {frequency:.3f} differs from "
                f"expected {expected_frequency:.3f} by more than {MARKER_FREQUENCY_TOLERANCE_HZ:.3f} Hz"
            )
        if score < MARKER_FREQUENCY_SCORE_MIN:
            violations.append(
                f"channel_{channel}_frequency_identity_score {score:.6f} < "
                f"{MARKER_FREQUENCY_SCORE_MIN:.6f}"
            )
    if expected_outside_peak >= OUTSIDE_THRESHOLD:
        violations.append(
            f"expected_outside_peak {expected_outside_peak:.9f} >= {OUTSIDE_THRESHOLD}"
        )
    if other_peak >= OUTSIDE_THRESHOLD:
        violations.append(f"other_peak {other_peak:.9f} >= {OUTSIDE_THRESHOLD}")
    result = {
        "expected_destination": expected_destination,
        "expected_capture": str(expected_path),
        "other_capture": str(other_path),
        "expected_active_frames": expected_active_frames,
        "expected_active_start_frame": active_start,
        "expected_active_end_frame_exclusive": active_end_exclusive,
        "expected_peak": expected_peak,
        "other_peak": other_peak,
        "expected_outside_peak": expected_outside_peak,
        "expected_frequency_hz": expected_frequency,
        "measured_frequency_hz": measured_frequency,
        "channel_measured_frequency_hz": channel_frequencies,
        "frequency_identity_score": frequency_score,
        "channel_frequency_identity_score": channel_scores,
        "marker_min_active_frames": MARKER_MIN_ACTIVE_FRAMES,
        "marker_frequency_tolerance_hz": MARKER_FREQUENCY_TOLERANCE_HZ,
        "expected_active_threshold": ACTIVE_THRESHOLD,
        "other_silence_threshold": OUTSIDE_THRESHOLD,
        "valid": not violations,
        "violations": violations,
    }
    _write_json(json_path, result)
    _write_text(
        human_path,
        "\n".join(
            [
                f"expected_destination: {expected_destination}",
                f"expected_active_frames: {result['expected_active_frames']}",
                f"expected_frequency_hz: {result['expected_frequency_hz']:.3f}",
                f"measured_frequency_hz: {result['measured_frequency_hz']:.3f}",
                f"channel_measured_frequency_hz: {result['channel_measured_frequency_hz']}",
                f"frequency_identity_score: {result['frequency_identity_score']:.6f}",
                f"channel_frequency_identity_score: {result['channel_frequency_identity_score']}",
                f"expected_peak: {expected_peak:.9f}",
                f"other_peak: {other_peak:.9f}",
                f"expected_outside_peak: {expected_outside_peak:.9f}",
                f"valid: {result['valid']}",
                f"violations: {', '.join(result['violations']) or 'none'}",
            ]
        ),
    )
    if not result["valid"]:
        raise ProbeError(
            f"target-only/monitor-only marker routing failed for {expected_destination}: {violations}"
        )


def _parse_pw_link_list(text: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    current_target: str | None = None
    for line in text.splitlines():
        if line.startswith("  |<- "):
            if current_target is not None:
                links.append((line[6:].strip(), current_target))
            continue
        if line.startswith("  |-> "):
            continue
        stripped = line.strip()
        if stripped and ":" in stripped:
            current_target = stripped
    return links


def verify_graph_command(
    dump_path: Path,
    ports_path: Path,
    target_name: str,
    monitor_name: str,
    json_path: Path,
    human_path: Path,
) -> None:
    if target_name == monitor_name:
        raise ProbeError("target and monitor system_name values are not distinct")
    objects = json.loads(dump_path.read_text(encoding="utf-8"))
    if not isinstance(objects, list):
        raise ProbeError(f"expected pw-dump array in {dump_path}")
    wanted = {target_name, monitor_name}
    found: dict[str, dict[str, object]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        info = obj.get("info")
        if not isinstance(info, dict):
            continue
        props = info.get("props")
        if not isinstance(props, dict):
            continue
        name = props.get("node.name")
        if name in wanted:
            found[str(name)] = {
                "media_class": props.get("media.class"),
                "factory_name": props.get("factory.name"),
            }
    if set(found) != wanted:
        raise ProbeError(f"pw-dump did not contain exactly the requested nodes: {found}")
    wrong_class = {name: item["media_class"] for name, item in found.items() if item["media_class"] != "Audio/Sink"}
    if wrong_class:
        raise ProbeError(f"requested nodes were not Audio/Sink nodes: {wrong_class}")
    wrong_factory = {
        name: item["factory_name"]
        for name, item in found.items()
        if item["factory_name"] != "support.null-audio-sink"
    }
    if wrong_factory:
        raise ProbeError(f"requested nodes were not native support.null-audio-sink adapters: {wrong_factory}")

    ports_text = ports_path.read_text(encoding="utf-8")
    port_assertions: dict[str, dict[str, bool]] = {}
    for name in (target_name, monitor_name):
        port_assertions[name] = {
            "playback": any(line.strip().startswith(f"{name}:playback_") for line in ports_text.splitlines()),
            "monitor": any(line.strip().startswith(f"{name}:monitor_") for line in ports_text.splitlines()),
        }
    if not all(all(assertion.values()) for assertion in port_assertions.values()):
        raise ProbeError(f"requested sinks lacked playback and monitor ports: {port_assertions}")

    result = {
        "target_system_name": target_name,
        "monitor_system_name": monitor_name,
        "distinct_system_names": True,
        "nodes": found,
        "ports": port_assertions,
        "native_factory_required": "support.null-audio-sink",
    }
    _write_json(json_path, result)
    _write_text(
        human_path,
        "\n".join(
            [
                f"target_system_name: {target_name}",
                f"monitor_system_name: {monitor_name}",
                "distinct_system_names: true",
                "media_class: Audio/Sink (both)",
                "ports: playback_* and monitor_* (both)",
                "native_factory: verified separately through pw_virtual_device_native::list_nodes",
            ]
        ),
    )


def verify_links_command(
    links_path: Path,
    target_sink: str,
    monitor_sink: str,
    target_capture: str,
    monitor_capture: str,
    json_path: Path,
    human_path: Path,
) -> None:
    links = _parse_pw_link_list(links_path.read_text(encoding="utf-8"))
    expected = {target_capture: target_sink, monitor_capture: monitor_sink}
    details: dict[str, object] = {}
    for capture, sink in expected.items():
        capture_links = [(source, target) for source, target in links if target.startswith(f"{capture}:")]
        intended = {
            channel: any(
                source == f"{sink}:monitor_{channel}"
                and target.startswith(f"{capture}:")
                and target.endswith(f"_{channel}")
                for source, target in capture_links
            )
            for channel in ("FL", "FR")
        }
        wrong_sources = [source for source, _target in capture_links if not source.startswith(f"{sink}:monitor_")]
        details[capture] = {
            "intended_sink": sink,
            "links": capture_links,
            "intended_channels": intended,
            "wrong_sources": wrong_sources,
            "valid": all(intended.values()) and not wrong_sources,
        }
    valid = all(bool(item["valid"]) for item in details.values())
    result = {
        "target_sink": target_sink,
        "monitor_sink": monitor_sink,
        "captures": details,
        "valid": valid,
    }
    _write_json(json_path, result)
    _write_text(
        human_path,
        "\n".join(
            [
                f"target_capture: {target_capture} -> {target_sink}:monitor_FL/monitor_FR",
                f"monitor_capture: {monitor_capture} -> {monitor_sink}:monitor_FL/monitor_FR",
                f"valid: {valid}",
            ]
        ),
    )
    if not valid:
        raise ProbeError(f"capture links did not target the intended distinct monitor ports: {details}")


def wrapper_command(
    log_path: Path,
    target_name: str,
    monitor_name: str,
    json_path: Path,
    human_path: Path,
) -> None:
    text = log_path.read_text(encoding="utf-8")
    expected = [
        (target_name, "1.00"),
        (monitor_name, "1.00"),
        (target_name, "0.10"),
        (monitor_name, "0.10"),
    ]
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("argv:"):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)

    violations: list[str] = []
    invocations: list[dict[str, object]] = []
    seen: dict[tuple[str, str], int] = {}
    for block_number, block in enumerate(blocks, start=1):
        argv = block[0][len("argv:") :].strip().split()
        try:
            target_index = argv.index("--target")
            volume_index = argv.index("--volume")
            target = argv[target_index + 1]
            volume = argv[volume_index + 1]
        except (ValueError, IndexError):
            violations.append(f"invocation {block_number} has incomplete playback argv")
            continue
        pair = (target, volume)
        seen[pair] = seen.get(pair, 0) + 1
        lower_block = "\n".join(block).lower()
        playback = "--playback" in argv
        streaming = bool(re.search(r"\bstreaming\b", lower_block))
        control_volume = "1.000" if volume == "1.00" else "0.100" if volume == "0.10" else None
        matching_control = bool(
            control_volume
            and re.search(
                rf"stream set volume to {re.escape(control_volume)} - success",
                lower_block,
            )
        )
        exits = re.findall(r"^exit=([0-9]+)$", "\n".join(block), flags=re.MULTILINE)
        invocation_valid = playback and streaming and matching_control and exits == ["0"] and pair in expected
        if not playback:
            violations.append(f"invocation {block_number} is missing --playback")
        if not streaming:
            violations.append(f"invocation {block_number} is missing STREAMING evidence")
        if not matching_control:
            violations.append(f"invocation {block_number} is missing matching volume success evidence")
        if exits != ["0"]:
            violations.append(f"invocation {block_number} exit evidence is {exits}, expected ['0']")
        if pair not in expected:
            violations.append(f"invocation {block_number} has unexpected target/volume pair {pair}")
        invocations.append(
            {
                "target": target,
                "volume": volume,
                "playback": playback,
                "streaming": streaming,
                "matching_control_success": matching_control,
                "exit_statuses": exits,
                "valid": invocation_valid,
            }
        )

    observed = [
        {"target": invocation["target"], "volume": invocation["volume"]}
        for invocation in invocations
    ]
    for pair in expected:
        if seen.get(pair, 0) != 1:
            violations.append(f"expected playback {pair} has {seen.get(pair, 0)} invocation blocks, expected 1")
    if len(blocks) != len(expected):
        violations.append(f"wrapper log has {len(blocks)} invocation blocks, expected {len(expected)}")
    streaming_count = sum(bool(invocation["streaming"]) for invocation in invocations)
    control_success = {
        "1.000": sum(
            bool(invocation["matching_control_success"])
            for invocation in invocations
            if invocation["volume"] == "1.00"
        ),
        "0.100": sum(
            bool(invocation["matching_control_success"])
            for invocation in invocations
            if invocation["volume"] == "0.10"
        ),
    }
    exits = [status for invocation in invocations for status in invocation["exit_statuses"]]
    valid = not violations
    result = {
        "expected_plays": [{"target": target, "volume": volume} for target, volume in expected],
        "observed_plays": observed,
        "invocations": invocations,
        "streaming_count": streaming_count,
        "control_success": control_success,
        "exit_statuses": exits,
        "valid": valid,
        "violations": violations,
    }
    _write_json(json_path, result)
    _write_text(
        human_path,
        "\n".join(
            [
                f"observed_plays: {observed}",
                f"streaming_count: {streaming_count}",
                f"control_success: {control_success}",
                f"exit_statuses: {exits}",
                f"valid: {valid}",
                f"violations: {', '.join(violations) or 'none'}",
            ]
        ),
    )
    if not valid:
        raise ProbeError(
            f"pw-cat wrapper did not prove four successful streaming volume controls: {violations}"
        )


def _write_f32_capture(
    path: Path,
    amplitude: float,
    frames: int = FIXTURE_FRAMES,
    active_start: int = ACTIVE_START,
    active_end: int = ACTIVE_END,
    frequency_hz: float = 997.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for frame in range(frames):
            if active_start <= frame < active_end:
                sample = amplitude * math.sin(
                    2.0 * math.pi * frequency_hz * (frame - active_start) / SAMPLE_RATE
                )
            else:
                sample = 0.0
            output.write(struct.pack("<ff", sample, sample))


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="pipe-deck-407-probe-self-test-") as directory:
        root = Path(directory)
        fixture = root / "fixture.wav"
        fixture_meta = root / "fixture.json"
        make_fixture(fixture, fixture_meta)
        with wave.open(str(fixture), "rb") as input_wav:
            if (input_wav.getframerate(), input_wav.getnchannels(), input_wav.getsampwidth(), input_wav.getnframes()) != (48_000, 2, 2, 96_000):
                raise ProbeError("fixture self-test did not produce the required WAV shape")

        target_100 = root / "target-100.raw"
        target_10 = root / "target-10.raw"
        monitor_100 = root / "monitor-100.raw"
        monitor_10 = root / "monitor-10.raw"
        _write_f32_capture(target_100, 0.25)
        _write_f32_capture(target_10, 0.025)
        _write_f32_capture(monitor_100, 0.25)
        _write_f32_capture(monitor_10, 0.025)
        for raw, volume in ((target_100, 100), (target_10, 10), (monitor_100, 100), (monitor_10, 10)):
            result = analyze_capture(raw, volume)
            if not result["valid"]:
                raise ProbeError(f"metrics self-test failed for {raw}: {result['violations']}")

        extended = root / "extended.raw"
        _write_f32_capture(extended, 0.25, frames=288_000)
        extended_result = analyze_capture(extended, 100)
        if not extended_result["valid"]:
            raise ProbeError(f"extended capture self-test failed: {extended_result['violations']}")

        truncated = root / "truncated.raw"
        _write_f32_capture(truncated, 0.25, frames=80_000, active_end=80_000)
        truncated_result = analyze_capture(truncated, 100)
        if truncated_result["valid"]:
            raise ProbeError("truncated 80,000-frame capture unexpectedly passed metrics")

        tone_to_eof = root / "tone-to-eof.raw"
        _write_f32_capture(tone_to_eof, 0.25, active_end=FIXTURE_FRAMES)
        tone_to_eof_result = analyze_capture(tone_to_eof, 100)
        if tone_to_eof_result["valid"]:
            raise ProbeError("96,000-frame capture without trailing silence unexpectedly passed metrics")

        late_733_captures: dict[tuple[str, int], Path] = {}
        for destination in ("target", "monitor"):
            for volume, amplitude in ((100, 0.25), (10, 0.025)):
                raw = root / f"late-733-{destination}-{volume}.raw"
                _write_f32_capture(
                    raw,
                    amplitude,
                    frames=288_000,
                    active_start=200_000,
                    active_end=272_000,
                    frequency_hz=733.0,
                )
                result = analyze_capture(raw, volume)
                if result["valid"]:
                    raise ProbeError(
                        f"late 733 Hz {destination} {volume}% capture unexpectedly passed: {result}"
                    )
                late_733_captures[(destination, volume)] = raw
        _expect_probe_error(
            lambda: ratio_command(
                _metric_file(root, late_733_captures[("target", 100)], 100),
                _metric_file(root, late_733_captures[("target", 10)], 10),
                _metric_file(root, late_733_captures[("monitor", 100)], 100),
                _metric_file(root, late_733_captures[("monitor", 10)], 10),
                root / "late-733-ratio.json",
                root / "late-733-ratio.txt",
            ),
            "late 733 Hz capture ratio evidence",
        )

        def write_separated_bursts(path: Path, amplitude: float) -> None:
            with path.open("wb") as output:
                for frame in range(FIXTURE_FRAMES):
                    if 12_000 <= frame < 22_000 or 74_000 <= frame < 84_000:
                        sample = amplitude * math.sin(
                            2.0 * math.pi * 997.0 * (frame % 10_000) / SAMPLE_RATE
                        )
                    else:
                        sample = 0.0
                    output.write(struct.pack("<ff", sample, sample))

        for volume, amplitude in ((100, 0.25), (10, 0.025)):
            separated = root / f"separated-997-{volume}.raw"
            write_separated_bursts(separated, amplitude)
            result = analyze_capture(separated, volume)
            if result["valid"] or result["active_frames"] >= MIN_ACTIVE_FRAMES:
                raise ProbeError(
                    f"separated 997 Hz bursts unexpectedly passed the contiguous-duration gate: {result}"
                )

        ambiguous = root / "ambiguous-997.raw"
        with ambiguous.open("wb") as output:
            for frame in range(288_000):
                if 12_000 <= frame < 84_000 or 120_000 <= frame < 192_000:
                    sample = 0.25 * math.sin(
                        2.0 * math.pi * 997.0 * (frame % 72_000) / SAMPLE_RATE
                    )
                else:
                    sample = 0.0
                output.write(struct.pack("<ff", sample, sample))
        ambiguous_result = analyze_capture(ambiguous, 100)
        if ambiguous_result["valid"]:
            raise ProbeError(f"multiple authenticated fixture runs unexpectedly passed: {ambiguous_result}")

        shifted = root / "shifted-997.raw"
        _write_f32_capture(
            shifted,
            0.25,
            frames=320_000,
            active_start=30_000,
            active_end=102_000,
        )
        shifted_result = analyze_capture(shifted, 100)
        if not shifted_result["valid"]:
            raise ProbeError(f"valid fixture with extra leading/trailing silence failed: {shifted_result}")

        frequency_at_tolerance = root / "frequency-at-tolerance.raw"
        _write_f32_capture(
            frequency_at_tolerance,
            0.25,
            frequency_hz=997.0 + 10.0,
        )
        if not analyze_capture(frequency_at_tolerance, 100)["valid"]:
            raise ProbeError("fixture frequency exactly at tolerance unexpectedly failed")

        frequency_outside_tolerance = root / "frequency-outside-tolerance.raw"
        _write_f32_capture(
            frequency_outside_tolerance,
            0.25,
            frequency_hz=997.0 + 10.1,
        )
        if analyze_capture(frequency_outside_tolerance, 100)["valid"]:
            raise ProbeError("fixture frequency outside tolerance unexpectedly passed")

        ratio = root / "ratio.json"
        ratio_human = root / "ratio.txt"
        ratio_command(
            _metric_file(root, target_100, 100),
            _metric_file(root, target_10, 10),
            _metric_file(root, monitor_100, 100),
            _metric_file(root, monitor_10, 10),
            ratio,
            ratio_human,
        )

        marker = root / "marker.raw"
        silence = root / "silence.raw"
        _write_f32_capture(
            marker,
            0.25,
            frames=12_000,
            active_start=600,
            active_end=11_400,
            frequency_hz=733.0,
        )
        _write_f32_capture(silence, 0.0, frames=12_000, active_start=600, active_end=11_400)
        marker_command(marker, silence, root / "marker.json", root / "marker.txt", "target")

        monitor_marker = root / "monitor-marker.raw"
        _write_f32_capture(
            monitor_marker,
            0.25,
            frames=12_000,
            active_start=600,
            active_end=11_400,
            frequency_hz=1237.0,
        )
        marker_command(
            monitor_marker,
            silence,
            root / "monitor-marker.json",
            root / "monitor-marker.txt",
            "monitor",
        )

        wrong_frequency_marker = root / "wrong-frequency-marker.raw"
        _write_f32_capture(
            wrong_frequency_marker,
            0.25,
            frames=12_000,
            active_start=600,
            active_end=11_400,
        )
        _expect_probe_error(
            lambda: marker_command(
                wrong_frequency_marker,
                silence,
                root / "wrong-frequency-marker.json",
                root / "wrong-frequency-marker.txt",
                "target",
            ),
            "wrong-frequency marker",
        )

        impulse = root / "impulse.raw"
        with impulse.open("wb") as output:
            for frame in range(12_000):
                sample = 0.25 if frame == 6_000 else 0.0
                output.write(struct.pack("<ff", sample, sample))
        _expect_probe_error(
            lambda: marker_command(
                impulse,
                silence,
                root / "impulse-marker.json",
                root / "impulse-marker.txt",
                "target",
            ),
            "one-frame marker impulse",
        )

        phase_cancelled_marker = root / "phase-cancelled-marker.raw"
        with phase_cancelled_marker.open("wb") as output:
            for frame in range(12_000):
                if 600 <= frame < 11_400:
                    elapsed = frame - 600
                    common = 0.06 * math.sin(2.0 * math.pi * 733.0 * elapsed / SAMPLE_RATE)
                    wrong = 0.25 * math.sin(2.0 * math.pi * 997.0 * elapsed / SAMPLE_RATE)
                    left, right = common + wrong, common - wrong
                else:
                    left = right = 0.0
                output.write(struct.pack("<ff", left, right))
        _expect_probe_error(
            lambda: marker_command(
                phase_cancelled_marker,
                silence,
                root / "phase-cancelled-marker.json",
                root / "phase-cancelled-marker.txt",
                "target",
            ),
            "stereo phase-cancelled wrong-frequency marker",
        )

        links = root / "links.txt"
        _write_text(
            links,
            "\n".join(
                [
                    "capture-target:input_FL",
                    "  |<- target:monitor_FL",
                    "capture-target:input_FR",
                    "  |<- target:monitor_FR",
                    "capture-monitor:input_FL",
                    "  |<- monitor:monitor_FL",
                    "capture-monitor:input_FR",
                    "  |<- monitor:monitor_FR",
                ]
            ),
        )
        verify_links_command(
            links,
            "target",
            "monitor",
            "capture-target",
            "capture-monitor",
            root / "links.json",
            root / "links-human.txt",
        )

        wrapper = root / "pw-cat.log"
        _write_text(
            wrapper,
            "\n".join(
                [
                    "argv: --playback --target target --volume 1.00 fixture.wav",
                    "stream state: STREAMING",
                    "stream set volume to 1.000 - success",
                    "exit=0",
                    "argv: --playback --target monitor --volume 1.00 fixture.wav",
                    "stream state: STREAMING",
                    "stream set volume to 1.000 - success",
                    "exit=0",
                    "argv: --playback --target target --volume 0.10 fixture.wav",
                    "stream state: STREAMING",
                    "stream set volume to 0.100 - success",
                    "exit=0",
                    "argv: --playback --target monitor --volume 0.10 fixture.wav",
                    "stream state: STREAMING",
                    "stream set volume to 0.100 - success",
                    "exit=0",
                ]
            ),
        )
        wrapper_command(wrapper, "target", "monitor", root / "wrapper.json", root / "wrapper.txt")

        aggregate_wrapper = root / "aggregate-pw-cat.log"
        _write_text(
            aggregate_wrapper,
            "\n".join(
                [
                    "argv: --playback --target target --volume 1.00 fixture.wav",
                    "stream state: STREAMING",
                    "stream state: STREAMING",
                    "stream state: STREAMING",
                    "stream state: STREAMING",
                    "stream set volume to 1.000 - success",
                    "stream set volume to 1.000 - success",
                    "stream set volume to 0.100 - success",
                    "stream set volume to 0.100 - success",
                    "exit=0",
                    "argv: --playback --target monitor --volume 1.00 fixture.wav",
                    "exit=0",
                    "argv: --playback --target target --volume 0.10 fixture.wav",
                    "exit=0",
                    "argv: --playback --target monitor --volume 0.10 fixture.wav",
                    "exit=0",
                ]
            ),
        )
        _expect_probe_error(
            lambda: wrapper_command(
                aggregate_wrapper,
                "target",
                "monitor",
                root / "aggregate-wrapper.json",
                root / "aggregate-wrapper.txt",
            ),
            "aggregate wrapper evidence",
        )
    print("self-test: ok")


def _metric_file(root: Path, raw_path: Path, volume: int) -> Path:
    path = root / f"{raw_path.stem}.json"
    _write_json(path, analyze_capture(raw_path, volume))
    return path


def _required_path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--output", required=True, type=_required_path)
    fixture.add_argument("--metadata", required=True, type=_required_path)

    marker = subparsers.add_parser("marker")
    marker.add_argument("--output", required=True, type=_required_path)
    marker.add_argument("--metadata", required=True, type=_required_path)
    marker.add_argument("--frequency", required=True, type=float)

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--raw", required=True, type=_required_path)
    metrics.add_argument("--json", required=True, type=_required_path)
    metrics.add_argument("--human", required=True, type=_required_path)
    metrics.add_argument("--volume", required=True, type=int, choices=(10, 100))

    ratio = subparsers.add_parser("ratio")
    ratio.add_argument("--target-100", required=True, type=_required_path)
    ratio.add_argument("--target-10", required=True, type=_required_path)
    ratio.add_argument("--monitor-100", required=True, type=_required_path)
    ratio.add_argument("--monitor-10", required=True, type=_required_path)
    ratio.add_argument("--json", required=True, type=_required_path)
    ratio.add_argument("--human", required=True, type=_required_path)

    marker_check = subparsers.add_parser("marker-check")
    marker_check.add_argument("--expected", required=True, type=_required_path)
    marker_check.add_argument("--other", required=True, type=_required_path)
    marker_check.add_argument("--json", required=True, type=_required_path)
    marker_check.add_argument("--human", required=True, type=_required_path)
    marker_check.add_argument("--destination", required=True)

    graph = subparsers.add_parser("verify-graph")
    graph.add_argument("--dump", required=True, type=_required_path)
    graph.add_argument("--ports", required=True, type=_required_path)
    graph.add_argument("--target", required=True)
    graph.add_argument("--monitor", required=True)
    graph.add_argument("--json", required=True, type=_required_path)
    graph.add_argument("--human", required=True, type=_required_path)

    links = subparsers.add_parser("verify-links")
    links.add_argument("--links", required=True, type=_required_path)
    links.add_argument("--target-sink", required=True)
    links.add_argument("--monitor-sink", required=True)
    links.add_argument("--target-capture", required=True)
    links.add_argument("--monitor-capture", required=True)
    links.add_argument("--json", required=True, type=_required_path)
    links.add_argument("--human", required=True, type=_required_path)

    wrapper = subparsers.add_parser("wrapper")
    wrapper.add_argument("--log", required=True, type=_required_path)
    wrapper.add_argument("--target", required=True)
    wrapper.add_argument("--monitor", required=True)
    wrapper.add_argument("--json", required=True, type=_required_path)
    wrapper.add_argument("--human", required=True, type=_required_path)

    subparsers.add_parser("self-test")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "fixture":
            make_fixture(args.output, args.metadata)
        elif args.command == "marker":
            make_marker(args.output, args.metadata, args.frequency)
        elif args.command == "metrics":
            metrics_command(args.raw, args.json, args.human, args.volume)
        elif args.command == "ratio":
            ratio_command(args.target_100, args.target_10, args.monitor_100, args.monitor_10, args.json, args.human)
        elif args.command == "marker-check":
            marker_command(args.expected, args.other, args.json, args.human, args.destination)
        elif args.command == "verify-graph":
            verify_graph_command(args.dump, args.ports, args.target, args.monitor, args.json, args.human)
        elif args.command == "verify-links":
            verify_links_command(args.links, args.target_sink, args.monitor_sink, args.target_capture, args.monitor_capture, args.json, args.human)
        elif args.command == "wrapper":
            wrapper_command(args.log, args.target, args.monitor, args.json, args.human)
        elif args.command == "self-test":
            self_test()
        else:
            raise ProbeError(f"unknown command: {args.command}")
    except (OSError, ValueError, json.JSONDecodeError, wave.Error, ProbeError) as error:
        print(f"probe helper failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
