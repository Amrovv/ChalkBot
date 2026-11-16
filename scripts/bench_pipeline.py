"""Benchmark the fast path on real footage: image in, coaching speech out.

Three commands, run from the repo root:

  uv run python -m scripts.bench_pipeline stages --out docs/benchmarks/stages.json
      Every frame of each clip, one stage at a time, as fast as it decodes: how long decoding,
      pose inference, contact tracking, the coach brain and the voice lookup each take.
      Needs no audio device and plays nothing.

  uv run python -m scripts.bench_pipeline e2e --out docs/benchmarks/e2e.json
      Each clip in real time through the real threads, queues, brain and voice, with a real
      audio output stream. For every spoken cue: the time from the capture of the frame that
      triggered it to the moment its audio reaches the output. Muted unless --audible (the
      stream is the same one, fed silence, so the timing is the same).

  uv run python -m scripts.bench_pipeline report docs/benchmarks/stages.json \
      docs/benchmarks/e2e.json > docs/BENCHMARKS.md
      The two result files as one markdown report.

--clips takes name fragments ("climb_1_" "demo") to run a subset; the default is every clip
under videos/ that has a hold map and is not an empty-wall clip.

What the latency figure covers, and what it does not, is written in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run as a script as well as with -m

from coach.bus import Bus  # noqa: E402
from coach.contracts import SpeakRequest  # noqa: E402
from scripts.bench_latency import percentile, table  # noqa: E402

log = logging.getLogger("bench_pipeline")

VIDEO_DIR = ROOT / "videos"
CONFIG_PATH = ROOT / "config" / "default.yaml"
LATENCY_TARGET_S = 0.8  # the project's goal: cue audio within ~0.8 s of the frame
QUIET_POLL_S = 0.05  # how often the end of a clip checks whether the coach has finished
QUIET_SETTLE_S = 0.5  # the coach must stay quiet this long before a clip counts as over
QUIET_TIMEOUT_S = 25.0  # never wait longer than this for a debrief to finish
PROGRESS_EVERY_FRAMES = 300
KEPT_TOPICS = ("event", "spoken", "suppressed", "attempt")
SESSION_PART = re.compile(r"(\d{2})-\d+$")  # clip "12-1", "12-2": one session filmed in parts
COPY_SUFFIX = re.compile(r" \(\d+\)$")  # "clip (1).mp4": another take with the same name

# What each numbered test clip was filmed to provoke (TEST_CLIMBS.md), as the coach brain
# defines the event: quiet_feet fires on a fidget, so the clean-footwork clip must stay silent.
# ("event", kind): the brain must detect it. ("no_event", kind): it must not.
# ("outcome", o): an attempt must end that way. ("starts", n): at least n attempts start.
EXPECTED: dict[str, tuple[str, Any]] = {
    "01": ("no_event", "attempt_start"),
    "02": ("outcome", "top"),
    "03": ("event", "feet_low"),
    "04": ("event", "arms_bent"),
    "05": ("event", "hand_searching"),
    "06": ("event", "feet_cut"),
    "07": ("event", "stalling"),
    "08": ("no_event", "quiet_feet"),
    "09": ("event", "quiet_feet"),
    "10": ("event", "weight_shift"),
    "11": ("event", "route_preview"),
    "12": ("event", "high_point"),
    "13": ("event", "fall"),
    "14": ("outcome", "downclimb"),
    "15": ("outcome", "fall"),
    "16": ("event", "weight_shift"),
    "17": ("event", "weight_shift"),
    "18": ("starts", 2),
}


# ---- Statistics ---------------------------------------------------------------------------


def summarise(seconds: Sequence[float]) -> dict:
    """n, mean, median, p95, p99 and max of a list of durations, in milliseconds."""
    if not seconds:
        return {"n": 0}
    ms = [s * 1000 for s in seconds]
    return {
        "n": len(ms),
        "mean": round(statistics.fmean(ms), 3),
        "p50": round(percentile(ms, 50), 3),
        "p95": round(percentile(ms, 95), 3),
        "p99": round(percentile(ms, 99), 3),
        "max": round(max(ms), 3),
    }


def merge_ms(parts: Iterable[Sequence[float]]) -> list[float]:
    return [value for part in parts for value in part]


def summarise_ms(values_ms: Sequence[float]) -> dict:
    return summarise([value / 1000 for value in values_ms])


# ---- Clips ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    path: Path
    problem: str
    number: str | None  # the clip's number in TEST_CLIMBS.md ("07"), if its name has one
    session: str  # clips that share a session run through one brain, in order
    opens_session: bool = False  # the first part of a session filmed in several parts

    @property
    def name(self) -> str:
        return self.path.name


def clip_number(stem: str, problem: str) -> tuple[str | None, str | None]:
    """("12", "12-1") for "climb_6_12-1_new_high_point": the test number and, for a session
    filmed in several parts, which part. (number, None) for an ordinary numbered clip."""
    rest = stem[len(problem) :].lstrip("_")
    token = rest.split("_", 1)[0]
    part = SESSION_PART.fullmatch(token)
    if part:
        return part.group(1), token
    return (token, None) if re.fullmatch(r"\d{2}", token) else (None, None)


def find_clips(
    video_dir: Path = VIDEO_DIR,
    known: Iterable[str] | None = None,
    only: Sequence[str] = (),
) -> list[Clip]:
    """Every clip that can be analysed: it has a hold map and is not an empty-wall clip."""
    import run

    known = set(run.holdmap_ids() if known is None else known)
    clips = []
    for path in run.list_clips(video_dir):
        problem = run.problem_for(path, "", known)
        if problem not in known or run.is_empty_wall(path):
            continue
        if only and not any(fragment in path.name for fragment in only):
            continue
        number, part = clip_number(COPY_SUFFIX.sub("", path.stem), problem)
        session = f"{problem}/{number}" if part else path.stem
        opens = bool(part) and all(clip.session != session for clip in clips)
        clips.append(Clip(path, problem, number, session, opens))
    return clips


def sessions_of(clips: Sequence[Clip]) -> list[list[Clip]]:
    """Clips grouped into sessions, keeping their order."""
    groups: dict[str, list[Clip]] = {}
    for clip in clips:
        groups.setdefault(clip.session, []).append(clip)
    return list(groups.values())


def video_info(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {"frames": frames, "fps": round(fps, 2), "duration_s": round(duration, 2),
            "size": [width, height]}  # fmt: skip


# ---- Environment ----------------------------------------------------------------------------


def cpu_name() -> str:
    """The processor's marketing name, where the platform will say."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20, check=False,
            ).stdout  # fmt: skip
            if out.strip():
                return out.strip().splitlines()[0]
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        log.debug("could not read the CPU name")
    return platform.processor() or platform.machine()


def environment() -> dict:
    versions = {}
    for package in ("mediapipe", "opencv-contrib-python", "numpy", "kokoro", "sounddevice"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        # Capture times and audio start times are both time.monotonic(): its tick is the
        # finest step any latency here can show (15.6 ms on Windows before Python 3.13).
        "clock_resolution_ms": round(time.get_clock_info("monotonic").resolution * 1000, 3),
        "packages": versions,
        "date": time.strftime("%Y-%m-%d"),
    }


# ---- Stage benchmark ------------------------------------------------------------------------


class TimedDetector:
    """Wraps the pose detector and remembers how long its last call took."""

    def __init__(self, inner: Callable) -> None:
        self.inner = inner
        self.last_s = 0.0

    def __call__(self, frame, timestamp_ms: int):
        start = time.perf_counter()
        try:
            return self.inner(frame, timestamp_ms)
        finally:
            self.last_s = time.perf_counter() - start

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close:
            close()


class SilentSpeaker:
    """An output that plays nothing and is never busy: the stage benchmark's speaker."""

    latency = 0.0

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock

    def play(self, samples: np.ndarray) -> float:
        return self._clock()

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class TimedVoice:
    """The voice as the brain sees it, timing each say() and noting whether it spoke."""

    def __init__(self, voice) -> None:
        self._voice = voice
        self.say_s = 0.0
        self.spoke = False

    def reset(self) -> None:
        self.say_s, self.spoke = 0.0, False

    def say(self, req: SpeakRequest) -> bool:
        start = time.perf_counter()
        try:
            ok = self._voice.say(req)
        finally:
            self.say_s += time.perf_counter() - start
        self.spoke = self.spoke or ok
        return ok

    def __getattr__(self, name: str):
        return getattr(self._voice, name)


def offline_voice(cfg: dict, bus, clock: Callable[[], float]):
    """The real Voice (phrase bank, cached clips, stitching) over a speaker that plays nothing."""
    from coach.voice.tts import make_tts
    from coach.voice.voice import BANK_PATH, Voice

    voice_cfg = cfg["voice"]
    persona = yaml.safe_load(Path(voice_cfg["persona"]).read_text(encoding="utf-8"))
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    return Voice(
        bank,
        make_tts(voice_cfg, persona),
        SilentSpeaker(clock),
        bus,
        fillers=persona["fillers"],
        clock=clock,
        hold_bands=persona.get("hold_places", ()),
    )


def load_map(problem: str):
    """(HoldMap, shapes), or (None, {}) when the problem has no map."""
    from coach.perception.holds import load_holdmap_with_shapes

    try:
        return load_holdmap_with_shapes(problem)
    except FileNotFoundError:
        return None, {}


STAGES = ("decode", "pose", "contacts", "brain", "voice")
STAGE_LABELS = {
    "decode": "Decode the frame (OpenCV)",
    "pose": "Pose inference (MediaPipe, incl. resize)",
    "contacts": "Smoothing, left/right fix, hold contacts",
    "brain": "Coach brain (features, 9 detectors, memory, arbiter)",
    "voice": "Voice: pick line, load and stitch clips",
}


def stage_clip(
    path: Path,
    problem: str,
    cfg: dict,
    *,
    detector: Callable | None = None,
    make_voice: Callable = offline_voice,
    max_frames: int | None = None,
) -> dict:
    """Run every frame of a clip through each stage in turn and time them.

    Capture time is video time (frame / fps), so the brain behaves exactly as it does on a
    replay; only the cost of each stage is measured, with no queues and no waiting.
    """
    from coach.brain import Brain
    from coach.perception.pose import MediapipeDetector, PoseTracker, load_perception_config
    from coach.sources import FileSource

    holdmap, shapes = load_map(problem)
    pcfg = load_perception_config()
    timed = TimedDetector(detector or MediapipeDetector(pcfg))
    tracker = PoseTracker(pcfg, detector=timed, holdmap=holdmap, shapes=shapes)
    now = [0.0]
    bus = Bus()
    spoken = bus.subscribe("spoken")
    voice = TimedVoice(make_voice(cfg, bus, lambda: now[0]))
    brain = Brain(cfg, bus, voice)
    brain.attempts.holdmap = holdmap
    if holdmap is not None:
        bus.publish("holdmap", holdmap)  # the voice names the colour and the holds from it

    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    frame_total: list[float] = []
    cue_total: list[float] = []
    source = FileSource(path, fast=True)
    try:
        mark = time.perf_counter()
        for _, idx, frame in source:
            decode = time.perf_counter() - mark
            now[0] = idx / source.fps
            start = time.perf_counter()
            state = tracker.process(now[0], idx, frame)
            perception = time.perf_counter() - start
            voice.reset()
            start = time.perf_counter()
            brain._process_state(state)
            thinking = time.perf_counter() - start
            while not spoken.empty():  # what the brain's own loop does between frames
                brain.memory.on_spoken(spoken.get_nowait())

            samples["decode"].append(decode)
            samples["pose"].append(timed.last_s)
            samples["contacts"].append(max(perception - timed.last_s, 0.0))
            samples["brain"].append(max(thinking - voice.say_s, 0.0))
            total = decode + perception + thinking
            frame_total.append(total)
            if voice.spoke:
                samples["voice"].append(voice.say_s)
                cue_total.append(total)
            if max_frames is not None and len(frame_total) >= max_frames:
                break
            if len(frame_total) % PROGRESS_EVERY_FRAMES == 0:
                log.info("  %s: %d frames", path.name, len(frame_total))
            mark = time.perf_counter()
    finally:
        source.close()
        tracker.close()
        close = getattr(voice, "close", None)
        if close:
            close()
    to_ms = lambda values: [round(v * 1000, 4) for v in values]  # noqa: E731
    return {
        "clip": path.name,
        "problem": problem,
        "frames": len(frame_total),
        "samples_ms": {stage: to_ms(values) for stage, values in samples.items()},
        "frame_total_ms": to_ms(frame_total),
        "cue_frame_total_ms": to_ms(cue_total),
    }


def stage_summary(clips: Sequence[dict]) -> dict:
    """The per-clip samples as one table of statistics."""
    out = {stage: summarise_ms(merge_ms(c["samples_ms"][stage] for c in clips)) for stage in STAGES}
    frame = merge_ms(c["frame_total_ms"] for c in clips)
    out["frame"] = summarise_ms(frame)
    out["cue_frame"] = summarise_ms(merge_ms(c["cue_frame_total_ms"] for c in clips))
    out["throughput_fps"] = round(1000 / statistics.fmean(frame), 1) if frame else 0.0
    return out


def run_stages(clips: Sequence[Clip], cfg: dict, max_frames: int | None = None) -> dict:
    results = []
    for n, clip in enumerate(clips, 1):
        log.info("[%d/%d] stages: %s (%s)", n, len(clips), clip.name, clip.problem)
        results.append(stage_clip(clip.path, clip.problem, cfg, max_frames=max_frames))
    clips_out = [
        {"clip": r["clip"], "problem": r["problem"], "frames": r["frames"]} for r in results
    ]
    return {
        "kind": "stages",
        "environment": environment(),
        "clips": clips_out,
        "summary": stage_summary(results),
    }


# ---- End-to-end benchmark -------------------------------------------------------------------


class Probe:
    """The bus as every layer sees it, stamping each publish at the moment it happens.

    Perception publishes from its own thread and the brain and voice from the event loop, so
    the stamps are the real hand-over times, not the times a logger got round to writing them.
    """

    def __init__(self, bus: Bus, clock: Callable[[], float] = time.monotonic) -> None:
        self._bus, self._clock = bus, clock
        self.records: list[tuple[float, str, Any]] = []
        self.state_delays: list[float] = []

    def publish(self, topic: str, payload: Any) -> None:
        now = self._clock()
        if topic == "state":
            self.state_delays.append(now - payload.t)
        elif topic in KEPT_TOPICS:
            self.records.append((now, topic, payload))
        self._bus.publish(topic, payload)

    def take(self) -> tuple[list[tuple[float, str, Any]], list[float]]:
        """What was published since the last take()."""
        records, delays = self.records, self.state_delays
        self.records, self.state_delays = [], []
        return records, delays

    def __getattr__(self, name: str):
        return getattr(self._bus, name)


@dataclass
class CueTiming:
    kind: str
    capture_to_event: float | None  # frame captured -> the brain published the event
    event_to_audio: float | None  # event published -> audio reaches the output
    total: float  # frame captured -> audio reaches the output


def cue_timings(records: Sequence[tuple[float, str, Any]]) -> list[CueTiming]:
    """One timing per spoken line, matched to its event by (capture time, kind)."""
    published: dict[tuple[float, str], float] = {}
    cues = []
    for stamp, topic, payload in records:
        if topic == "event":
            published.setdefault((payload.t, payload.kind), stamp)
        elif topic == "spoken":
            at = published.get((payload.t_event, payload.kind))
            cues.append(
                CueTiming(
                    payload.kind,
                    None if at is None else at - payload.t_event,
                    None if at is None else payload.t_audio_start - at,
                    payload.t_audio_start - payload.t_event,
                )
            )
    return cues


def scenario_check(number: str | None, events: dict, outcomes: Sequence[str]) -> dict | None:
    """Did the clip show what it was filmed to show? None for a clip with no expectation."""
    if number not in EXPECTED:
        return None
    rule, want = EXPECTED[number]
    if rule == "event":
        return {"expected": f"`{want}` detected", "ok": events.get(want, 0) > 0}
    if rule == "no_event":
        return {"expected": f"no `{want}`", "ok": events.get(want, 0) == 0}
    if rule == "outcome":
        return {"expected": f'an attempt ends "{want}"', "ok": want in outcomes}
    return {
        "expected": f"at least {want} attempts start",
        "ok": events.get("attempt_start", 0) >= want,
    }


def clip_result(clip: Clip, info: dict, records, state_delays: Sequence[float]) -> dict:
    """Everything measured on one clip, as plain JSON."""
    events = Counter(p.kind for _, topic, p in records if topic == "event")
    spoken = Counter(p.kind for _, topic, p in records if topic == "spoken")
    suppressed = Counter(p.reason for _, topic, p in records if topic == "suppressed")
    attempts = [p for _, topic, p in records if topic == "attempt"]
    outcomes = [a.outcome for a in attempts]
    cues = cue_timings(records)
    analysed = len(state_delays)
    return {
        "clip": clip.name,
        "problem": clip.problem,
        "number": clip.number,
        **info,
        "frames_analysed": analysed,
        "analysed_fps": round(analysed / info["duration_s"], 1) if info["duration_s"] else 0.0,
        "events": dict(events),
        "spoken": dict(spoken),
        "suppressed": dict(suppressed),
        "attempts": [
            {"no": a.attempt_no, "outcome": a.outcome, "high_hold": a.high_hold_id}
            for a in attempts
        ],
        # A session's first part only sets the scene (a new high point needs an attempt to beat).
        "check": None if clip.opens_session else scenario_check(clip.number, events, outcomes),
        "state_delay_ms": [round(d * 1000, 2) for d in state_delays],
        "cues": [
            {
                "kind": c.kind,
                "capture_to_event_ms": None
                if c.capture_to_event is None
                else round(c.capture_to_event * 1000, 2),
                "event_to_audio_ms": None
                if c.event_to_audio is None
                else round(c.event_to_audio * 1000, 2),
                "total_ms": round(c.total * 1000, 2),
            }
            for c in cues
        ],
    }


@dataclass
class Rig:
    """What an end-to-end run needs; tests swap in fakes for the slow or noisy parts."""

    cfg: dict
    probe: Probe
    voice: Any
    make_brain: Callable
    parts: Any = None  # perception's building blocks (run.a_parts() when None)
    info: Callable[[Path], dict] = video_info
    quiet_settle_s: float = QUIET_SETTLE_S
    quiet_timeout_s: float = QUIET_TIMEOUT_S
    results: list[dict] = field(default_factory=list)


async def wait_until_quiet(voice, settle_s: float, timeout_s: float) -> None:
    """Return once the coach has said nothing for settle_s (a debrief may still be running
    when a clip ends), or after timeout_s."""
    deadline = time.monotonic() + timeout_s
    quiet_since: float | None = None
    while time.monotonic() < deadline:
        if voice.is_speaking():
            quiet_since = None
        else:
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since >= settle_s:
                return
        await asyncio.sleep(QUIET_POLL_S)
    log.warning("the coach was still speaking after %.0f s; moving on", timeout_s)


async def run_session(rig: Rig, clips: Sequence[Clip]) -> None:
    """Play one session's clips in real time through one brain, the way run.py wires it:
    reader thread -> 1-slot queue -> pose thread -> 1-slot queue -> brain on the loop."""
    import run
    from coach import stubs

    loop = asyncio.get_running_loop()
    states = stubs.AsyncLatestQueue(loop)
    brain = rig.make_brain(rig.cfg, rig.probe, rig.voice)
    brain_task = asyncio.create_task(brain.run(states), name="brain")
    try:
        for clip in clips:
            ended = asyncio.Event()
            workers = run.PoseThreads(
                str(clip.path), clip.problem, rig.cfg, rig.probe, states, parts=rig.parts
            )
            workers.on_ended = lambda ended=ended: loop.call_soon_threadsafe(ended.set)
            rig.probe.take()
            workers.start()
            await ended.wait()
            await wait_until_quiet(rig.voice, rig.quiet_settle_s, rig.quiet_timeout_s)
            await asyncio.to_thread(workers.stop)
            records, delays = rig.probe.take()
            rig.results.append(clip_result(clip, rig.info(clip.path), records, delays))
    finally:
        brain_task.cancel()
        await asyncio.gather(brain_task, return_exceptions=True)
        rig.voice.stop()


def mute(voice) -> None:
    """Feed the real output stream silence: same stream, same callbacks, same reported
    latency, nothing audible. The clip's length is kept, so the coach is busy as long."""
    speaker = voice._speaker
    play = speaker.play
    speaker.play = lambda samples: play(np.zeros_like(samples))


def device_split(cues: Sequence[dict], output_latency_s: float) -> dict:
    """The software part of each cue's latency, without the audio device's output buffer.

    Speaker.play() stamps a cue's audio start as the moment it was called plus the stream's
    output latency, which is fixed for the stream. Taking that latency off again gives exactly
    when the line was handed to the audio device. The buffer depends on the sound card and
    driver, not on this code, so it is reported apart.
    """
    latency_ms = output_latency_s * 1000
    return {
        "to_device": summarise_ms([c["total_ms"] - latency_ms for c in cues]),
        "event_to_device": summarise_ms(
            [
                c["event_to_audio_ms"] - latency_ms
                for c in cues
                if c["event_to_audio_ms"] is not None
            ]
        ),
    }


def e2e_summary(clips: Sequence[dict], output_latency_s: float) -> dict:
    cues = [cue for clip in clips for cue in clip["cues"]]
    totals = [c["total_ms"] for c in cues]
    by_kind: dict[str, list[float]] = {}
    for cue in cues:
        by_kind.setdefault(cue["kind"], []).append(cue["total_ms"])
    suppressed: Counter = Counter()
    for clip in clips:
        suppressed.update(clip["suppressed"])
    checks = [clip["check"] for clip in clips if clip["check"]]
    target_ms = LATENCY_TARGET_S * 1000
    return {
        "cues": len(cues),
        "total": summarise_ms(totals),
        **device_split(cues, output_latency_s),
        "capture_to_event": summarise_ms(
            [c["capture_to_event_ms"] for c in cues if c["capture_to_event_ms"] is not None]
        ),
        "event_to_audio": summarise_ms(
            [c["event_to_audio_ms"] for c in cues if c["event_to_audio_ms"] is not None]
        ),
        "frame_to_state": summarise_ms(merge_ms(clip["state_delay_ms"] for clip in clips)),
        "by_kind": {kind: summarise_ms(values) for kind, values in sorted(by_kind.items())},
        "within_target": sum(t <= target_ms for t in totals),
        "target_ms": target_ms,
        "output_latency_ms": round(output_latency_s * 1000, 2),
        "events": sum(sum(clip["events"].values()) for clip in clips),
        "suppressed": dict(suppressed),
        "frames_total": sum(clip["frames"] for clip in clips),
        "frames_analysed": sum(clip["frames_analysed"] for clip in clips),
        "footage_s": round(sum(clip["duration_s"] for clip in clips), 1),
        "checks_passed": sum(check["ok"] for check in checks),
        "checks_total": len(checks),
    }


async def run_e2e(clips: Sequence[Clip], cfg: dict, audible: bool = False) -> dict:
    from coach.brain import Brain
    from coach.voice import make_voice

    bus = Bus()
    bus.bind_loop()
    probe = Probe(bus)
    voice = make_voice(cfg, probe, no_llm=True)  # template debriefs: no network in the benchmark
    if not audible:
        mute(voice)
    # make_voice loads the TTS model on the voice's worker; wait for that, as a coach that
    # has been switched on for a minute would have, so it doesn't compete with the first clip.
    await asyncio.to_thread(voice._pool.submit(lambda: None).result)
    output_latency = float(voice._speaker.latency)
    rig = Rig(cfg, probe, voice, Brain)
    groups = sessions_of(clips)
    try:
        for n, group in enumerate(groups, 1):
            names = ", ".join(clip.name for clip in group)
            log.info("[%d/%d] e2e: %s", n, len(groups), names)
            await run_session(rig, group)
    finally:
        voice.close()
        bus.close()
    slim = [{k: v for k, v in clip.items() if k != "state_delay_ms"} for clip in rig.results]
    return {
        "kind": "e2e",
        "environment": {**environment(), "audible": audible},
        "summary": e2e_summary(rig.results, output_latency),
        "clips": slim,
    }


# ---- Report ---------------------------------------------------------------------------------


def stat_row(label: str, stats: dict) -> list[str]:
    if not stats.get("n"):
        return [label, "–", "–", "–", "–", "0"]
    cells = [f"{stats[key]:.1f}" for key in ("p50", "mean", "p95", "max")]
    return [label, *cells, str(stats["n"])]


STAT_HEADERS = ["median", "mean", "p95", "max", "n"]


def environment_lines(env: dict) -> list[str]:
    packages = ", ".join(f"{name} {version}" for name, version in env["packages"].items())
    return [
        f"- **Machine:** {env['cpu']}, {env['logical_cores']} logical cores, {env['os']}",
        f"- **Software:** Python {env['python']}; {packages}",
        f"- **Timer:** `time.monotonic()`, {env.get('clock_resolution_ms', '?')} ms per tick: "
        "no latency below is finer than that",
        f"- **Run on:** {env['date']}",
    ]


def e2e_report(data: dict) -> str:
    s = data["summary"]
    total = s["total"]
    share = f"{s['within_target']} of {s['cues']}" if s["cues"] else "0 of 0"
    late = s["suppressed"].get("late", 0)
    fps = s["frames_analysed"] / s["footage_s"] if s["footage_s"] else 0.0
    lines = [
        "## Image to speech, in real time",
        "",
        f"{len(data['clips'])} clips ({s['footage_s'] / 60:.1f} minutes of footage) played at "
        "1x through the real pipeline: reader thread, pose thread, coach brain, voice and an "
        "audio output stream"
        + ("." if data["environment"].get("audible") else ", fed silence so nothing is heard."),
        "",
    ]
    if not s["cues"]:
        return "\n".join([*lines, "No cue was spoken."])
    device = s.get("to_device")
    if device is None:  # results from before the split was recorded
        device = device_split(
            [c for clip in data["clips"] for c in clip["cues"]], s["output_latency_ms"] / 1000
        )
        s = {**s, **device}
        device = s["to_device"]
    lines += [
        f"**{s['cues']} cues spoken. Median {device['p50']:.0f} ms from the frame that "
        f"triggered a cue to its line being handed to the audio device; 95th percentile "
        f"{device['p95']:.0f} ms; slowest {device['max']:.0f} ms.** That is the coach's own "
        "work: pose, hold contacts, the brain, the arbiter and the voice picking and stitching "
        "the line.",
        "",
        f"The audio device then adds its output buffer, which depends on the sound card and "
        f"driver rather than on this code: {s['output_latency_ms']:.0f} ms here, as PortAudio "
        f"reports it. With it, a cue reaches the output a median {total['p50']:.0f} ms after "
        f"its frame (95th percentile {total['p95']:.0f} ms, slowest {total['max']:.0f} ms), "
        f"and {share} cues are inside the {s['target_ms']:.0f} ms target.",
        "",
        table(
            ["Milliseconds", *STAT_HEADERS],
            [
                stat_row(
                    "Frame captured → event published (pose, contacts, brain)",
                    s["capture_to_event"],
                ),
                stat_row(
                    "Event published → line handed to the audio device (arbiter, voice)",
                    s["event_to_device"],
                ),
                stat_row("**Frame captured → line handed to the audio device**", device),
                stat_row("Total, with the device's output buffer", total),
            ],
        ),  # fmt: skip
        "",
        table(
            ["Per frame, milliseconds", *STAT_HEADERS],
            [stat_row("Frame captured → pose and contacts published", s["frame_to_state"])],
        ),
        "",
        f"The pose thread analysed {s['frames_analysed']} of {s['frames_total']} frames "
        f"({fps:.1f} per second on average); the 1-slot queues drop the rest rather than fall "
        "behind.",
        "",
        f"The brain raised {s['events']} events; the arbiter let {s['cues']} be spoken. "
        f"Turned down: {reasons_text(s['suppressed'])}. "
        f"{late} cue{'s were' if late != 1 else ' was'} dropped for missing the 200 ms start "
        "deadline.",
        "",
        "### By cue",
        "",
        table(
            ["Cue", *STAT_HEADERS],
            [stat_row(f"`{kind}`", stats) for kind, stats in s["by_kind"].items()],
        ),
    ]
    return "\n".join(lines)


def reasons_text(suppressed: dict) -> str:
    if not suppressed:
        return "none"
    return ", ".join(f"{n} {reason}" for reason, n in sorted(suppressed.items()))


def counts_text(counts: dict) -> str:
    return ", ".join(f"{kind} ×{n}" for kind, n in sorted(counts.items())) or "–"


def checks_report(data: dict) -> str:
    s = data["summary"]
    rows = []
    for clip in data["clips"]:
        check = clip["check"]
        if not check:
            continue
        outcomes = ", ".join(a["outcome"] for a in clip["attempts"]) or "–"
        rows.append(
            [
                f"`{clip['clip']}`",
                check["expected"],
                "yes" if check["ok"] else "**no**",
                outcomes,
                counts_text(clip["spoken"]),
            ]
        )
    other = [
        [f"`{c['clip']}`", ", ".join(a["outcome"] for a in c["attempts"]) or "–",
         counts_text(c["spoken"])]
        for c in data["clips"]
        if not c["check"]
    ]  # fmt: skip
    lines = [
        "## Did each test clip show what it was filmed to show?",
        "",
        "Each numbered clip was filmed to provoke one thing (`TEST_CLIMBS.md`). This checks the "
        "brain's events in the same real-time run, frame drops included. It is a smoke test on "
        "the team's own footage, one clip per scenario, not an accuracy figure.",
        "",
        f"**{s['checks_passed']} of {s['checks_total']} clips did.**",
        "",
        table(["Clip", "Filmed to show", "Seen", "Attempts ended", "Spoken"], rows),
    ]
    if other:
        lines += [
            "",
            "Clips with no single expected event (demo sessions and hard cases for perception):",
            "",
            table(["Clip", "Attempts ended", "Spoken"], other),
        ]
    return "\n".join(lines)


def stages_report(data: dict) -> str:
    s = data["summary"]
    frames = sum(clip["frames"] for clip in data["clips"])
    frame = s["frame"]
    lines = [
        "## Where the time goes, stage by stage",
        "",
        f"{frames} frames from {len(data['clips'])} clips, every frame, one stage at a time on "
        "one thread, with no pacing and no queues. This is compute cost only.",
        "",
        table(
            ["Milliseconds per frame", *STAT_HEADERS],
            [
                *[stat_row(STAGE_LABELS[stage], s[stage]) for stage in STAGES if stage != "voice"],
                stat_row("**Whole frame, decode to brain**", frame),
            ],
        ),
        "",
        table(
            ["Milliseconds per spoken cue", *STAT_HEADERS],
            [
                stat_row(STAGE_LABELS["voice"], s["voice"]),
                stat_row("**Whole frame on which a cue was spoken**", s["cue_frame"]),
            ],
        ),
        "",
        f"One thread sustains {s['throughput_fps']:.1f} frames per second end to end. "
        f"Pose inference is {share(s['pose'], frame):.0f}% of a frame's cost; the coach brain "
        f"is {share(s['brain'], frame):.1f}%.",
    ]
    return "\n".join(lines)


def share(part: dict, whole: dict) -> float:
    return 100 * part["mean"] / whole["mean"] if whole.get("mean") else 0.0


def report(results: Sequence[dict]) -> str:
    by_kind = {r["kind"]: r for r in results}
    env = next(iter(by_kind.values()))["environment"]
    parts = ["# Benchmarks", "", *environment_lines(env)]
    if "e2e" in by_kind:
        parts += ["", e2e_report(by_kind["e2e"])]
    if "stages" in by_kind:
        parts += ["", stages_report(by_kind["stages"])]
    if "e2e" in by_kind:
        parts += ["", checks_report(by_kind["e2e"])]
    return "\n".join(parts) + "\n"


# ---- CLI ------------------------------------------------------------------------------------


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    log.info("wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stages", "e2e"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", type=Path, required=True, help="result file (JSON)")
        cmd.add_argument("--clips", nargs="+", default=(), metavar="NAME", help="name fragments")
        cmd.add_argument("--videos", type=Path, default=VIDEO_DIR)
    sub.choices["stages"].add_argument("--max-frames", type=int, help="per clip")
    sub.choices["e2e"].add_argument("--audible", action="store_true", help="let the coach speak")
    rep = sub.add_parser("report")
    rep.add_argument("results", nargs="+", type=Path, help="result files from stages and e2e")
    args = parser.parse_args(argv)

    os.chdir(ROOT)  # the phrase bank, persona and audio cache are found relative to the repo
    if args.command == "report":
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(report(results))
        return 0

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    clips = find_clips(args.videos, only=args.clips)
    if not clips:
        parser.error(f"no clips with a hold map under {args.videos}")
    if args.command == "stages":
        write_json(args.out, run_stages(clips, cfg, args.max_frames))
    else:
        write_json(args.out, asyncio.run(run_e2e(clips, cfg, args.audible)))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    raise SystemExit(main())
