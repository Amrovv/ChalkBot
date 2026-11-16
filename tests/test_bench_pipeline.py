"""scripts/bench_pipeline.py: its statistics and cue matching, and both harnesses run on fakes.

The real measurements need footage, the pose model and an audio device, so they are run by
hand (docs/BENCHMARKS.md). These tests make sure the harness measures what it says it does.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from coach.brain import Brain
from coach.bus import Bus
from coach.contracts import (
    AttemptSummary,
    ClimberState,
    CoachEvent,
    SpeakRequest,
    SpokenLine,
    Suppressed,
)
from coach.sources import LatestQueue
from scripts import bench_pipeline as bench
from tests.test_pipeline import FakeVoice, _synthetic_climb
from tests.test_pose import make_pose

BRAIN_BUDGET_MS = 5.0  # per frame; a 30 fps frame is 33 ms and pose inference takes most of it


def real_config() -> dict:
    return yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))


def event(t: float, kind: str = "feet_low", priority: int = 1) -> CoachEvent:
    return CoachEvent(t=t, kind=kind, priority=priority, ttl_s=1.2)


# ---- Statistics -----------------------------------------------------------------------------


def test_summarise_reports_milliseconds():
    stats = bench.summarise([0.010, 0.020, 0.030, 0.040])
    assert stats == {"n": 4, "mean": 25.0, "p50": 25.0, "p95": 38.5, "p99": 39.7, "max": 40.0}


def test_summarise_of_nothing_says_so_instead_of_raising():
    assert bench.summarise([]) == {"n": 0}
    assert bench.stat_row("Voice", {"n": 0}) == ["Voice", "–", "–", "–", "–", "0"]


# ---- Clips ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "problem", "expected"),
    [
        ("climb_3_07_stall", "climb_3", ("07", None)),
        ("climb_6_12-1_new_high_point", "climb_6", ("12", "12-1")),
        ("climb_12_demo_session_a", "climb_12", (None, None)),
        ("climb_15_demo", "climb_15", (None, None)),
    ],
)
def test_clip_number_reads_the_test_number_and_session_part(stem, problem, expected):
    assert bench.clip_number(stem, problem) == expected


def test_find_clips_keeps_clips_with_a_hold_map_and_groups_a_session_filmed_in_parts(tmp_path):
    names = [
        "climb_1_00_empty_wall.mp4",  # the hold map is built from it: nothing to coach
        "climb_1_03_feet_low.mp4",
        "climb_6_12-1_new_high_point.mp4",
        "climb_6_12-2_new_high_point.mp4",
        "climb_12_demo_session_a.mp4",
        "climb_7_13_fall.mp4",  # no hold map for climb_7
        "holiday.mp4",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    known = {"climb_1", "climb_6", "climb_12"}

    clips = bench.find_clips(tmp_path, known)

    assert [(c.name, c.problem, c.number) for c in clips] == [
        ("climb_1_03_feet_low.mp4", "climb_1", "03"),
        ("climb_6_12-1_new_high_point.mp4", "climb_6", "12"),
        ("climb_6_12-2_new_high_point.mp4", "climb_6", "12"),
        ("climb_12_demo_session_a.mp4", "climb_12", None),
    ]
    assert [[c.name for c in group] for group in bench.sessions_of(clips)] == [
        ["climb_1_03_feet_low.mp4"],
        ["climb_6_12-1_new_high_point.mp4", "climb_6_12-2_new_high_point.mp4"],
        ["climb_12_demo_session_a.mp4"],
    ]
    assert [c.opens_session for c in clips] == [False, True, False, False]
    only = bench.find_clips(tmp_path, known, only=["demo"])
    assert [c.name for c in only] == ["climb_12_demo_session_a.mp4"]


# ---- Cue timing -----------------------------------------------------------------------------


def test_a_spoken_cue_is_timed_from_its_frame_through_its_event_to_its_audio():
    records = [
        (10.05, "event", event(10.0)),
        (10.05, "event", event(10.0, "arms_bent")),  # turned down: never spoken, never timed
        (10.06, "spoken", SpokenLine(10.0, 10.15, "feet_low", "Feet up.")),
    ]
    (cue,) = bench.cue_timings(records)
    assert cue.kind == "feet_low"
    assert cue.capture_to_event == pytest.approx(0.05)
    assert cue.event_to_audio == pytest.approx(0.10)
    assert cue.total == pytest.approx(0.15)


def test_a_spoken_line_with_no_event_on_record_still_has_its_total():
    (cue,) = bench.cue_timings([(11.0, "spoken", SpokenLine(9.0, 9.5, "top", "Topped it."))])
    assert (cue.capture_to_event, cue.event_to_audio) == (None, None)
    assert cue.total == pytest.approx(0.5)


def test_two_cues_of_one_kind_are_matched_by_their_capture_time():
    records = [
        (1.02, "event", event(1.0)),
        (5.04, "event", event(5.0)),
        (5.05, "spoken", SpokenLine(5.0, 5.2, "feet_low", "Feet up.")),
    ]
    (cue,) = bench.cue_timings(records)
    assert cue.capture_to_event == pytest.approx(0.04)


# ---- Scenario checks ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "events", "outcomes", "ok"),
    [
        ("07", {"stalling": 2}, [], True),
        ("07", {"feet_low": 1}, [], False),
        ("01", {}, [], True),  # standing at the base: no attempt may start
        ("01", {"attempt_start": 1}, [], False),
        ("08", {"quiet_feet": 1}, ["top"], False),  # clean footwork must not read as fidgeting
        ("14", {}, ["downclimb"], True),
        ("14", {"fall": 1}, ["fall"], False),
        ("18", {"attempt_start": 2}, ["fall"], True),  # the second need not have ended yet
        ("18", {"attempt_start": 1}, ["top"], False),
    ],
)
def test_scenario_check_compares_a_clip_with_what_it_was_filmed_to_show(
    number, events, outcomes, ok
):
    assert bench.scenario_check(number, events, outcomes)["ok"] is ok


def test_a_clip_with_no_expectation_has_no_check():
    assert bench.scenario_check(None, {"top": 1}, ["top"]) is None
    assert bench.scenario_check("25", {}, []) is None  # a hard case for perception


def test_every_expected_event_is_a_real_event_kind_or_outcome():
    from coach.contracts import EVENT_KINDS

    for rule, want in bench.EXPECTED.values():
        if rule in ("event", "no_event"):
            assert want in EVENT_KINDS
        elif rule == "outcome":
            assert want in ("top", "fall", "downclimb")


# ---- The probe ------------------------------------------------------------------------------


def test_probe_stamps_each_publish_and_still_delivers_it():
    bus = Bus()
    delivered = bus.subscribe("event")
    now = [50.0]
    probe = bench.Probe(bus, clock=lambda: now[0])

    probe.publish("state", ClimberState(49.96, 0, True, {}, {}, {}, (0.5, 0.5)))
    probe.publish("frame", b"jpeg")  # not kept: the video is not part of the measurement
    now[0] = 50.01
    first = event(49.96)
    probe.publish("event", first)

    assert delivered.get_nowait() is first
    records, delays = probe.take()
    assert records == [(50.01, "event", first)]
    assert delays == pytest.approx([0.04])
    assert probe.take() == ([], [])  # a clip starts from nothing
    assert probe.subscribe("spoken") is not None  # everything else is the bus's own


# ---- Results and report ---------------------------------------------------------------------


def one_clip_result() -> dict:
    clip = bench.Clip(Path("climb_3_07_stall.mp4"), "climb_3", "07", "climb_3_07_stall")
    stall = event(10.0, "stalling")
    records = [
        (9.0, "event", event(8.97, "attempt_start", 0)),
        (9.0, "spoken", SpokenLine(8.97, 9.1, "attempt_start", "Off you go.")),
        (10.04, "event", stall),
        (10.04, "spoken", SpokenLine(10.0, 10.9, "stalling", "Move.")),  # 900 ms: over target
        (11.0, "suppressed", Suppressed(event(11.0, "arms_bent"), "cooldown")),
        (12.0, "attempt", AttemptSummary(1, 3.0, 6, "fall", [], ["stalling"], [])),
    ]
    info = {"frames": 120, "fps": 30.0, "duration_s": 4.0, "size": [1080, 1920]}
    return bench.clip_result(clip, info, records, [0.03] * 100)


def test_clip_result_counts_what_the_coach_did_on_a_clip():
    result = one_clip_result()
    assert result["frames_analysed"] == 100 and result["analysed_fps"] == 25.0
    assert result["events"] == {"attempt_start": 1, "stalling": 1}
    assert result["spoken"] == {"attempt_start": 1, "stalling": 1}
    assert result["suppressed"] == {"cooldown": 1}
    assert result["attempts"] == [{"no": 1, "outcome": "fall", "high_hold": 6}]
    assert result["check"] == {"expected": "`stalling` detected", "ok": True}
    assert [cue["total_ms"] for cue in result["cues"]] == pytest.approx([130.0, 900.0])


def test_the_first_part_of_a_session_sets_the_scene_and_is_not_checked():
    opener = bench.Clip(Path("climb_6_12-1_x.mp4"), "climb_6", "12", "climb_6/12", True)
    info = {"frames": 30, "fps": 30.0, "duration_s": 1.0, "size": [6, 4]}
    assert bench.clip_result(opener, info, [], [])["check"] is None


def test_summary_counts_cues_inside_the_target_and_splits_them_by_kind():
    summary = bench.e2e_summary([one_clip_result()], output_latency_s=0.093)
    assert summary["cues"] == 2 and summary["within_target"] == 1
    assert summary["total"]["max"] == pytest.approx(900.0)
    assert set(summary["by_kind"]) == {"attempt_start", "stalling"}
    assert summary["frame_to_state"]["p50"] == pytest.approx(30.0)
    assert summary["output_latency_ms"] == 93.0
    assert (summary["checks_passed"], summary["checks_total"]) == (1, 1)
    assert (summary["frames_analysed"], summary["frames_total"]) == (100, 120)


def test_the_audio_device_buffer_is_split_off_the_total():
    # The speaker stamps audio start as play() time plus the stream's fixed output latency,
    # so taking that latency off leaves when the line was handed to the device.
    summary = bench.e2e_summary([one_clip_result()], output_latency_s=0.093)
    assert summary["to_device"]["max"] == pytest.approx(900.0 - 93.0)
    assert summary["to_device"]["n"] == summary["total"]["n"]
    stalling = one_clip_result()["cues"]
    expected = [
        c["event_to_audio_ms"] - 93.0 for c in stalling if c["event_to_audio_ms"] is not None
    ]
    assert summary["event_to_device"]["mean"] == pytest.approx(sum(expected) / len(expected))


def test_report_on_older_results_derives_the_split_from_the_cues():
    clip = one_clip_result()
    summary = bench.e2e_summary([clip], 0.093)
    del summary["to_device"], summary["event_to_device"]
    e2e = {
        "kind": "e2e",
        "environment": {**bench.environment(), "audible": False},
        "summary": summary,
        "clips": [clip],
    }
    assert "line being handed to the audio device" in bench.report([e2e])


def test_report_states_the_headline_figure_and_lists_the_checks():
    clip = one_clip_result()
    e2e = {
        "kind": "e2e",
        "environment": {**bench.environment(), "audible": False},
        "summary": bench.e2e_summary([clip], 0.093),
        "clips": [clip],
    }
    text = bench.report([e2e])
    assert text.startswith("# Benchmarks")
    assert "2 cues spoken" in text and "1 of 2 cues are inside the 800 ms target" in text
    assert "line being handed to the audio device" in text
    assert "| `climb_3_07_stall.mp4` | `stalling` detected | yes |" in text
    assert "1 cooldown" in text


# ---- Stage benchmark on a synthetic clip ----------------------------------------------------


@pytest.fixture
def clip(tmp_path) -> Path:
    path = tmp_path / "wall.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    for i in range(20):
        writer.write(np.full((24, 32, 3), i * 10, np.uint8))
    writer.release()
    return path


class StandThenClimb:
    """A pose detector: the person stands for three frames, then is 0.2 higher up the wall."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame, timestamp_ms: int):
        self.calls += 1
        lift = 0.2 if self.calls > 3 else 0.0
        return [[(x, y - lift, v) for x, y, v in make_pose()]]


def test_stage_benchmark_times_every_stage_of_every_frame(clip):
    voices = []

    def make_voice(cfg, bus, clock):
        voices.append(FakeVoice())
        return voices[-1]

    detector = StandThenClimb()
    result = bench.stage_clip(
        clip, "no-such-problem", real_config(), detector=detector, make_voice=make_voice
    )

    assert result["frames"] == detector.calls == 20
    for stage in ("decode", "pose", "contacts", "brain"):
        assert len(result["samples_ms"][stage]) == 20
        assert all(value >= 0 for value in result["samples_ms"][stage])
    # With no hold map the attempt starts from the hip staying raised: one line was spoken,
    # so the voice stage and the "frame a cue was spoken on" total each have one sample.
    assert [req.event.kind for req in voices[0].spoken] == ["attempt_start"]
    assert len(result["samples_ms"]["voice"]) == len(result["cue_frame_total_ms"]) == 1

    summary = bench.stage_summary([result])
    assert summary["frame"]["n"] == 20 and summary["throughput_fps"] > 0
    assert "Pose inference" in bench.stages_report(
        {"summary": summary, "clips": [{"clip": clip.name, "frames": 20}]}
    )


def test_stage_benchmark_stops_at_max_frames(clip):
    result = bench.stage_clip(
        clip,
        "no-such-problem",
        real_config(),
        detector=StandThenClimb(),
        make_voice=lambda cfg, bus, clock: FakeVoice(),
        max_frames=5,
    )
    assert result["frames"] == 5


# ---- End-to-end harness on fakes ------------------------------------------------------------


class PacedSource:
    """Five frames 10 ms apart, stamped with the real clock like a camera or a paced file."""

    def __iter__(self):
        for i in range(5):
            time.sleep(0.01)
            yield time.monotonic(), i, np.zeros((4, 6, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class EchoTracker:
    def process(self, t: float, idx: int, img) -> ClimberState:
        return ClimberState(t, idx, True, {}, {}, {}, (0.5, 0.5))

    def close(self) -> None:
        pass


def no_holdmap(problem_id: str):
    raise FileNotFoundError(problem_id)


class BusVoice:
    """Publishes what it says like the real Voice, 100 ms after the frame."""

    def __init__(self, bus) -> None:
        self.bus = bus
        self.stops = 0

    def say(self, req: SpeakRequest) -> bool:
        line = SpokenLine(req.event.t, req.event.t + 0.1, req.event.kind, "Feet up.")
        self.bus.publish("spoken", line)
        return True

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        self.stops += 1


class OneCueBrain:
    """Speaks one cue on the first state it ever sees, then stays quiet."""

    def __init__(self, cfg, bus, voice) -> None:
        self.bus, self.voice, self.spoke = bus, voice, False

    async def run(self, states) -> None:
        while True:
            state = await states.get()
            if not self.spoke:
                self.spoke = True
                cue = event(state.t)
                self.bus.publish("event", cue)
                self.voice.say(SpeakRequest(cue, 0, "neutral", state.t + 0.2))


async def test_e2e_harness_plays_a_session_through_the_real_threads_and_times_its_cue():
    bus = Bus()
    bus.bind_loop()
    probe = bench.Probe(bus)
    voice = BusVoice(probe)
    parts = SimpleNamespace(
        open_source=lambda spec, **kwargs: PacedSource(),
        make_tracker=lambda holdmap, shapes: EchoTracker(),
        load_holdmap=no_holdmap,
        frame_queue=LatestQueue,
    )
    info = {"frames": 5, "fps": 100.0, "duration_s": 0.05, "size": [6, 4]}
    rig = bench.Rig(
        {},
        probe,
        voice,
        OneCueBrain,
        parts=parts,
        info=lambda path: dict(info),
        quiet_settle_s=0.02,
        quiet_timeout_s=2.0,
    )
    session = [
        bench.Clip(Path("climb_6_12-1_x.mp4"), "climb_6", "12", "climb_6/12"),
        bench.Clip(Path("climb_6_12-2_x.mp4"), "climb_6", "12", "climb_6/12"),
    ]

    await bench.run_session(rig, session)

    first, second = rig.results
    assert first["clip"] == "climb_6_12-1_x.mp4" and 1 <= first["frames_analysed"] <= 5
    assert first["spoken"] == {"feet_low": 1}
    (cue,) = first["cues"]
    assert cue["total_ms"] == pytest.approx(100.0)
    assert 0 <= cue["capture_to_event_ms"] < 1000  # the frame really crossed two threads
    # One brain for the whole session: it had already spoken, so the second part is silent
    # and its records are its own (a fresh brain per clip would have spoken again).
    assert second["spoken"] == {} and second["cues"] == [] and second["frames_analysed"] >= 1
    assert voice.stops == 1


async def test_a_clip_is_over_only_once_the_coach_has_gone_quiet():
    class Voice:
        polls = 0

        def is_speaking(self) -> bool:
            Voice.polls += 1
            return Voice.polls <= 3  # a debrief still running when the clip ends

    start = time.monotonic()
    await bench.wait_until_quiet(Voice(), settle_s=0.0, timeout_s=5.0)
    assert Voice.polls == 4 and time.monotonic() - start < 2.0


async def test_a_coach_that_never_goes_quiet_does_not_hang_the_benchmark():
    never_quiet = SimpleNamespace(is_speaking=lambda: True)
    start = time.monotonic()
    await bench.wait_until_quiet(never_quiet, settle_s=0.0, timeout_s=0.1)
    assert time.monotonic() - start < 2.0


def test_mute_feeds_the_real_stream_silence_of_the_same_length():
    played = []
    speaker = SimpleNamespace(play=lambda samples: played.append(samples) or 1.5)
    voice = SimpleNamespace(_speaker=speaker)

    bench.mute(voice)
    start_time = speaker.play(np.ones(240, dtype=np.float32))

    assert start_time == 1.5  # the stream's own start time comes back unchanged
    assert played[0].shape == (240,) and not played[0].any()


# ---- A budget for the code the team wrote -----------------------------------------------------


def test_the_coach_brain_costs_a_small_fraction_of_a_frame():
    """Features, nine detectors, memory and the arbiter run on every frame, on the event loop.
    Measured at about 0.1 ms; the budget is far above that so a slow CI machine passes, and
    far below a frame so a detector that starts doing real work per frame is noticed."""
    brain = Brain(real_config(), Bus(), FakeVoice())
    states = [state for _ in range(20) for state in _synthetic_climb()]
    costs = []
    for n, state in enumerate(states):
        state.t = n * 0.5  # one continuous session, not the same nine seconds repeated
        start = time.perf_counter()
        brain._process_state(state)
        costs.append((time.perf_counter() - start) * 1000)
    assert statistics.median(costs) < BRAIN_BUDGET_MS
