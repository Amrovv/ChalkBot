# Benchmarks

- **Machine:** Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz, 6 logical cores, Windows-11-10.0.26300-SP0
- **Software:** Python 3.12.13; mediapipe 1.2.2, opencv-contrib-python 5.0.0.93, numpy 2.3.5, kokoro 0.9.4, sounddevice 0.5.6
- **Timer:** `time.monotonic()`, 15.625 ms per tick: no latency below is finer than that
- **Run on:** 2026-11-16

## Image to speech, in real time

32 clips (16.1 minutes of footage) played at 1x through the real pipeline: reader thread, pose thread, coach brain, voice and an audio output stream, fed silence so nothing is heard.

**164 cues spoken. Median 47 ms from the frame that triggered a cue to its line being handed to the audio device; 95th percentile 78 ms; slowest 93 ms.** That is the coach's own work: pose, hold contacts, the brain, the arbiter and the voice picking and stitching the line.

The audio device then adds its output buffer, which depends on the sound card and driver rather than on this code: 93 ms here, as PortAudio reports it. With it, a cue reaches the output a median 140 ms after its frame (95th percentile 171 ms, slowest 186 ms), and 164 of 164 cues are inside the 800 ms target.

| Milliseconds | median | mean | p95 | max | n |
|---|---|---|---|---|---|
| Frame captured → event published (pose, contacts, brain) | 47.0 | 43.0 | 63.0 | 78.0 | 164 |
| Event published → line handed to the audio device (arbiter, voice) | 0.0 | 5.2 | 16.0 | 32.0 | 164 |
| **Frame captured → line handed to the audio device** | 47.0 | 48.2 | 78.0 | 93.0 | 164 |
| Total, with the device's output buffer | 140.3 | 141.5 | 171.3 | 186.3 | 164 |

| Per frame, milliseconds | median | mean | p95 | max | n |
|---|---|---|---|---|---|
| Frame captured → pose and contacts published | 47.0 | 43.9 | 63.0 | 219.0 | 27445 |

The pose thread analysed 27445 of 31475 frames (28.4 per second on average); the 1-slot queues drop the rest rather than fall behind.

The brain raised 356 events; the arbiter let 164 be spoken. Turned down: 29 budget, 153 busy, 3 cooldown, 5 late, 2 lower_priority. 5 cues were dropped for missing the 200 ms start deadline.

### By cue

| Cue | median | mean | p95 | max | n |
|---|---|---|---|---|---|
| `arms_bent` | 156.3 | 156.3 | 156.3 | 156.3 | 1 |
| `attempt_start` | 140.3 | 145.0 | 171.3 | 186.3 | 36 |
| `crux_warning` | 140.3 | 140.3 | 140.3 | 140.3 | 1 |
| `fall` | 171.3 | 161.0 | 171.3 | 171.3 | 3 |
| `feet_cut` | 132.8 | 132.6 | 140.3 | 140.3 | 4 |
| `feet_low` | 140.3 | 137.4 | 155.9 | 156.3 | 29 |
| `hand_searching` | 139.8 | 137.1 | 156.3 | 156.3 | 20 |
| `high_point` | 155.3 | 155.3 | 155.3 | 155.3 | 2 |
| `quiet_feet` | 140.3 | 141.0 | 171.3 | 171.3 | 18 |
| `route_preview` | 140.3 | 141.0 | 171.3 | 171.3 | 20 |
| `stalling` | 148.3 | 148.3 | 155.5 | 156.3 | 2 |
| `told_you_so` | 140.3 | 142.5 | 156.0 | 156.3 | 7 |
| `top` | 140.3 | 144.5 | 171.3 | 171.3 | 11 |
| `weight_shift` | 140.3 | 140.1 | 155.3 | 155.3 | 10 |

## Where the time goes, stage by stage

31475 frames from 32 clips, every frame, one stage at a time on one thread, with no pacing and no queues. This is compute cost only.

| Milliseconds per frame | median | mean | p95 | max | n |
|---|---|---|---|---|---|
| Decode the frame (OpenCV) | 4.0 | 4.2 | 5.2 | 49.3 | 31475 |
| Pose inference (MediaPipe, incl. resize) | 30.9 | 30.9 | 33.6 | 163.8 | 31475 |
| Smoothing, left/right fix, hold contacts | 0.3 | 0.3 | 0.5 | 6.7 | 31475 |
| Coach brain (features, 9 detectors, memory, arbiter) | 0.1 | 0.1 | 0.2 | 3.6 | 31475 |
| **Whole frame, decode to brain** | 35.3 | 35.5 | 38.7 | 169.9 | 31475 |

| Milliseconds per spoken cue | median | mean | p95 | max | n |
|---|---|---|---|---|---|
| Voice: pick line, load and stitch clips | 0.7 | 1.6 | 8.2 | 33.7 | 204 |
| **Whole frame on which a cue was spoken** | 36.3 | 37.9 | 45.9 | 103.6 | 204 |

One thread sustains 28.2 frames per second end to end. Pose inference is 87% of a frame's cost; the coach brain is 0.4%.

## Did each test clip show what it was filmed to show?

Each numbered clip was filmed to provoke one thing (`TEST_CLIMBS.md`). This checks the brain's events in the same real-time run, frame drops included. It is a smoke test on the team's own footage, one clip per scenario, not an accuracy figure.

**14 of 22 clips did.**

| Clip | Filmed to show | Seen | Attempts ended | Spoken |
|---|---|---|---|---|
| `climb_1_01_standing_at_base.mp4` | no `attempt_start` | yes | – | route_preview ×1 |
| `climb_1_02_clean_send_steady_pace.mp4` | an attempt ends "top" | yes | top | attempt_start ×1, feet_low ×1, route_preview ×1, top ×1 |
| `climb_1_03_clip_feet_low.mp4` | `feet_low` detected | **no** | top | attempt_start ×1, quiet_feet ×1, route_preview ×1, top ×1 |
| `climb_1_04_bent_arms.mp4` | `arms_bent` detected | **no** | top | attempt_start ×1, top ×1, weight_shift ×2 |
| `climb_2_05_hand_searching.mp4` | `hand_searching` detected | yes | top | attempt_start ×1, feet_low ×2, hand_searching ×1, told_you_so ×1, top ×1 |
| `climb_3_06_feet_cut.mp4` | `feet_cut` detected | yes | top | attempt_start ×1, feet_low ×1, top ×1 |
| `climb_3_07_stall.mp4` | `stalling` detected | yes | downclimb | attempt_start ×1, hand_searching ×1, stalling ×2 |
| `climb_4_08_quiet_feet.mp4` | no `quiet_feet` | **no** | – | attempt_start ×1, feet_low ×2, hand_searching ×1, route_preview ×1, weight_shift ×1 |
| `climb_4_09_noisy_feet (1).mp4` | `quiet_feet` detected | yes | fall | attempt_start ×1, fall ×1, feet_low ×3, told_you_so ×2 |
| `climb_4_09_noisy_feet.mp4` | `quiet_feet` detected | yes | – | attempt_start ×1, feet_low ×1, quiet_feet ×2, route_preview ×1, weight_shift ×1 |
| `climb_5_10_weight_not_shifted.mp4` | `weight_shift` detected | yes | – | attempt_start ×1, feet_low ×2, hand_searching ×2, quiet_feet ×1, weight_shift ×2 |
| `climb_5_11_big_reach_coming_up.mp4` | `route_preview` detected | **no** | top | attempt_start ×1, feet_low ×2, top ×1 |
| `climb_6_12-2_new_high_point.mp4` | `high_point` detected | yes | fall | attempt_start ×1, hand_searching ×1, high_point ×1, route_preview ×1 |
| `climb_6_12-3_new_high_point.mp4` | `high_point` detected | yes | top | attempt_start ×1, high_point ×1, route_preview ×1, top ×1 |
| `climb_7_13_fall_at_the_crux.mp4` | `fall` detected | **no** | downclimb, downclimb | attempt_start ×3, feet_low ×1, hand_searching ×1, quiet_feet ×2 |
| `climb_8_14_downclimb.mp4` | an attempt ends "downclimb" | **no** | – | attempt_start ×1, feet_low ×2, hand_searching ×1, quiet_feet ×1, told_you_so ×1 |
| `climb_8_15_jump_off.mp4` | an attempt ends "fall" | **no** | – | attempt_start ×1, hand_searching ×1 |
| `climb_9_16_told_you_so_weight_not_shifted.mp4` | `weight_shift` detected | **no** | – | attempt_start ×1, feet_low ×1, quiet_feet ×2, route_preview ×1 |
| `climb_9_17_follow_the_advice_weight_not_shifted.mp4` | `weight_shift` detected | yes | top | attempt_start ×1, hand_searching ×1, quiet_feet ×1, route_preview ×1, top ×1, weight_shift ×1 |
| `climb_9_18_rest_between_attempts (1).mp4` | at least 2 attempts start | yes | downclimb | attempt_start ×2, hand_searching ×2, quiet_feet ×1, route_preview ×1 |
| `climb_9_18_rest_between_attempts.mp4` | at least 2 attempts start | yes | downclimb, downclimb | attempt_start ×2, feet_cut ×1, hand_searching ×1, route_preview ×1 |
| `climb_14_04_arms_bent.mp4` | `arms_bent` detected | yes | top | arms_bent ×1, attempt_start ×1, feet_low ×2, hand_searching ×2, route_preview ×1, told_you_so ×1, top ×1, weight_shift ×1 |

Clips with no single expected event (demo sessions and hard cases for perception):

| Clip | Attempts ended | Spoken |
|---|---|---|
| `climb_6_12-1_new_high_point.mp4` | downclimb | attempt_start ×1, hand_searching ×1 |
| `climb_9_20_person_walks_infront_of_camera.mp4` | top | attempt_start ×1, feet_low ×1, quiet_feet ×2, route_preview ×1, told_you_so ×1, top ×1 |
| `climb_9_21_spotter_standing_close.mp4` | fall | attempt_start ×1, fall ×1, hand_searching ×1, route_preview ×1 |
| `climb_9_22_climber_partly_out_of_frame.mp4` | – | route_preview ×1 |
| `climb_10_25_clothing_same_colour_as_holds.mp4` | – | attempt_start ×1, feet_low ×4, route_preview ×1 |
| `climb_11_28_fast_dynamic_move.mp4` | – | attempt_start ×1, feet_cut ×1, feet_low ×1, quiet_feet ×1, route_preview ×1 |
| `climb_12_demo_session_a.mp4` | downclimb, downclimb | attempt_start ×3, crux_warning ×1, hand_searching ×1, quiet_feet ×3, route_preview ×1 |
| `climb_13_demo_session_b.mp4` | downclimb, top | attempt_start ×2, feet_cut ×1, feet_low ×3, quiet_feet ×1, route_preview ×1, told_you_so ×1, top ×1, weight_shift ×1 |
| `climb_13_demo_session_c.mp4` | – | route_preview ×1 |
| `climb_15_demo.mp4` | fall | attempt_start ×1, fall ×1, feet_cut ×1, hand_searching ×2, weight_shift ×1 |

## Misses this run shows

Reported as measured. No threshold was tuned to make a clip pass.

- `climb_1_03_clip_feet_low`: no `feet_low`, although the same detector fires on most other clips.
- `climb_1_04_bent_arms`: no `arms_bent`. It does fire on `climb_14_04_arms_bent`.
- `climb_4_08_quiet_feet`: filmed as careful footwork, but the brain raised `quiet_feet` anyway
  (a false positive; the arbiter happened not to speak it).
- `climb_5_11_big_reach_coming_up`: no `route_preview` during the attempt.
- `climb_7_13_fall_at_the_crux`: the falls were classed as downclimbs.
- `climb_8_14_downclimb` and `climb_8_15_jump_off`: the attempt never closed before the clip
  ended, so there is no outcome to check. Several demo and hard-case clips show the same thing.
- `climb_9_16_told_you_so_weight_not_shifted`: no `weight_shift`.

## What this measures and what it does not

- **Two latency figures.** Both start from the timestamp of the frame that triggered a cue,
  taken right after decode. The headline one ends when the line is handed to the audio device
  (`Speaker.play()`). That is the coach's own work, and it does not depend on the sound
  hardware. The total adds the output buffer the audio device reports (PortAudio's
  `latency`, 93 ms on this machine with Windows MME at `latency="low"`). That buffer depends
  on the sound card, driver and host API, so it differs between machines. Neither figure is
  measured with a microphone.
- **Under one timer tick.** Handing an event to the audio device (arbiter, line choice,
  stitching) usually takes less than one 15.6 ms tick, so its median reads 0. The stage
  benchmark times the voice's part directly: 0.7 ms median.
- **Not included:** camera sensor and transport delay (the footage is files played at 1x), and
  anything after the sound card, such as a wireless headset's own delay.
- **Not included:** the detectors' debounce. A condition has to hold for about 400 ms before
  an event fires. That is by design, so a cue does not fire on a single noisy frame, but it
  adds to what a climber feels and is not in the figures above.
- **Timer resolution** is 15.6 ms (`time.monotonic()` on Windows, Python 3.12). That is why
  many values cluster at 140.3 and 156.3 ms.
- **Only spoken cues are timed.** Cues dropped for missing the 200 ms start deadline (`late`)
  are counted above, not timed.
- **One machine**: i5-9600K, 6 cores, CPU-only pose inference. The footage is the team's own
  and was filmed for this project.
- **The scenario table is a smoke test** with one clip per scenario. It is not an accuracy
  figure, and should not be read as precision or recall.
