# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Primary audience for the web dashboard (`web/index.html`): hackathon judges watching the live demo on a
projector or big screen, reading it from across a room while someone climbs. The climber hears the
spoken cues but is not the dashboard's main reader. The team also uses its latency and timeline
panels while building, but that is secondary.

## Product Purpose
Climbing Coach is a real-time AI bouldering coach built for incident.io's "Live commentator"
hackathon challenge (24 h, team of 4). A fixed camera watches a climber. The system speaks short
coaching cues mid-climb and gives a persona-driven debrief between attempts, remembering earlier
attempts. Success means the live demo convinces judges of three things:
- **Low latency:** cues land in real time (target: cue audio within ~0.8 s), visibly in sync with the climb.
- **Useful coaching:** the cues and debriefs would help a climber improve, and they build on earlier attempts.
- **Reliability on stage:** it runs live without falling over.

## Positioning
Two speeds. A fully local fast path (MediaPipe pose -> limb/hold contacts -> cue detectors -> speech
arbiter -> pre-rendered audio) gives the live commentary, with no network calls in the loop. A slow
path (AttemptSummary -> Gemini -> Kokoro TTS) gives the reflective debrief between attempts. The
coach knows the actual problem: a hold map built from the empty wall, numbered bottom (1) to top (N).

## Operating Context
- Live demo: a climber on a bouldering wall, a fixed camera, the dashboard on a projector, the coach's voice from speakers.
- Dashboard: live video with pose/hold overlay, a large caption for the current cue or debrief, the coach's mood, cue latency, current attempt, attempt history and an event timeline.
- Dashboard is fed over a WebSocket (`/ws`, one JSON message per bus topic) plus a video stream (`/video` MJPEG or WebSocket JPEG frames). `?demo=1` runs it on fake data with no server.

## Capabilities and Constraints
- Stack (existing): Python 3.12 backend (FastAPI + uvicorn); the dashboard is plain HTML/CSS/JS with no build step.
- Coach mood values: `neutral`, `impressed`, `exasperated`.
- Fallback modes the UI must show honestly: offline / no-LLM mode; colour-only hold detection.
- Every cue has a deadline; late cues are skipped, and suppressed cues are reported to the dashboard.
- Coordinates are normalised 0-1, origin top-left, y down.
- Limbs: left hand, right hand, left foot, right foot.
- Payload shapes come from `coach/contracts.py`.
- Persona: Coach Harrow (`config/personas/coach.yaml`), a dry retired competition coach, voiced by Kokoro.

## Evidence on Hand
- Footage: the team's own test clips in `videos/` (gitignored), one per scenario, listed in `TEST_CLIMBS.md`. The pose model comes from MediaPipe's public URL (`models/README.md`).
- Latency is measured: `docs/BENCHMARKS.md`, from `scripts/bench_pipeline.py`. On 32 clips, 164 cues spoken, frame to audio median 140 ms, p95 171 ms, max 186 ms, all inside the 800 ms target. Quote only figures from that file, with its caveats.
- No user feedback or testimonials. Don't make any up.

## Product Principles
1. **Latency is visible.** The live loop and its timing should be plain to see; being in sync with the climb is the proof.
2. **Readable from across the room.** Judges watch from a distance while the climb is happening, so what matters must read at a glance.
3. **Coaching, not just commentary.** Cues and debriefs refer to the actual problem, holds and earlier attempts.
4. **Degrade honestly.** Offline, no-LLM and fallback modes are labelled clearly and the demo keeps running.
