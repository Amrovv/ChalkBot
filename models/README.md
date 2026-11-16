# models/

Gitignored. One file is needed:

- `pose_landmarker_full.task`: MediaPipe PoseLandmarker, full model (9 MB). Download it from
  Google's model page, <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models>,
  or directly:

  ```sh
  curl -L -o models/pose_landmarker_full.task     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
  ```

Without it `run.py` still starts, showing video with no pose (and so no coaching).
