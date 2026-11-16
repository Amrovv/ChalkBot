# data/

Gitignored, except this file, the hold maps and the phrase bank. Footage and other big files
live on the shared drive, never in git.

| Path | Holds | In git |
|---|---|---|
| `holdmaps/<problem>.json` | One hold map per problem: holds numbered bottom (1) to top (N), with an optional polygon for a slanted hold | yes |
| `phrasebank.json` | The coach's lines, by event kind, specificity level and mood | yes |
| `audio_cache/` | Every line and fragment pre-rendered by Kokoro (`.npy`, 24 kHz mono) | no |
| `pose_cache/<clip>.jsonl` | One `ClimberState` per line, written by `scripts/record_pose.py` | no |
| `stills/<problem>/` | Pictures uploaded to the dashboard's hold map editor | no |
| `logs/` | `bus-<problem>-<time>.jsonl` (every bus topic except frames) and `--transcript` files from `run.py` | no |

Clips go in `videos/` (also gitignored), named `<problem>_<clip no>_<name>.mp4` as in
`TEST_CLIMBS.md`; clip `00` of a problem is the empty wall its hold map is built from.
