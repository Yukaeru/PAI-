# Collected datasets

## `grasp_auto_20261102_012854`

Primary behavior-cloning collection:

- 30 automatically executed episodes
- 30 successful grasps, 0 failures
- cube x range: -9.54 mm to +9.82 mm
- cube y range: -109.60 mm to -90.06 mm
- measured lift range: 44.34 mm to 47.46 mm
- 5,036 merged transitions
- observation: five measured joints plus cube x/y/z
- action: five controller references approximately 0.2 seconds ahead

Use `bc_collection/transitions.jsonl` for training and
`bc_collection/summary.json` for field order and episode counts.

## `grasp_auto_20261102_012814`

One-episode smoke test created before the primary collection. It verifies the
automatic capture and conversion pipeline but is not needed in the main
training split.
