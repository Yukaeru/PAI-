# BC grasp v1 results

## Offline test

The 30 demonstration episodes were split into 24 training, 3 validation, and
3 test episodes. No transition from a validation or test episode was present
in training.

- held-out test MAE: 0.006299 rad
- held-out test RMSE: 0.012802 rad
- current-reference persistence baseline MAE: 0.032124 rad
- network: 8 inputs, two 128-unit ReLU layers, 5 outputs

The complete split, normalization values, and per-joint metrics are in
`../../models/bc_grasp_v1/report.json`.

## Gazebo closed-loop test

Ten positions were sampled from the same x/y range as collection with a new
random seed. Success required the cube to rise at least 0.03 m above its
initial measured height.

- stateless policy without gripper hysteresis: 6/10 successes
- policy with demonstration-consistent close latch: 9/10 successes
- final mean maximum lift over all trials: 0.047645 m
- final lift range: 0.019035 to 0.056139 m

`gazebo_evaluation.json` is the final 9/10 result.
`gazebo_evaluation_without_latch.json` preserves the 6/10 comparison on the
same positions. `gazebo_success.mp4` is a compact representative successful
rollout; `gazebo_success_contact_sheet.png` gives a quick visual overview.

Gazebo contact physics is not perfectly deterministic. One final trial lifted
the cube 0.019 m but did not reach the 0.03 m threshold, so the result is
reported as 90%, not 100%.

## Scope

The policy observes five measured joint positions and the cube XYZ position
provided directly by Gazebo. It does not infer the cube position from a camera,
and the evaluation covers only the approximately 20 mm by 20 mm collection
region. Wider placement ranges, camera observations, and more independent
evaluation seeds are future work.
