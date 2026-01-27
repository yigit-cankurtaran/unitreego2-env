# Unitree Go2 MuJoCo Gymnasium Environment

Work in progress. This repository is an early, minimal setup for simulating the Unitree Go2 in MuJoCo with a Gymnasium-compatible interface. Expect breaking changes as the environment matures.

## What is here

- A custom Gymnasium environment implemented in `custom_mujoco_env.py`.
- A MuJoCo model in `model/unitree_go2.xml` with the required mesh assets in `model/assets/`.
- Notes on how the environment was assembled in `env_creation_explanation.md`.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "gymnasium[mujoco]"
```

## Quick start

Run the script to launch the MuJoCo viewer and a short random rollout:

```bash
source .venv/bin/activate
python custom_mujoco_env.py
```

## Programmatic usage

```python
from custom_mujoco_env import make_unitree_go2_env

env = make_unitree_go2_env(render=False)
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Diagnostics

- Each `step()` returns additional info fields such as `base_height`, `is_healthy`, `upright`, `ctrl_cost`, `contact_cost`, base roll/pitch, and low-speed/fall penalty terms.
- `make_unitree_go2_env` wraps the environment in `RecordEpisodeStatistics` by default. Set `record_episode_statistics=False` to disable this.

## Reward shaping (locomotion-focused)

The current reward is tuned to discourage “stand still and survive” behaviors while still rewarding forward motion:

- Forward reward: `forward_reward_weight * max(x_velocity, 0)`.
- Survival reward: scaled by forward speed (`healthy_reward * clip(speed / low_speed_threshold, 0, 1)`), so standing still does not pay.
- Low-speed penalty: applied when `x_velocity < low_speed_threshold`.
- Idle penalty: per-second penalty when `forward_speed < idle_speed_threshold` (scaled by `dt`).
- Action-rate penalty: discourages jitter by penalizing changes in action from one step to the next.
- Orientation/lateral/control/contact penalties and a fall penalty remain.

All weights/thresholds are configurable via `UnitreeGo2Env` init args (`idle_speed_threshold`, `idle_penalty_weight`, `action_rate_penalty_weight`, etc.).

## Training defaults (SAC)

`train_sac.py` now exposes common SAC hyperparameters. Defaults are set for longer runs:

- `learning_rate=3e-4`, `buffer_size=1_000_000`, `batch_size=256`
- `learning_starts=10_000`, `train_freq=1`, `gradient_steps=1`
- `gamma=0.99`, `tau=0.005`, `ent_coef=auto`, `target_entropy=auto`
- Policy MLP: `[256, 256]`

For meaningful locomotion, expect to train for millions of timesteps (e.g., 3–10M).

## Resume training & checkpoints

- Resume from a saved policy (continues timesteps/logging by default):

```bash
python train_sac.py --resume-path models/SAC_1/sac_unitree_go2.zip --total-timesteps 1_000_000
```

- If you resume from a checkpoint in `models/SAC_<id>/checkpoints/`, the updated model is saved back to
  `models/SAC_<id>/sac_unitree_go2.zip` unless you pass `--model-path`.
- Replay buffer is saved to `models/SAC_<id>/replay_buffer.pkl` by default and is auto-loaded on resume when present.
  Use `--no-save-replay-buffer` or `--no-load-replay-buffer` to disable.
- Optional VecNormalize support (recommended for longer runs):

```bash
python train_sac.py --vecnormalize --normalize-reward --total-timesteps 3_000_000
```

This saves `vecnormalize.pkl` alongside the model and auto-loads it when resuming.

- Periodic checkpoints (model + replay buffer + VecNormalize):

```bash
python train_sac.py --checkpoint-freq 200000
```

Checkpoints land in `models/SAC_<id>/checkpoints/`.

## Evaluating with VecNormalize

If you trained with VecNormalize, pass the stats file when evaluating:

```bash
python evaluate_sac.py --run-id 1 --vecnormalize-path models/SAC_1/vecnormalize.pkl
```

## Current limitations

- The environment is not stabilized for training yet.
- Domain randomization is limited to ground friction and actuator strength.

## Roadmap ideas

- Add tests for basic rollouts and API compliance.
- Add domain randomization and configurable terrain.
- Improve reward shaping and observation features.

## License

No license yet.
