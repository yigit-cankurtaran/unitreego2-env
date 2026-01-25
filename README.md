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

- Each `step()` returns additional info fields such as `base_height`, `is_healthy`, `ctrl_cost`, and `contact_cost`.
- `make_unitree_go2_env` wraps the environment in `RecordEpisodeStatistics` by default. Set `record_episode_statistics=False` to disable this.

## Current limitations

- The environment is not stabilized for training yet.
- Domain randomization is limited to ground friction and actuator strength.

## Roadmap ideas

- Add tests for basic rollouts and API compliance.
- Add domain randomization and configurable terrain.
- Improve reward shaping and observation features.

## License

No license yet.
