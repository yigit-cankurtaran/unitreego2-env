from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_mujoco_env import make_unitree_go2_env


def make_env(seed: int, max_episode_steps: int | None, n_envs: int) -> DummyVecEnv:
    def _init():
        env = make_unitree_go2_env(
            render=False,
            max_episode_steps=max_episode_steps,
            record_episode_statistics=False,
        )
        env = Monitor(env)
        return env

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return make_vec_env(_init, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)


def _next_run_id(base_dir: Path) -> int:
    existing = []
    if base_dir.exists():
        for path in base_dir.iterdir():
            if path.is_dir() and path.name.startswith("SAC_"):
                try:
                    existing.append(int(path.name.split("_", 1)[1]))
                except ValueError:
                    continue
    return max(existing, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on Unitree Go2.")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    args = parser.parse_args()

    run_id = args.run_id or _next_run_id(Path("models"))
    model_dir = Path("models") / f"SAC_{run_id}"
    log_dir = Path("logs") / f"SAC_{run_id}"
    model_path = args.model_path or (model_dir / "sac_unitree_go2.zip")
    tensorboard_log = args.tensorboard_log or log_dir

    set_random_seed(args.seed)
    env = make_env(
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        n_envs=args.n_envs,
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(tensorboard_log),
        device="auto",
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    env.close()


if __name__ == "__main__":
    main()
