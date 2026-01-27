from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_mujoco_env import make_unitree_go2_env


def make_env(
    seed: int,
    max_episode_steps: int | None,
    n_envs: int,
    vec_env: str,
) -> DummyVecEnv:
    def _init():
        env = make_unitree_go2_env(
            render=False,
            max_episode_steps=max_episode_steps,
            record_episode_statistics=False,
        )
        env = Monitor(env)
        return env

    if vec_env == "auto":
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    elif vec_env == "subproc":
        vec_env_cls = SubprocVecEnv
    else:
        vec_env_cls = DummyVecEnv
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
    parser.add_argument(
        "--vec-env",
        choices=["auto", "subproc", "dummy"],
        default="auto",
        help="Vectorized env backend. Use dummy in environments where subprocesses fail.",
    )
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument(
        "--replay-buffer-path",
        type=Path,
        default=None,
        help="Optional path to load/save the replay buffer for resume stability.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent-coef", type=str, default="auto")
    parser.add_argument("--target-entropy", type=str, default="auto")
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
        vec_env=args.vec_env,
    )

    ent_coef = args.ent_coef if args.ent_coef.startswith("auto") else float(args.ent_coef)
    target_entropy = (
        args.target_entropy
        if args.target_entropy.startswith("auto")
        else float(args.target_entropy)
    )

    if args.resume_path is not None:
        if not args.resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {args.resume_path}")
        model = SAC.load(
            str(args.resume_path),
            env=env,
            device="auto",
            tensorboard_log=str(tensorboard_log),
        )
        if args.replay_buffer_path is not None and args.replay_buffer_path.exists():
            model.load_replay_buffer(str(args.replay_buffer_path))
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_log),
            device="auto",
            seed=args.seed,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            train_freq=(args.train_freq, "step"),
            gradient_steps=args.gradient_steps,
            gamma=args.gamma,
            tau=args.tau,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs={"net_arch": [256, 256]},
        )

    callback = None
    if args.checkpoint_dir is not None and args.checkpoint_freq > 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(args.checkpoint_dir),
            name_prefix="sac",
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    if args.replay_buffer_path is not None:
        args.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(str(args.replay_buffer_path))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    env.close()


if __name__ == "__main__":
    main()
