from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

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


class PeriodicCheckpointCallback(BaseCallback):
    """Save model (and optional replay buffer / VecNormalize) every N timesteps."""

    def __init__(
        self,
        save_freq: int,
        save_path: Path,
        name_prefix: str,
        save_replay_buffer: bool,
        vecnormalize: VecNormalize | None,
    ) -> None:
        super().__init__()
        self.save_freq = int(save_freq)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.vecnormalize = vecnormalize
        self._next_save = self.save_freq
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self) -> None:
        step_id = self.num_timesteps
        path = self.save_path / f"{self.name_prefix}_{step_id}_steps"
        self.model.save(str(path))
        if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
            self.model.save_replay_buffer(str(path) + "_replay.pkl")
        if self.vecnormalize is not None:
            self.vecnormalize.save(str(path) + "_vecnormalize.pkl")

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if self.num_timesteps >= self._next_save:
            self._save_checkpoint()
            while self.num_timesteps >= self._next_save:
                self._next_save += self.save_freq
        return True


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


def _infer_run_id_from_path(model_path: Path | None) -> int | None:
    if model_path is None:
        return None
    for parent in model_path.parents:
        if parent.name.startswith("SAC_"):
            try:
                return int(parent.name.split("_", 1)[1])
            except ValueError:
                continue
    return None


def _infer_run_dir_from_path(model_path: Path | None) -> Path | None:
    if model_path is None:
        return None
    for parent in model_path.parents:
        if parent.name.startswith("SAC_"):
            return parent
    return None


def _resolve_vecnormalize(
    env: DummyVecEnv,
    *,
    use_vecnormalize: bool,
    vecnormalize_path: Path,
    normalize_reward: bool,
    training: bool,
) -> tuple[DummyVecEnv, VecNormalize | None]:
    if vecnormalize_path.exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = training
        return env, env

    if not use_vecnormalize:
        return env, None

    env = VecNormalize(env, norm_obs=True, norm_reward=normalize_reward)
    return env, env


def _save_training_config(config_path: Path, payload: dict) -> None:
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


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
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Save checkpoints every N timesteps (0 disables).",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--save-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--load-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--replay-buffer-path", type=Path, default=None)
    parser.add_argument(
        "--vecnormalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable VecNormalize for observations (and optional rewards).",
    )
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize rewards when using VecNormalize.",
    )
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
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

    resume_run_dir = _infer_run_dir_from_path(args.resume_path)
    inferred_run_id = _infer_run_id_from_path(args.resume_path)
    run_id = args.run_id or inferred_run_id or _next_run_id(Path("models"))
    model_dir = resume_run_dir or (Path("models") / f"SAC_{run_id}")
    log_dir = Path("logs") / f"SAC_{run_id}"
    model_path = args.model_path or (model_dir / "sac_unitree_go2.zip")
    tensorboard_log = args.tensorboard_log or log_dir
    base_dir = resume_run_dir or model_dir
    checkpoint_dir = args.checkpoint_dir or (base_dir / "checkpoints")
    replay_buffer_path = args.replay_buffer_path or (base_dir / "replay_buffer.pkl")
    vecnormalize_path = args.vecnormalize_path or (base_dir / "vecnormalize.pkl")

    set_random_seed(args.seed)
    env = make_env(
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
    )
    use_vecnormalize = args.vecnormalize or (
        args.resume_path is not None and vecnormalize_path.exists()
    )
    env, vecnormalize_env = _resolve_vecnormalize(
        env,
        use_vecnormalize=use_vecnormalize,
        vecnormalize_path=vecnormalize_path,
        normalize_reward=args.normalize_reward,
        training=True,
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
        if args.load_replay_buffer and replay_buffer_path.exists():
            model.load_replay_buffer(str(replay_buffer_path))
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

    model_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_path is None:
        _save_training_config(
            model_dir / "train_config.json",
            {
                **{
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()
                },
                "run_id": run_id,
                "model_path": str(model_path),
                "tensorboard_log": str(tensorboard_log),
                "replay_buffer_path": str(replay_buffer_path),
                "vecnormalize_path": str(vecnormalize_path),
            },
        )

    callback = None
    if args.checkpoint_freq > 0:
        callback = PeriodicCheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=model_path.stem,
            save_replay_buffer=args.save_replay_buffer,
            vecnormalize=vecnormalize_env,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=args.resume_path is None,
        callback=callback,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    if args.save_replay_buffer and hasattr(model, "save_replay_buffer"):
        replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(str(replay_buffer_path))
    if vecnormalize_env is not None:
        vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)
        vecnormalize_env.save(str(vecnormalize_path))

    env.close()


if __name__ == "__main__":
    main()
