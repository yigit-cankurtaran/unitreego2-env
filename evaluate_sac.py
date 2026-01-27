from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from custom_mujoco_env import make_unitree_go2_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC policy.")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.model_path is None:
        if args.run_id is None:
            raise SystemExit("Provide --run-id or --model-path")
        model_path = Path("models") / f"SAC_{args.run_id}" / "sac_unitree_go2.zip"
    else:
        model_path = args.model_path

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None and args.run_id is not None:
        vecnormalize_path = Path("models") / f"SAC_{args.run_id}" / "vecnormalize.pkl"

    def _init():
        return make_unitree_go2_env(
            render=True,
            max_episode_steps=args.max_episode_steps,
            record_episode_statistics=False,
        )

    env = make_vec_env(_init, n_envs=1, vec_env_cls=DummyVecEnv)
    if vecnormalize_path is not None and vecnormalize_path.exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = args.normalize_reward

    model = SAC.load(str(model_path), env=env)

    returns, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=args.deterministic,
        render=True,
        return_episode_rewards=True,
    )

    for idx, (ret, length) in enumerate(zip(returns, lengths), start=1):
        print(f"Episode {idx}: return={ret:.2f}, length={length}")

    env.close()


if __name__ == "__main__":
    main()
