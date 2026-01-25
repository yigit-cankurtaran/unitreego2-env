from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from custom_mujoco_env import make_unitree_go2_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC policy.")
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_unitree_go2.zip"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = make_unitree_go2_env(
        render=True,
        max_episode_steps=args.max_episode_steps,
        record_episode_statistics=False,
    )

    model = SAC.load(str(args.model_path), env=env)

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
