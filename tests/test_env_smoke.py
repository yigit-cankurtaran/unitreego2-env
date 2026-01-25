import numpy as np
from gymnasium.wrappers import TimeLimit

from custom_mujoco_env import make_unitree_go2_env


def test_env_reset_and_step():
    env = make_unitree_go2_env(render=False, max_episode_steps=1)
    try:
        assert isinstance(env, TimeLimit)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert terminated or truncated
        assert isinstance(info, dict)
    finally:
        env.close()
