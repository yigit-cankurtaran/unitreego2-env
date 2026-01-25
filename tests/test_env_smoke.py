import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

from custom_mujoco_env import make_unitree_go2_env


def test_env_reset_and_step():
    env = make_unitree_go2_env(render=False, max_episode_steps=1)
    try:
        assert isinstance(env, RecordEpisodeStatistics)
        assert isinstance(env.env, TimeLimit)
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
        assert "base_height" in info
        assert "is_healthy" in info
        assert "ctrl_cost" in info
        assert "contact_cost" in info
    finally:
        env.close()


def test_domain_randomization_scales_friction():
    env = make_unitree_go2_env(
        render=False,
        domain_randomization=True,
        friction_scale_range=(0.5, 0.5),
        actuator_strength_scale_range=(1.0, 1.0),
    )
    try:
        base_env = env.unwrapped
        assert base_env._ground_geom_id is not None
        base = base_env._base_geom_friction.copy()
        env.reset()
        scaled = base_env.model.geom_friction[base_env._ground_geom_id, :]
        expected = base[base_env._ground_geom_id, :] * 0.5
        assert np.allclose(scaled, expected)
    finally:
        env.close()
