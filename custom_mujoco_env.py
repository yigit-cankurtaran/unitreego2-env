from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
import mujoco
import math

DEFAULT_MODEL_PATH = Path("model") / "unitree_go2.xml"
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array([0.0, 0.0, 0.35]),
    "elevation": -25.0,
}


class UnitreeGo2Env(MujocoEnv, utils.EzPickle):
    """Gymnasium-compatible MuJoCo environment backed by the Unitree Go2 XML."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": 100,
    }

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        frame_skip: int = 5,
        forward_reward_weight: float = 2.0,
        lateral_velocity_weight: float = 0.05,
        orientation_cost_weight: float = 0.2,
        ctrl_cost_weight: float = 1e-3,
        contact_cost_weight: float = 2e-4,
        foot_contact_cost_weight: float = 5e-5,
        rear_contact_reward_weight: float = 0.1,
        contact_balance_weight: float = 0.05,
        contact_force_clip: float = 1.0,
        healthy_reward: float = 0.5,
        low_speed_threshold: float = 0.3,
        low_speed_penalty_weight: float = 2.5,
        idle_speed_threshold: float = 0.1,
        idle_penalty_weight: float = 5.0,
        action_rate_penalty_weight: float = 1e-4,
        fall_penalty: float = 10.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.22, 0.5),
        healthy_orientation_range: tuple[float, float] = (-0.6, 0.6),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.01,
        exclude_current_positions_from_observation: bool = True,
        domain_randomization: bool = False,
        friction_scale_range: tuple[float, float] = (0.7, 1.3),
        actuator_strength_scale_range: tuple[float, float] = (0.85, 1.15),
        **kwargs,
    ):
        """Load the MuJoCo model and set reward / termination coefficients."""

        # Resolve model path relative to this file so importing from anywhere still works.
        self._model_path_argument = str(model_path)
        resolved_model_path = self._resolve_model_path(model_path)

        utils.EzPickle.__init__(
            self,
            self._model_path_argument,
            frame_skip,
            forward_reward_weight,
            lateral_velocity_weight,
            orientation_cost_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            foot_contact_cost_weight,
            rear_contact_reward_weight,
            contact_balance_weight,
            contact_force_clip,
            healthy_reward,
            low_speed_threshold,
            low_speed_penalty_weight,
            idle_speed_threshold,
            idle_penalty_weight,
            action_rate_penalty_weight,
            fall_penalty,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_orientation_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain_randomization,
            friction_scale_range,
            actuator_strength_scale_range,
            **kwargs,
        )

        # Cache reward shaping parameters for later use in step().
        self._forward_reward_weight = forward_reward_weight
        self._lateral_velocity_weight = lateral_velocity_weight
        self._orientation_cost_weight = orientation_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._foot_contact_cost_weight = foot_contact_cost_weight
        self._rear_contact_reward_weight = rear_contact_reward_weight
        self._contact_balance_weight = contact_balance_weight
        self._contact_force_clip = contact_force_clip
        self._healthy_reward = healthy_reward
        self._low_speed_threshold = low_speed_threshold
        self._low_speed_penalty_weight = low_speed_penalty_weight
        self._idle_speed_threshold = idle_speed_threshold
        self._idle_penalty_weight = idle_penalty_weight
        self._action_rate_penalty_weight = action_rate_penalty_weight
        self._fall_penalty = fall_penalty
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_orientation_range = healthy_orientation_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._domain_randomization = domain_randomization
        self._friction_scale_range = friction_scale_range
        self._actuator_strength_scale_range = actuator_strength_scale_range

        MujocoEnv.__init__(
            self,
            model_path=str(resolved_model_path),
            frame_skip=frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self._ground_geom_id = self._find_geom_id("ground")
        self._base_geom_friction = self.model.geom_friction.copy()
        self._base_actuator_gear = self.model.actuator_gear.copy()
        self._last_action = np.zeros(self.model.nu, dtype=np.float64)
        self._contact_force_buffer = np.zeros(6, dtype=np.float64)
        self._last_contact_summary: tuple[float, float, float, int, int, int] | None = None
        self._foot_geom_ids = {
            name: self._find_geom_id(name) for name in ("FL", "FR", "RL", "RR")
        }
        self._front_foot_geom_ids = {
            geom_id
            for name, geom_id in self._foot_geom_ids.items()
            if geom_id is not None and name in ("FL", "FR")
        }
        self._rear_foot_geom_ids = {
            geom_id
            for name, geom_id in self._foot_geom_ids.items()
            if geom_id is not None and name in ("RL", "RR")
        }
        self._all_foot_geom_ids = {
            geom_id for geom_id in self._foot_geom_ids.values() if geom_id is not None
        }

        # Build observation space dynamically from an initial observation vector.
        observation = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation.shape,
            dtype=np.float64,
        )

    @property
    def healthy_reward(self) -> float:
        """Reward granted for staying in an upright, healthy pose."""

        return float(self.is_healthy) * self._healthy_reward

    def control_cost(self, action: np.ndarray) -> float:
        """Quadratic penalty scaled by ``ctrl_cost_weight``."""

        return self._ctrl_cost_weight * float(np.sum(np.square(action)))

    @property
    def contact_forces(self) -> np.ndarray:
        """External contact forces clipped to a configurable range."""

        raw_forces = self.data.cfrc_ext.copy()
        min_value, max_value = self._contact_force_range
        return np.clip(raw_forces, min_value, max_value)

    @property
    def contact_cost(self) -> float:
        """Penalty term that discourages non-foot ground contacts."""

        if self._contact_cost_weight == 0.0:
            return 0.0
        summary = self._last_contact_summary
        if summary is None:
            summary = self._ground_contact_summary()
        _, _, nonfoot_force, _, _, _ = summary
        return self._contact_cost_weight * nonfoot_force

    @property
    def is_healthy(self) -> bool:
        """Check that the robot's base stays within the allowed height window."""

        state = self.state_vector()
        base_height = float(self.data.qpos[2])
        min_z, max_z = self._healthy_z_range
        roll, pitch = self._roll_pitch()
        min_angle, max_angle = self._healthy_orientation_range
        is_finite = np.isfinite(state).all()
        in_bounds = min_z <= base_height <= max_z
        upright = (min_angle <= roll <= max_angle) and (min_angle <= pitch <= max_angle)
        return bool(is_finite and in_bounds and upright)

    @property
    def terminated(self) -> bool:
        """Decide whether an episode should end because the robot fell."""

        if not self._terminate_when_unhealthy:
            return False
        return not self.is_healthy

    def step(self, action: np.ndarray):
        """Advance the simulation and compute reward / diagnostics."""

        # Track pelvis displacement to compute planar velocity.
        xy_position_before = self.get_body_com("base")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("base")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_speed = float(max(x_velocity, 0.0))
        forward_reward = self._forward_reward_weight * forward_speed
        speed_fraction = 1.0
        if self._low_speed_threshold > 0.0:
            speed_fraction = float(
                np.clip(forward_speed / self._low_speed_threshold, 0.0, 1.0)
            )
        healthy_reward = self.healthy_reward * speed_fraction
        ctrl_cost = self.control_cost(action)
        roll, pitch = self._roll_pitch()
        lateral_cost = self._lateral_velocity_weight * float(np.square(y_velocity))
        orientation_cost = self._orientation_cost_weight * float(roll * roll + pitch * pitch)
        low_speed_penalty = 0.0
        if self._low_speed_penalty_weight > 0.0 and x_velocity < self._low_speed_threshold:
            low_speed_penalty = self._low_speed_penalty_weight * float(
                self._low_speed_threshold - x_velocity
            )
        idle_penalty = 0.0
        if self._idle_penalty_weight > 0.0 and forward_speed < self._idle_speed_threshold:
            idle_penalty = self._idle_penalty_weight * self.dt
        action_rate_penalty = 0.0
        if self._action_rate_penalty_weight > 0.0:
            action_delta = action - self._last_action
            action_rate_penalty = self._action_rate_penalty_weight * float(
                np.sum(np.square(action_delta))
            )

        (
            front_contact_force,
            rear_contact_force,
            nonfoot_contact_force,
            front_contact_count,
            rear_contact_count,
            nonfoot_contact_count,
        ) = self._ground_contact_summary()
        self._last_contact_summary = (
            front_contact_force,
            rear_contact_force,
            nonfoot_contact_force,
            front_contact_count,
            rear_contact_count,
            nonfoot_contact_count,
        )
        foot_contact_cost = self._foot_contact_cost_weight * (
            front_contact_force + rear_contact_force
        )
        nonfoot_contact_cost = self._contact_cost_weight * nonfoot_contact_force
        rear_contact_reward = 0.0
        if self._rear_contact_reward_weight > 0.0:
            rear_contact_reward = (
                self._rear_contact_reward_weight * rear_contact_force * speed_fraction
            )
        contact_balance_penalty = 0.0
        if self._contact_balance_weight > 0.0:
            contact_balance_penalty = self._contact_balance_weight * float(
                (front_contact_force - rear_contact_force) ** 2
            )

        observation = self._get_obs()
        reward = (
            forward_reward
            + healthy_reward
            + rear_contact_reward
            - ctrl_cost
            - foot_contact_cost
            - nonfoot_contact_cost
            - contact_balance_penalty
            - lateral_cost
            - orientation_cost
            - low_speed_penalty
            - idle_penalty
            - action_rate_penalty
        )
        terminated = self.terminated
        fall_penalty = self._fall_penalty if terminated else 0.0
        reward -= fall_penalty

        self._last_action = np.array(action, copy=True)

        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -(foot_contact_cost + nonfoot_contact_cost + contact_balance_penalty),
            "reward_contact_foot": -foot_contact_cost,
            "reward_contact_nonfoot": -nonfoot_contact_cost,
            "reward_contact_balance": -contact_balance_penalty,
            "reward_rear_contact": rear_contact_reward,
            "reward_lateral": -lateral_cost,
            "reward_orientation": -orientation_cost,
            "reward_low_speed": -low_speed_penalty,
            "reward_idle": -idle_penalty,
            "reward_action_rate": -action_rate_penalty,
            "reward_fall": -fall_penalty,
            "ctrl_cost": ctrl_cost,
            "contact_cost": nonfoot_contact_cost,
            "foot_contact_cost": foot_contact_cost,
            "contact_balance_penalty": contact_balance_penalty,
            "lateral_cost": lateral_cost,
            "orientation_cost": orientation_cost,
            "low_speed_penalty": low_speed_penalty,
            "idle_penalty": idle_penalty,
            "action_rate_penalty": action_rate_penalty,
            "speed_fraction": speed_fraction,
            "fall_penalty": fall_penalty,
            "front_contact_force": front_contact_force,
            "rear_contact_force": rear_contact_force,
            "nonfoot_contact_force": nonfoot_contact_force,
            "front_contact_count": front_contact_count,
            "rear_contact_count": rear_contact_count,
            "nonfoot_contact_count": nonfoot_contact_count,
            "is_healthy": self.is_healthy,
            "base_height": float(self.data.qpos[2]),
            "base_roll": float(roll),
            "base_pitch": float(pitch),
            "x_position": float(xy_position_after[0]),
            "y_position": float(xy_position_after[1]),
            "x_velocity": float(x_velocity),
            "forward_speed": forward_speed,
            "y_velocity": float(y_velocity),
            "upright": self.is_healthy,
        }

        if self.render_mode == "human":
            self.render()

        # Return Gymnasium step tuple (observation, reward, terminated, truncated, info).
        return observation, reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """Assemble the low-level state vector returned to the agent."""

        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        # Optionally remove absolute x/y translation so the policy focuses on relative pose.
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))

    def reset_model(self) -> np.ndarray:
        """Apply small random noise around the reference pose to start an episode."""

        if self._domain_randomization:
            self._apply_domain_randomization()

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self._last_action = np.zeros(self.model.nu, dtype=np.float64)
        self._last_contact_summary = None
        return self._get_obs()

    @staticmethod
    def _resolve_model_path(model_path: str | Path) -> Path:
        """Handle absolute and relative paths to the MuJoCo XML."""

        candidate = Path(model_path)
        if candidate.is_absolute():
            return candidate
        return Path(__file__).resolve().parent / candidate

    def _roll_pitch(self) -> tuple[float, float]:
        """Compute roll and pitch from the base quaternion."""

        quat = self.data.qpos[3:7]
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        return roll, pitch

    def _find_geom_id(self, geom_name: str) -> int | None:
        """Return the geom id for a given name, or None if missing."""

        try:
            return int(self.model.geom(geom_name).id)
        except KeyError:
            return None

    def _apply_domain_randomization(self) -> None:
        """Randomize select physics properties for robustness."""

        self.model.geom_friction[:] = self._base_geom_friction
        self.model.actuator_gear[:] = self._base_actuator_gear

        friction_scale = self.np_random.uniform(*self._friction_scale_range)
        if self._ground_geom_id is not None:
            self.model.geom_friction[self._ground_geom_id, :] = (
                self._base_geom_friction[self._ground_geom_id, :] * friction_scale
            )

        actuator_scale = self.np_random.uniform(*self._actuator_strength_scale_range)
        self.model.actuator_gear[:] = self._base_actuator_gear * actuator_scale

    def _ground_contact_summary(self) -> tuple[float, float, float, int, int, int]:
        """Summarize ground contacts by front/rear foot and non-foot geoms."""

        if self._ground_geom_id is None or self.data.ncon == 0:
            return 0.0, 0.0, 0.0, 0, 0, 0

        front_force = 0.0
        rear_force = 0.0
        nonfoot_force = 0.0
        foot_geoms_in_contact: set[int] = set()
        nonfoot_geoms_in_contact: set[int] = set()

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self._ground_geom_id:
                other = geom2
            elif geom2 == self._ground_geom_id:
                other = geom1
            else:
                continue

            mujoco.mj_contactForce(
                self.model, self.data, contact_index, self._contact_force_buffer
            )
            normal_force = abs(float(self._contact_force_buffer[0]))
            if self._contact_force_clip > 0.0:
                normal_force = min(normal_force, self._contact_force_clip)

            if other in self._all_foot_geom_ids:
                foot_geoms_in_contact.add(other)
                if other in self._rear_foot_geom_ids:
                    rear_force += normal_force
                else:
                    front_force += normal_force
            else:
                nonfoot_geoms_in_contact.add(other)
                nonfoot_force += normal_force

        front_contact_count = len(
            self._front_foot_geom_ids.intersection(foot_geoms_in_contact)
        )
        rear_contact_count = len(
            self._rear_foot_geom_ids.intersection(foot_geoms_in_contact)
        )
        nonfoot_contact_count = len(nonfoot_geoms_in_contact)

        return (
            front_force,
            rear_force,
            nonfoot_force,
            front_contact_count,
            rear_contact_count,
            nonfoot_contact_count,
        )


def register_unitree_go2_env(
    env_id: str = "UnitreeGo2-v0",
    *,
    max_episode_steps: int = 1000,
    disable_env_checker: bool = False,
    additional_wrappers: Iterable | None = None,
    **env_kwargs,
) -> None:
    """Register the environment with Gymnasium's global registry."""

    if env_id in registry:
        return

    env_kwargs = env_kwargs or {}
    register(
        id=env_id,
        entry_point=UnitreeGo2Env,
        max_episode_steps=max_episode_steps,
        disable_env_checker=disable_env_checker,
        additional_wrappers=tuple(additional_wrappers or ()),
        kwargs=env_kwargs,
    )


def make_unitree_go2_env(
    env_id: str = "UnitreeGo2-v0",
    *,
    render: bool = False,
    max_episode_steps: int | None = 1000,
    record_episode_statistics: bool = True,
    register_kwargs: dict | None = None,
    **make_kwargs,
):
    """Helper that registers (if needed), applies a TimeLimit, and returns an env."""

    register_kwargs = register_kwargs or {}
    register_unitree_go2_env(env_id=env_id, **register_kwargs)
    if render and "render_mode" not in make_kwargs:
        make_kwargs["render_mode"] = "human"
    env = gym.make(env_id, **make_kwargs)
    if max_episode_steps is not None:
        if isinstance(env, TimeLimit):
            env._max_episode_steps = max_episode_steps
        else:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if record_episode_statistics and not isinstance(env, RecordEpisodeStatistics):
        env = RecordEpisodeStatistics(env)
    return env


if __name__ == "__main__":
    # Run a short random rollout so the viewer stays open long enough to inspect it.
    env = make_unitree_go2_env(render=True)
    obs, _ = env.reset()
    print("Observation shape:", obs.shape)

    try:
        for _ in range(1000):
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                env.reset()
    finally:
        env.close()
