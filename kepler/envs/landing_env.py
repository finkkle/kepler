"""Gymnasium-compliant 6-DOF precision rocket landing environment."""
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from kepler.config import BoosterConfig, SimConfig
from kepler.physics.dynamics import RocketDynamics, RocketState
from kepler.physics.quaternion import quat_to_euler


class KeplerLandingEnv(gym.Env):
    """
    6-DOF Autonomous Rocket Landing Environment.
    
    Goal: Steer the booster from initial altitude and lateral offset to land upright
    on a designated pad (radius 15m, bullseye 3m) with near-zero touchdown velocity,
    while minimizing fuel expenditure.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        booster_cfg: BoosterConfig | None = None,
        sim_cfg: SimConfig | None = None,
        difficulty: str = "normal",  # 'easy', 'normal', 'hard'
        wind_enabled: bool = True,
    ):
        super().__init__()
        self.cfg = booster_cfg or BoosterConfig()
        self.sim_cfg = sim_cfg or SimConfig()
        self.difficulty = difficulty
        self.wind_enabled = wind_enabled

        self.dynamics = RocketDynamics(self.cfg, self.sim_cfg)

        # Action space: [throttle, gimbal_pitch, gimbal_yaw, rcs_roll]
        # All actions in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: 17 elements
        # [x/100, y/100, z/500, vx/50, vy/50, vz/50, qw, qx, qy, qz, wx, wy, wz, tilt, fuel_frac, wx_wind/20, wy_wind/20]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        self.step_count = 0
        self.max_steps = int(self.sim_cfg.max_flight_time / self.sim_cfg.dt)
        self.initial_fuel = self.cfg.default_fuel
        self.initial_altitude = 500.0

    def _get_obs(self) -> np.ndarray:
        s = self.dynamics.state
        fuel_frac = s.fuel_mass / max(1.0, self.initial_fuel)
        wind = self.dynamics.wind.get_wind(s.pos[2], dt=0.0) if self.wind_enabled else np.zeros(3)

        obs = np.array([
            s.pos[0] / 100.0,
            s.pos[1] / 100.0,
            s.pos[2] / 500.0,
            s.vel[0] / 50.0,
            s.vel[1] / 50.0,
            s.vel[2] / 50.0,
            s.quat[0],
            s.quat[1],
            s.quat[2],
            s.quat[3],
            s.omega[0],
            s.omega[1],
            s.omega[2],
            s.tilt_rad,
            fuel_frac,
            wind[0] / 20.0,
            wind[1] / 20.0,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # Curriculum / difficulty configuration
        options = options or {}
        diff = options.get("difficulty", self.difficulty)

        if diff == "terminal_flare":
            # Very close terminal flare: 70m - 120m, vz = -10 to -16 m/s, minimal drift
            alt = float(np.random.uniform(70.0, 130.0))
            offset_x = float(np.random.uniform(-8.0, 8.0))
            offset_y = float(np.random.uniform(-8.0, 8.0))
            vz = float(np.random.uniform(-16.0, -8.0))
            vx = float(np.random.uniform(-1.0, 1.0))
            vy = float(np.random.uniform(-1.0, 1.0))
            tilt_deg = float(np.random.uniform(0.0, 2.5))
            wind_speed = 0.0
        elif diff == "easy":
            # Short altitude, minimal lateral drift, light wind
            alt = float(np.random.uniform(180.0, 300.0))
            offset_x = float(np.random.uniform(-20.0, 20.0))
            offset_y = float(np.random.uniform(-20.0, 20.0))
            vz = float(np.random.uniform(-25.0, -15.0))
            vx = float(np.random.uniform(-3.0, 3.0))
            vy = float(np.random.uniform(-3.0, 3.0))
            tilt_deg = float(np.random.uniform(0.0, 4.0))
            wind_speed = float(np.random.uniform(0.0, 3.0)) if self.wind_enabled else 0.0
        elif diff == "hard":
            # High altitude, strong crosswind, larger lateral offset
            alt = float(np.random.uniform(700.0, 1200.0))
            offset_x = float(np.random.uniform(-100.0, 100.0))
            offset_y = float(np.random.uniform(-100.0, 100.0))
            vz = float(np.random.uniform(-65.0, -45.0))
            vx = float(np.random.uniform(-12.0, 12.0))
            vy = float(np.random.uniform(-12.0, 12.0))
            tilt_deg = float(np.random.uniform(2.0, 15.0))
            wind_speed = float(np.random.uniform(6.0, 15.0)) if self.wind_enabled else 0.0
        else:  # 'normal'
            alt = float(np.random.uniform(400.0, 700.0))
            offset_x = float(np.random.uniform(-50.0, 50.0))
            offset_y = float(np.random.uniform(-50.0, 50.0))
            vz = float(np.random.uniform(-45.0, -30.0))
            vx = float(np.random.uniform(-6.0, 6.0))
            vy = float(np.random.uniform(-6.0, 6.0))
            tilt_deg = float(np.random.uniform(1.0, 8.0))
            wind_speed = float(np.random.uniform(2.0, 8.0)) if self.wind_enabled else 0.0

        self.initial_altitude = alt
        self.initial_fuel = self.cfg.default_fuel
        self.dynamics.wind.reset(base_speed=wind_speed, direction_deg=float(np.random.uniform(0.0, 360.0)))

        self.dynamics.reset(
            altitude=alt,
            lateral_offset=(offset_x, offset_y),
            initial_vel=(vx, vy, vz),
            initial_tilt_deg=tilt_deg,
            fuel_mass=self.initial_fuel
        )

        return self._get_obs(), {"initial_altitude": alt, "difficulty": diff}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Action decoding:
        # a0: Commanded throttle [-1, 1]. Below 0.0 -> Cutoff (0.0). Above 0.0 -> [min_throttle, max_throttle].
        # a1: Commanded target pitch tilt [-1, 1] -> [-16 deg, +16 deg].
        # a2: Commanded target roll tilt [-1, 1] -> [-16 deg, +16 deg].
        # a3: RCS roll command [-1, 1].
        a_throttle, a_tilt_p, a_tilt_r, a_rcs = action
        if a_throttle <= 0.0:
            throttle_cmd = 0.0
        else:
            throttle_cmd = self.cfg.min_throttle + (self.cfg.max_throttle - self.cfg.min_throttle) * float(a_throttle)

        # Current rocket attitude
        s_prev = self.dynamics.state
        euler = quat_to_euler(s_prev.quat)
        roll, pitch, yaw = euler
        wx, wy, wz = s_prev.omega
        z_curr = max(0.0, s_prev.pos[2])

        # Target tilt angles (enforce vertical alignment only in final 6m touchdown flare)
        max_commanded_tilt = np.radians(16.0)
        near_ground_scale = 1.0 if z_curr > 6.0 else max(0.0, z_curr / 6.0)
        target_pitch = float(a_tilt_p * max_commanded_tilt * near_ground_scale)
        target_roll = float(a_tilt_r * max_commanded_tilt * near_ground_scale)

        # High-bandwidth Inner-Loop TVC Flight Control System (FCS)
        kp_att = 4.2
        kd_att = 2.4
        gy_cmd = np.clip(kp_att * (pitch - target_pitch) + kd_att * wy, -self.cfg.max_gimbal_angle, self.cfg.max_gimbal_angle)
        gz_cmd = np.clip(-(kp_att * (roll - target_roll) + kd_att * wx), -self.cfg.max_gimbal_angle, self.cfg.max_gimbal_angle)
        rcs_cmd = float(a_rcs)

        # Autonomous Touchdown Settle Logic:
        # Since TWR > 1 at minimum throttle, when the booster has slowed to gentle descent (< 0.85 m/s)
        # near ground (z <= 12m), engine cuts off so Earth gravity smoothly settles the rocket onto the landing legs
        if z_curr <= 12.0 and s_prev.vel[2] >= -0.85:
            throttle_cmd = 0.0
        elif z_curr <= 0.5 and -1.8 <= s_prev.vel[2] <= 0.1:
            throttle_cmd = 0.0

        fuel_before = s_prev.fuel_mass

        # Step 6-DOF dynamics with RK4 integrator
        state = self.dynamics.step(
            throttle_cmd=throttle_cmd,
            gimbal_y_cmd=float(gy_cmd),
            gimbal_z_cmd=float(gz_cmd),
            rcs_cmd=rcs_cmd
        )

        fuel_used = max(0.0, fuel_before - state.fuel_mass)

        # Distances and velocities
        dist_xy = np.linalg.norm(state.pos[:2])
        vz = state.vel[2]
        vh = np.linalg.norm(state.vel[:2])
        tilt = state.tilt_rad

        # Multi-objective reward formulation:
        # 1. Fuel conservation penalty (directly incentivizes suicide burn coasting!)
        r_fuel = -0.008 * fuel_used

        # 2. Upright attitude penalty
        r_tilt = -1.2 * (tilt ** 2)

        # 3. Angular velocity penalty
        r_omega = -0.08 * np.sum(state.omega ** 2)

        # 4. Target distance & velocity penalty
        r_dist = -0.03 * dist_xy - 0.05 * vh

        # 5. Altitude glideslope tracking
        ideal_vz = -np.sqrt(2.0 * 12.0 * max(0.1, state.pos[2]))
        r_descent = -0.01 * min(abs(vz - ideal_vz), 25.0)

        step_reward = r_fuel + r_tilt + r_omega + r_dist + r_descent

        # Check termination conditions
        terminated = False
        truncated = False
        info = {
            "pos": state.pos.tolist(),
            "vel": state.vel.tolist(),
            "fuel_mass": state.fuel_mass,
            "fuel_used": self.initial_fuel - state.fuel_mass,
            "tilt_deg": np.degrees(tilt),
            "touchdown": False,
            "landing_success": False,
        }

        # Case 1: Ground contact (z <= 0)
        if state.ground_contact or state.pos[2] <= 0.0:
            terminated = True
            info["touchdown"] = True

            # Assess landing success criteria
            soft_touchdown = (-self.cfg.max_touchdown_vz <= vz <= 0.5)
            low_horizontal_speed = (vh <= self.cfg.max_touchdown_vh)
            upright_attitude = (tilt <= self.cfg.max_touchdown_tilt)
            on_pad = (dist_xy <= self.cfg.pad_radius)

            if soft_touchdown and low_horizontal_speed and upright_attitude and on_pad:
                # SUCCESSFUL LANDING!
                info["landing_success"] = True
                # Terminal reward: big landing bonus + reward for remaining fuel reserve + accuracy bonus
                fuel_reserve_frac = state.fuel_mass / max(1.0, self.initial_fuel)
                accuracy_bonus = max(0.0, 1.0 - (dist_xy / self.cfg.pad_radius))
                softness_bonus = max(0.0, 1.0 - (abs(vz) / self.cfg.max_touchdown_vz))

                terminal_bonus = 600.0 + 400.0 * fuel_reserve_frac + 100.0 * accuracy_bonus + 100.0 * softness_bonus
                step_reward += terminal_bonus
            else:
                # CRASH / HARD LANDING
                crash_penalty = -250.0
                if not soft_touchdown:
                    crash_penalty -= min(150.0, 10.0 * abs(vz))
                if not upright_attitude:
                    crash_penalty -= min(100.0, 50.0 * np.degrees(tilt))
                if not on_pad:
                    crash_penalty -= min(100.0, 2.0 * dist_xy)
                step_reward += crash_penalty

        # Case 2: Extreme tumbling / inverted flight (tilt > 85 deg)
        elif tilt > np.radians(85.0):
            terminated = True
            step_reward -= 200.0
            info["failure_reason"] = "inverted_attitude"

        # Case 3: Flown out of corridor (dist > 400m)
        elif dist_xy > 400.0:
            terminated = True
            step_reward -= 200.0
            info["failure_reason"] = "out_of_corridor"

        # Case 4: Max steps reached
        elif self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), float(step_reward), terminated, truncated, info
