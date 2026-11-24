"""Autonomous Flight Computer baseline: Cascaded PID and Hoverslam (Suicide Burn) Guidance."""
import numpy as np
from kepler.config import BoosterConfig
from kepler.physics.quaternion import quat_to_euler


class KeplerFlightComputer:
    """
    Classical cascaded flight computer implementing:
    1. Hoverslam / Suicide-burn altitude ignition trigger.
    2. Vertical velocity damping to target 1.0 m/s touchdown.
    3. Lateral position and velocity proportional-derivative guidance.
    4. TVC gimbal and attitude stabilization.
    """

    def __init__(self, booster_cfg: BoosterConfig | None = None):
        self.cfg = booster_cfg or BoosterConfig()

        # Altitude / Velocity gains
        self.kp_z = 0.8
        self.kd_z = 1.2

        # Lateral guidance gains: gentle proportional gain with strong damping to prevent overshoot
        self.kp_pos = 0.015
        self.kd_pos = 0.08
        self.max_guidance_tilt = np.radians(12.0)  # Max commanded bank angle

        # Attitude loop gains (target tilt error -> gimbal command)
        self.kp_att = 4.0
        self.kd_att = 2.2

        self.ignited = False

    def reset(self):
        """Reset flight computer state for new episode."""
        self.ignited = False

    def compute_suicide_burn_altitude(self, vz: float, mass: float) -> float:
        """Calculate the altitude where thrust brings vz to flare altitude."""
        if vz >= -2.0:
            return 0.0
        thrust = 0.75 * self.cfg.max_thrust
        a_net = (thrust / max(1.0, mass)) - self.cfg.g0
        if a_net <= 0.5:
            return 2000.0
        return 2.5 * (vz ** 2) / (2.0 * a_net) + 30.0

    def get_action(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        """
        Compute normalized action [-1, 1]^4 from observation.
        obs: [x/100, y/100, z/500, vx/50, vy/50, vz/50, qw, qx, qy, qz, wx, wy, wz, tilt, fuel_frac, wx_wind/20, wy_wind/20]
        """
        x = obs[0] * 100.0
        y = obs[1] * 100.0
        z = obs[2] * 500.0
        vx = obs[3] * 50.0
        vy = obs[4] * 50.0
        vz = obs[5] * 50.0
        q = obs[6:10]
        wx = obs[10]
        wy = obs[11]
        wz = obs[12]
        fuel_frac = obs[14]

        euler = quat_to_euler(q)
        roll, pitch, yaw = euler

        fuel_mass = fuel_frac * self.cfg.default_fuel
        mass = self.cfg.dry_mass + fuel_mass

        # 2. Lateral Guidance -> Velocity Tracking Guidance (gentle banking)
        # Target lateral velocity directs rocket smoothly toward pad center (0, 0)
        v_cap = 1.4 if z > 25.0 else 0.6
        target_vx = -float(np.clip(0.18 * x, -v_cap, v_cap))
        target_vy = -float(np.clip(0.18 * y, -v_cap, v_cap))

        # Command gentle banking action (max ~3-4 deg) to track target velocity
        max_action = 0.25 if z > 6.0 else max(0.02, z / 25.0)
        a_tp = float(np.clip(0.08 * (target_vx - vx), -max_action, max_action))
        a_tr = float(np.clip(-0.08 * (target_vy - vy), -max_action, max_action))
        rcs_roll = float(np.clip(-0.5 * wz, -1.0, 1.0))

        # 3. Hoverslam Ignition Decision
        burn_alt = self.compute_suicide_burn_altitude(vz, mass)

        # Latch ignition once reached
        if (z <= burn_alt and vz < -3.0) or (z < 100.0 and vz < -10.0):
            self.ignited = True

        if not self.ignited and z > 150.0:
            # Coasting phase: save 100% fuel, engine CUTOFF!
            throttle_norm = -1.0  # Cutoff
            return np.array([throttle_norm, a_tp, a_tr, rcs_roll], dtype=np.float32)

        # Gravity-Settle Rule: Since minimum engine throttle (40%) has TWR > 1,
        # anytime descent speed slows to softer than -0.8 m/s, cut off engine
        # so Earth gravity smoothly brings it onto the landing legs!
        if vz >= -0.8:
            throttle_norm = -1.0
            return np.array([throttle_norm, a_tp, a_tr, rcs_roll], dtype=np.float32)

        # Target vertical speed: high deceleration aloft, gentle linear flare to -1.0 m/s near ground
        if z <= 35.0:
            target_vz = -1.0 - 0.25 * z
        else:
            target_vz = -np.clip(np.sqrt(2.0 * 12.0 * z), 9.0, 70.0)
        vz_error = target_vz - vz

        # Aggressive throttle response with damping
        desired_accel_z = self.cfg.g0 + 5.0 * vz_error
        desired_thrust_n = mass * desired_accel_z
        throttle = np.clip(desired_thrust_n / self.cfg.max_thrust, self.cfg.min_throttle, 1.0)
        # Map throttle [min_throttle, 1.0] linearly to [0.0, 1.0]
        throttle_norm = float(np.clip((throttle - self.cfg.min_throttle) / (1.0 - self.cfg.min_throttle), 0.0, 1.0))

        return np.array([throttle_norm, a_tp, a_tr, rcs_roll], dtype=np.float32)
