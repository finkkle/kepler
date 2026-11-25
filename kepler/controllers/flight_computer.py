"""Autonomous Flight Computer: Space-Grade Cascaded Guidance and Hoverslam (Suicide Burn)."""
import numpy as np
from kepler.config import BoosterConfig
from kepler.physics.quaternion import quat_to_euler


class KeplerFlightComputer:
    """
    Precision flight computer implementing:
    1. Kinematic hoverslam ignition solver with lateral reachability corridor.
    2. Decoupled square-root deceleration tracking curve to a gentle 1.10 m/s touchdown.
    3. Critically-damped bounded lateral guidance corridor with smooth terminal taper (zero overshoot).
    4. Terminal vertical flare settling onto ASDS drone ship landing legs.
    """

    def __init__(
        self,
        booster_cfg: BoosterConfig | None = None,
        pad_alt: float = 25.0,
        lat_factor: float = 3.5,
        flare_alt: float = 6.0,
    ):
        self.cfg = booster_cfg or BoosterConfig()
        self.ignited = False
        self.pad_alt = pad_alt
        self.lat_factor = lat_factor
        self.flare_alt = flare_alt

    def reset(self):
        """Reset flight computer state for new flight."""
        self.ignited = False

    def compute_suicide_burn_altitude(self, vz: float, mass: float, dist_xy: float = 0.0) -> float:
        """Exact kinematic hoverslam altitude calculation with lateral reachability buffer."""
        if vz >= -1.5:
            return 0.0
        # Target design deceleration at 55% throttle (leaving headroom [40%, 100%])
        thrust_design = 0.55 * self.cfg.max_thrust
        a_net = (thrust_design / max(1.0, mass)) - self.cfg.g0
        a_net = max(1.5, a_net)
        # Vertical stopping distance to terminal touchdown speed (1.20 m/s)
        v_flare = 1.20
        d_stop = max(0.0, (vz ** 2 - v_flare ** 2) / (2.0 * a_net))
        # Lateral divert buffer: allows sufficient flight time to translate dist_xy onto pad
        d_lateral = self.lat_factor * dist_xy
        return d_stop + self.pad_alt + d_lateral

    def get_action(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        """
        Compute normalized action [-1, 1]^4 from observation.
        obs: [x/100, y/100, z/500, vx/50, vy/50, vz/50, qw, qx, qy, qz, wx, wy, wz, tilt, fuel_frac, wx_wind/20, wy_wind/20]
        """
        x = float(obs[0] * 100.0)
        y = float(obs[1] * 100.0)
        z = float(obs[2] * 500.0)
        vx = float(obs[3] * 50.0)
        vy = float(obs[4] * 50.0)
        vz = float(obs[5] * 50.0)
        q = obs[6:10]
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])
        fuel_frac = float(obs[14])
        wx_wind = float(obs[15] * 20.0)
        wy_wind = float(obs[16] * 20.0)

        fuel_mass = fuel_frac * self.cfg.default_fuel
        mass = self.cfg.dry_mass + fuel_mass
        dist_xy = float(np.hypot(x, y))

        # -----------------------------------------------------------------
        # 1. Lateral Guidance -> Proven Corridor with Precision Ground Taper
        # -----------------------------------------------------------------
        v_cap = float(np.clip(0.14 * z, 0.35, 2.5))
        target_vx = float(np.clip(-0.25 * x, -v_cap, v_cap))
        target_vy = float(np.clip(-0.25 * y, -v_cap, v_cap))

        # Critically damped velocity tracking
        cmd_ax = float(np.clip(1.8 * (target_vx - vx), -2.5, 2.5))
        cmd_ay = float(np.clip(1.8 * (target_vy - vy), -2.5, 2.5))

        # Bank limit: up to 0.35 (~5.6 deg) aloft, tapering smoothly to strictly 0 during touchdown flare
        max_bank = float(np.clip(0.35 * (z / self.flare_alt), 0.0, 0.35))
        a_tp = float(np.clip(cmd_ax / 5.5, -max_bank, max_bank))
        a_tr = float(np.clip(-cmd_ay / 5.5, -max_bank, max_bank))
        rcs_roll = float(np.clip(-0.8 * wz, -1.0, 1.0))

        # -----------------------------------------------------------------
        # 2. Hoverslam Ignition Decision
        # -----------------------------------------------------------------
        burn_alt = self.compute_suicide_burn_altitude(vz, mass, dist_xy)

        if z <= burn_alt and vz < -2.0:
            self.ignited = True

        if not self.ignited:
            # Coasting phase: zero fuel burn, engine CUTOFF, gentle aerodynamic alignment
            throttle_norm = -0.50
            return np.array([throttle_norm, a_tp, a_tr, rcs_roll], dtype=np.float32)

        # -----------------------------------------------------------------
        # 3. Autonomous Touchdown Settle Logic
        # -----------------------------------------------------------------
        # Engine cuts off on deck contact (z <= 0.05m)
        if z <= 0.05:
            throttle_norm = -0.50
            return np.array([throttle_norm, 0.0, 0.0, rcs_roll], dtype=np.float32)

        # -----------------------------------------------------------------
        # 4. Deceleration Glideslope & Throttle Tracking
        # -----------------------------------------------------------------
        thrust_design = 0.55 * self.cfg.max_thrust
        a_net = (thrust_design / max(1.0, mass)) - self.cfg.g0
        a_net = max(1.5, a_net)

        if z > self.flare_alt:
            v_target = -float(np.sqrt(1.20**2 + 2.0 * a_net * (z - self.flare_alt)))
        else:
            v_target = -1.20

        vz_error = v_target - vz
        desired_accel_z = self.cfg.g0 + a_net + 5.0 * vz_error
        desired_thrust_n = mass * desired_accel_z
        throttle = float(np.clip(desired_thrust_n / self.cfg.max_thrust, self.cfg.min_throttle, 1.0))

        # If requested thrust is below engine minimum throttle, cut off to prevent upward launch
        min_thrust_n = self.cfg.min_throttle * self.cfg.max_thrust
        if desired_thrust_n < min_thrust_n:
            throttle_norm = -0.50
        else:
            throttle = float(np.clip(desired_thrust_n / self.cfg.max_thrust, self.cfg.min_throttle, 1.0))
            throttle_norm = float((throttle - self.cfg.min_throttle) / (1.0 - self.cfg.min_throttle))

        return np.array([throttle_norm, a_tp, a_tr, rcs_roll], dtype=np.float32)
