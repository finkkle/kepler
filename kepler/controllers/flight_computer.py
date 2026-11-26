"""Autonomous Flight Computer: Space-Grade Cascaded Guidance and Precision Hoverslam."""
import numpy as np
from kepler.config import BoosterConfig


class KeplerFlightComputer:
    """
    Space-Grade Autonomous Flight Computer:
    1. Kinematic hoverslam ignition solver with lateral reachability buffer.
    2. Continuous single-burn hoverslam with zero engine cutoffs / chattering during burn.
    3. Decoupled square-root deceleration glideslope with smooth terminal flare.
    4. Zero-hover terminal touchdown onto ASDS drone ship landing legs.
    5. Precision lateral guidance corridor with aerodynamic feedforward wind compensation.
    6. Upright vertical touchdown (< 0.5 deg tilt, < 0.5 m/s horizontal drift).
    """

    def __init__(
        self,
        booster_cfg: BoosterConfig | None = None,
        thrust_design_frac: float = 0.55,
        pad_alt: float = 25.0,
        lat_factor: float = 3.5,
        flare_alt: float = 6.0,
        target_vz: float = -1.15,
        pos_gain: float = 0.30,
    ):
        self.cfg = booster_cfg or BoosterConfig()
        self.thrust_design_frac = thrust_design_frac
        self.pad_alt = pad_alt
        self.lat_factor = lat_factor
        self.flare_alt = flare_alt
        self.target_vz = target_vz
        self.pos_gain = pos_gain
        self.ignited = False

    def reset(self):
        """Reset flight computer state for new flight."""
        self.ignited = False

    def compute_suicide_burn_altitude(self, vz: float, mass: float, dist_xy: float = 0.0) -> float:
        """Exact kinematic hoverslam altitude calculation with lateral reachability buffer."""
        if vz >= -1.5:
            return 0.0
        thrust_design = self.thrust_design_frac * self.cfg.max_thrust
        a_net = (thrust_design / max(1.0, mass)) - self.cfg.g0
        a_net = max(1.5, a_net)
        v_flare = abs(self.target_vz)
        d_stop = max(0.0, (vz ** 2 - v_flare ** 2) / (2.0 * a_net))
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
        # 1. Precision Lateral Guidance Corridor & Wind Resistance
        # -----------------------------------------------------------------
        v_cap = float(np.clip(0.12 * z, 0.35, 5.5))
        target_vx = float(np.clip(-self.pos_gain * x, -v_cap, v_cap))
        target_vy = float(np.clip(-self.pos_gain * y, -v_cap, v_cap))

        # Critically damped velocity tracking with wind drift feedforward
        cmd_ax = float(np.clip(1.9 * (target_vx - vx) - 0.04 * wx_wind, -3.0, 3.0))
        cmd_ay = float(np.clip(1.9 * (target_vy - vy) - 0.04 * wy_wind, -3.0, 3.0))

        # Dynamic bank limit: up to 0.45 (~7.2 deg) aloft, tapering strictly to 0 at touchdown
        if z > 30.0:
            max_bank = 0.45
        elif z > self.flare_alt:
            max_bank = float(0.35 + 0.10 * ((z - self.flare_alt) / (30.0 - self.flare_alt)))
        else:
            max_bank = float(0.35 * (z / self.flare_alt))

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
            # Coasting phase: zero fuel burn, engine CUTOFF, grid fin / RCS alignment
            return np.array([-0.50, a_tp, a_tr, rcs_roll], dtype=np.float32)

        # -----------------------------------------------------------------
        # 3. Autonomous Touchdown Settle Logic
        # -----------------------------------------------------------------
        if z <= 0.05:
            return np.array([-0.50, 0.0, 0.0, rcs_roll], dtype=np.float32)

        # -----------------------------------------------------------------
        # 4. Deceleration Glideslope & Continuous Throttle Tracking
        # -----------------------------------------------------------------
        thrust_design = self.thrust_design_frac * self.cfg.max_thrust
        a_net = (thrust_design / max(1.0, mass)) - self.cfg.g0
        a_net = max(1.5, a_net)

        if z > self.flare_alt:
            v_target = -float(np.sqrt(abs(self.target_vz)**2 + 2.0 * a_net * (z - self.flare_alt)))
            a_net_blend = a_net
        else:
            v_target = self.target_vz
            # Blend deceleration overpressure to zero to prevent hovering
            a_net_blend = a_net * (z / self.flare_alt)

        vz_error = v_target - vz
        desired_accel_z = self.cfg.g0 + a_net_blend + 5.0 * vz_error
        desired_thrust_n = mass * desired_accel_z

        # Continuous single-burn hoverslam: throttle never drops below min_throttle during burn
        throttle = float(np.clip(desired_thrust_n / self.cfg.max_thrust, self.cfg.min_throttle, 1.0))
        throttle_norm = float((throttle - self.cfg.min_throttle) / (1.0 - self.cfg.min_throttle))

        return np.array([throttle_norm, a_tp, a_tr, rcs_roll], dtype=np.float32)
