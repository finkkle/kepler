"""Autonomous Flight Computer: Frontier Cascaded Guidance, Aerodynamic Crossflow Compensation, and Precision Hoverslam."""
from __future__ import annotations
import numpy as np
from kepler.config import BoosterConfig
from kepler.physics.atmosphere import Atmosphere


class KeplerFlightComputer:
    """
    Space-Grade Autonomous Flight Computer (Kepler-K3 Frontier Champion):
    1. Kinematic hoverslam ignition solver with adaptive lateral reachability buffer.
    2. Dynamic crossflow aerodynamic feedforward for severe wind and crossflow drag rejection.
    3. Decoupled TVC thrust vectoring (cos(tilt) scaling) preventing vertical braking starvation during banking.
    4. Dual-loop proportional lateral guidance with integral wind-drift centering corridor.
    5. Retained terminal crab authority floor down to touchdown to cancel residual wind gusts on ASDS drone ship.
    6. Decoupled square-root deceleration glideslope with continuous single-burn zero-hover touchdown.
    """

    def __init__(
        self,
        booster_cfg: BoosterConfig | None = None,
        thrust_design_frac: float = 0.55,
        pad_alt: float = 24.0,
        lat_factor: float = 2.4,
        flare_alt: float = 5.5,
        target_vz: float = -1.22,
        pos_gain: float = 0.32,
    ):
        self.cfg = booster_cfg or BoosterConfig()
        self.thrust_design_frac = thrust_design_frac
        self.pad_alt = pad_alt
        self.lat_factor = lat_factor
        self.flare_alt = flare_alt
        self.target_vz = target_vz
        self.pos_gain = pos_gain
        self.ignited = False
        self.int_x = 0.0
        self.int_y = 0.0

    def reset(self):
        """Reset flight computer state for new flight."""
        self.ignited = False
        self.int_x = 0.0
        self.int_y = 0.0

    def compute_suicide_burn_altitude(self, vz: float, mass: float, dist_xy: float = 0.0) -> float:
        """Exact kinematic hoverslam altitude calculation with adaptive lateral reachability buffer."""
        if vz >= -1.5:
            return 0.0
        thrust_design = self.thrust_design_frac * self.cfg.max_thrust
        a_net = (thrust_design / max(1.0, mass)) - self.cfg.g0
        a_net = max(1.5, a_net)
        v_flare = abs(self.target_vz)
        d_stop = max(0.0, (vz ** 2 - v_flare ** 2) / (2.0 * a_net))
        # Adaptive lateral buffer capped at 35m to prevent burn initiation starvation
        d_lateral = self.lat_factor * min(dist_xy, 35.0)
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
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])
        fuel_frac = float(obs[14])
        wx_wind = float(obs[15] * 20.0)
        wy_wind = float(obs[16] * 20.0)

        fuel_mass = fuel_frac * self.cfg.default_fuel
        mass = self.cfg.dry_mass + fuel_mass
        dist_xy = float(np.hypot(x, y))

        # -----------------------------------------------------------------
        # 1. Dynamic Wind Aerodynamic Disturbance Rejection
        # -----------------------------------------------------------------
        rho = Atmosphere.density(max(0.0, z))
        w_speed = float(np.hypot(wx_wind, wy_wind))
        cd_cross = self.cfg.cd_cross
        area = self.cfg.cross_section_area
        a_wind_x = 0.5 * rho * w_speed * wx_wind * cd_cross * area / mass
        a_wind_y = 0.5 * rho * w_speed * wy_wind * cd_cross * area / mass

        # -----------------------------------------------------------------
        # 2. Precision Dual-Loop Lateral Guidance Corridor
        # -----------------------------------------------------------------
        v_cap = float(np.clip(0.12 * z + 0.30, 0.40, 5.8))
        target_vx = float(np.clip(-self.pos_gain * x, -v_cap, v_cap))
        target_vy = float(np.clip(-self.pos_gain * y, -v_cap, v_cap))

        # Dynamic integral position trim near the pad
        dt = 0.02
        if dist_xy < 5.0 and z < 30.0:
            self.int_x = float(np.clip(self.int_x + (-x) * dt, -1.0, 1.0))
            self.int_y = float(np.clip(self.int_y + (-y) * dt, -1.0, 1.0))
        else:
            self.int_x *= 0.85
            self.int_y *= 0.85

        kp_v = 2.3
        ki_p = 0.12
        cmd_ax = float(np.clip(kp_v * (target_vx - vx) + ki_p * self.int_x - 0.85 * a_wind_x - 0.03 * wx_wind, -3.5, 3.5))
        cmd_ay = float(np.clip(kp_v * (target_vy - vy) + ki_p * self.int_y - 0.85 * a_wind_y - 0.03 * wy_wind, -3.5, 3.5))

        # Dynamic bank limit with retained crab authority floor down to touchdown
        if z > 35.0:
            max_bank = 0.50
        elif z > self.flare_alt:
            max_bank = float(0.35 + 0.15 * ((z - self.flare_alt) / (35.0 - self.flare_alt)))
        else:
            # Maintain active crab authority down to touchdown to counter surface crossflow
            w_norm = min(1.0, w_speed / 10.0)
            crab_floor = float(0.08 + 0.15 * w_norm)
            max_bank = float(max(crab_floor, 0.35 * (z / self.flare_alt)))

        a_tp = float(np.clip(cmd_ax / 5.2, -max_bank, max_bank))
        a_tr = float(np.clip(-cmd_ay / 5.2, -max_bank, max_bank))
        rcs_roll = float(np.clip(-0.8 * wz, -1.0, 1.0))

        # -----------------------------------------------------------------
        # 3. Hoverslam Ignition Decision
        # -----------------------------------------------------------------
        burn_alt = self.compute_suicide_burn_altitude(vz, mass, dist_xy)
        if z <= burn_alt and vz < -2.0:
            self.ignited = True

        if not self.ignited:
            # Coasting phase: zero fuel burn, engine CUTOFF, grid fin / RCS alignment
            return np.array([-0.50, a_tp, a_tr, rcs_roll], dtype=np.float32)

        # -----------------------------------------------------------------
        # 4. Autonomous Touchdown Settle Logic
        # -----------------------------------------------------------------
        if z <= 0.05:
            return np.array([-0.50, 0.0, 0.0, rcs_roll], dtype=np.float32)

        # -----------------------------------------------------------------
        # 5. Deceleration Glideslope & Decoupled TVC Thrust Allocation
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

        # Decouple vertical braking from TVC bank angle (cos(tilt) scaling)
        tilt_cmd_rad = np.hypot(a_tp, a_tr) * np.radians(16.0)
        cos_tilt = float(np.cos(tilt_cmd_rad))
        cos_tilt = max(0.92, cos_tilt)
        desired_thrust_n = (mass * desired_accel_z) / cos_tilt

        # Continuous single-burn hoverslam: throttle never drops below min_throttle during burn
        throttle = float(np.clip(desired_thrust_n / self.cfg.max_thrust, self.cfg.min_throttle, 1.0))
        throttle_norm = float((throttle - self.cfg.min_throttle) / (1.0 - self.cfg.min_throttle))

        return np.array([throttle_norm, a_tp, a_tr, rcs_roll], dtype=np.float32)
