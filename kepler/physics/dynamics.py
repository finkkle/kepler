"""Vectorized 6-DOF rigid-body flight dynamics and Runge-Kutta 4th order integrator."""
from dataclasses import dataclass, field
import numpy as np

from kepler.config import BoosterConfig, SimConfig
from kepler.physics.quaternion import (
    quat_normalize,
    quat_rotate_vector,
    quat_to_rot_matrix,
    quat_derivative,
    tilt_angle_from_vertical,
    euler_to_quat,
    quat_to_euler,
)
from kepler.physics.atmosphere import Atmosphere
from kepler.physics.wind import WindField


@dataclass
class RocketState:
    """Complete instantaneous 6-DOF dynamic state of the booster."""
    # Inertial frame position [x, y, z] (meters) - z=0 is ground/pad
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    # Inertial frame velocity [vx, vy, vz] (m/s)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    # Orientation quaternion [qw, qx, qy, qz] (Body to World)
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    # Angular velocity in body frame [wx, wy, wz] (rad/s)
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    # Propellant mass remaining (kg)
    fuel_mass: float = 8000.0

    # Actuator states (subject to rate limits)
    throttle: float = 0.0             # Current throttle [0.0 or min_throttle..1.0]
    gimbal_y: float = 0.0             # Current gimbal pitch (rad)
    gimbal_z: float = 0.0             # Current gimbal yaw (rad)
    rcs_cmd: float = 0.0              # Current RCS roll torque (-1 to 1)

    # Telemetry records
    thrust_magnitude: float = 0.0
    mass_total: float = 33000.0
    tilt_rad: float = 0.0
    ground_contact: bool = False


class RocketDynamics:
    """6-DOF Rocket flight dynamics engine with RK4 numerical solver."""

    def __init__(self, booster_cfg: BoosterConfig | None = None, sim_cfg: SimConfig | None = None):
        self.cfg = booster_cfg or BoosterConfig()
        self.sim_cfg = sim_cfg or SimConfig()
        self.wind = WindField()
        self.state = RocketState()

    def get_mass_properties(self, fuel: float) -> tuple[float, float, np.ndarray]:
        """Compute instantaneous total mass, center of mass, and diagonal inertia tensor."""
        fuel_clamped = max(0.0, min(fuel, self.cfg.max_landing_fuel))
        m_total = self.cfg.dry_mass + fuel_clamped
        fuel_frac = fuel_clamped / max(1.0, self.cfg.max_landing_fuel)

        # Center of mass shifts downward as fuel empties (fuel tank is in lower section)
        z_cm = self.cfg.cm_empty * (1.0 - fuel_frac) + self.cfg.cm_full * fuel_frac

        # Inertia tensor approx for cylinder
        j_roll = m_total * self.cfg.j_roll_factor
        j_pitch = m_total * self.cfg.j_pitch_factor
        inertia = np.array([j_pitch, j_pitch, j_roll], dtype=np.float64)

        return m_total, z_cm, inertia

    def compute_derivatives(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        fuel: float,
        throttle_cmd: float,
        gimbal_y_cmd: float,
        gimbal_z_cmd: float,
        rcs_cmd: float,
        wind_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute state time derivatives [dpos/dt, dvel/dt, dquat/dt, domega/dt, dfuel/dt]."""
        m_total, z_cm, inertia = self.get_mass_properties(fuel)
        q_norm = quat_normalize(quat)
        rot_mat = quat_to_rot_matrix(q_norm)

        # 1. Propulsion: Main Engine
        # Engine is active only if fuel remains and throttle is requested
        if fuel > 0.0 and throttle_cmd > 0.05:
            # Enforce throttle range [min_throttle, 1.0]
            eff_throttle = np.clip(throttle_cmd, self.cfg.min_throttle, self.cfg.max_throttle)
            altitude = max(0.0, pos[2])
            # Isp interpolates with ambient air density / pressure
            rho = Atmosphere.density(altitude)
            isp = self.cfg.isp_sea_level + (self.cfg.isp_vacuum - self.cfg.isp_sea_level) * (1.0 - min(1.0, rho / Atmosphere.RHO_0))
            thrust_n = eff_throttle * self.cfg.max_thrust
            fuel_burn_rate = thrust_n / (isp * self.cfg.g0)
        else:
            eff_throttle = 0.0
            thrust_n = 0.0
            fuel_burn_rate = 0.0

        # Gimbal angles: delta_pitch (around body Y) and delta_yaw (around body X)
        g_pitch = np.clip(gimbal_y_cmd, -self.cfg.max_gimbal_angle, self.cfg.max_gimbal_angle)
        g_yaw = np.clip(gimbal_z_cmd, -self.cfg.max_gimbal_angle, self.cfg.max_gimbal_angle)

        # Deflection in body X and Y
        # A deflection along X produces torque about Y (pitch)
        # A deflection along Y produces torque about X (yaw)
        sin_p = np.sin(g_pitch)
        sin_y = np.sin(g_yaw)
        cos_tot = np.sqrt(max(0.01, 1.0 - sin_p**2 - sin_y**2))

        t_dir_body = np.array([sin_p, sin_y, cos_tot], dtype=np.float64)
        f_thrust_body = thrust_n * t_dir_body

        # Thrust vector in world coordinates
        f_thrust_world = rot_mat @ f_thrust_body

        # Thrust torque around center of mass in body frame
        # Engine is at z = 0, so vector from CM to engine is [0, 0, -z_cm]
        # tau = r x F = [0, 0, -z_cm] x [Fx, Fy, Fz]
        # tau_x = -(-z_cm) * Fy = +z_cm * Fy
        # tau_y = +(-z_cm) * Fx = -z_cm * Fx
        # tau_z = 0
        tau_thrust_body = np.array([
            z_cm * f_thrust_body[1],
            -z_cm * f_thrust_body[0],
            0.0
        ], dtype=np.float64)

        # 2. Reaction Control System (RCS) torque
        # RCS nozzles provide roll torque and attitude fine-trim
        tau_rcs_body = np.array([0.0, 0.0, rcs_cmd * self.cfg.rcs_thrust * self.cfg.radius], dtype=np.float64)

        # 3. Aerodynamics
        v_rel_world = vel - wind_vel
        speed_rel = np.linalg.norm(v_rel_world)
        if speed_rel > 0.1:
            rho = Atmosphere.density(pos[2])
            q_inf = 0.5 * rho * (speed_rel ** 2)

            # Transform relative velocity to body frame
            v_rel_body = rot_mat.T @ v_rel_world
            v_unit_body = v_rel_body / speed_rel

            # Axial drag vs crossflow drag
            # Longitudinal axis is +Z
            cos_alpha = np.clip(v_unit_body[2], -1.0, 1.0)
            sin_alpha = np.sqrt(max(0.0, 1.0 - cos_alpha**2))

            # Drag force in body frame opposed to relative velocity
            cd_eff = self.cfg.cd_base * np.abs(cos_alpha) + self.cfg.cd_cross * sin_alpha
            f_aero_mag = q_inf * cd_eff * self.cfg.cross_section_area
            f_aero_world = -f_aero_mag * (v_rel_world / speed_rel)

            # Aerodynamic moment about CM (restoring moment from grid fins/center of pressure)
            r_cp_cm = np.array([0.0, 0.0, self.cfg.cp_nominal - z_cm], dtype=np.float64)
            f_aero_body = rot_mat.T @ f_aero_world
            tau_aero_body = np.cross(r_cp_cm, f_aero_body)
        else:
            f_aero_world = np.zeros(3, dtype=np.float64)
            tau_aero_body = np.zeros(3, dtype=np.float64)

        # 4. Total Force & Translational Acceleration
        f_grav_world = np.array([0.0, 0.0, -m_total * self.cfg.g0], dtype=np.float64)
        f_net_world = f_thrust_world + f_aero_world + f_grav_world
        acc_world = f_net_world / m_total

        # 5. Total Torque & Rotational Acceleration (Euler's Equations)
        tau_total_body = tau_thrust_body + tau_rcs_body + tau_aero_body
        # J * d_omega/dt + omega x (J * omega) = tau
        omega_cross_j_omega = np.cross(omega, inertia * omega)
        d_omega = (tau_total_body - omega_cross_j_omega) / inertia

        # 6. Kinematic derivatives
        d_pos = vel.copy()
        d_vel = acc_world
        d_quat = quat_derivative(q_norm, omega)
        d_fuel = -fuel_burn_rate

        return d_pos, d_vel, d_quat, d_omega, d_fuel

    def step(
        self,
        throttle_cmd: float,
        gimbal_y_cmd: float,
        gimbal_z_cmd: float,
        rcs_cmd: float,
        dt: float | None = None
    ) -> RocketState:
        """Advance dynamics by timestep dt using 4th-order Runge-Kutta integration."""
        step_dt = dt or self.sim_cfg.dt
        sub_dt = step_dt / self.sim_cfg.substeps

        s = self.state
        wind_vel = self.wind.get_wind(s.pos[2], dt=step_dt)

        for _ in range(self.sim_cfg.substeps):
            # Check ground contact
            if s.pos[2] <= 0.0 and s.vel[2] <= 0.0:
                s.ground_contact = True
                s.pos[2] = 0.0
                s.vel = np.zeros(3, dtype=np.float64)
                s.omega = np.zeros(3, dtype=np.float64)
                break

            # RK4 Stage 1
            k1_p, k1_v, k1_q, k1_w, k1_f = self.compute_derivatives(
                s.pos, s.vel, s.quat, s.omega, s.fuel_mass,
                throttle_cmd, gimbal_y_cmd, gimbal_z_cmd, rcs_cmd, wind_vel
            )

            # RK4 Stage 2
            p2 = s.pos + 0.5 * sub_dt * k1_p
            v2 = s.vel + 0.5 * sub_dt * k1_v
            q2 = quat_normalize(s.quat + 0.5 * sub_dt * k1_q)
            w2 = s.omega + 0.5 * sub_dt * k1_w
            f2 = max(0.0, s.fuel_mass + 0.5 * sub_dt * k1_f)
            k2_p, k2_v, k2_q, k2_w, k2_f = self.compute_derivatives(
                p2, v2, q2, w2, f2, throttle_cmd, gimbal_y_cmd, gimbal_z_cmd, rcs_cmd, wind_vel
            )

            # RK4 Stage 3
            p3 = s.pos + 0.5 * sub_dt * k2_p
            v3 = s.vel + 0.5 * sub_dt * k2_v
            q3 = quat_normalize(s.quat + 0.5 * sub_dt * k2_q)
            w3 = s.omega + 0.5 * sub_dt * k2_w
            f3 = max(0.0, s.fuel_mass + 0.5 * sub_dt * k2_f)
            k3_p, k3_v, k3_q, k3_w, k3_f = self.compute_derivatives(
                p3, v3, q3, w3, f3, throttle_cmd, gimbal_y_cmd, gimbal_z_cmd, rcs_cmd, wind_vel
            )

            # RK4 Stage 4
            p4 = s.pos + sub_dt * k3_p
            v4 = s.vel + sub_dt * k3_v
            q4 = quat_normalize(s.quat + sub_dt * k3_q)
            w4 = s.omega + sub_dt * k3_w
            f4 = max(0.0, s.fuel_mass + sub_dt * k3_f)
            k4_p, k4_v, k4_q, k4_w, k4_f = self.compute_derivatives(
                p4, v4, q4, w4, f4, throttle_cmd, gimbal_y_cmd, gimbal_z_cmd, rcs_cmd, wind_vel
            )

            # Combine RK4 updates
            s.pos += (sub_dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
            s.vel += (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            s.quat = quat_normalize(s.quat + (sub_dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q))
            s.omega += (sub_dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
            s.fuel_mass = max(0.0, s.fuel_mass + (sub_dt / 6.0) * (k1_f + 2.0 * k2_f + 2.0 * k3_f + k4_f))

            if s.pos[2] <= 0.0:
                s.pos[2] = 0.0
                s.ground_contact = True
                break

        # Update telemetry
        s.throttle = float(throttle_cmd if s.fuel_mass > 0.0 else 0.0)
        s.gimbal_y = float(gimbal_y_cmd)
        s.gimbal_z = float(gimbal_z_cmd)
        s.rcs_cmd = float(rcs_cmd)
        m_tot, _, _ = self.get_mass_properties(s.fuel_mass)
        s.mass_total = float(m_tot)
        s.tilt_rad = float(tilt_angle_from_vertical(s.quat))
        s.thrust_magnitude = float(s.throttle * self.cfg.max_thrust if s.fuel_mass > 0.0 and s.throttle > 0.05 else 0.0)

        return s

    def reset(
        self,
        altitude: float = 1000.0,
        lateral_offset: tuple[float, float] = (0.0, 0.0),
        initial_vel: tuple[float, float, float] = (0.0, 0.0, -60.0),
        initial_tilt_deg: float = 5.0,
        fuel_mass: float | None = None
    ) -> RocketState:
        """Reset booster state to specified initial conditions."""
        self.state = RocketState()
        self.state.pos = np.array([lateral_offset[0], lateral_offset[1], altitude], dtype=np.float64)
        self.state.vel = np.array(initial_vel, dtype=np.float64)

        tilt_rad = np.radians(initial_tilt_deg)
        # Random initial tilt orientation azimuth
        azimuth = np.random.uniform(0.0, 2.0 * np.pi)
        roll = tilt_rad * np.cos(azimuth)
        pitch = tilt_rad * np.sin(azimuth)
        self.state.quat = euler_to_quat(roll, pitch, 0.0)

        self.state.omega = np.random.uniform(-0.02, 0.02, size=3)
        self.state.fuel_mass = self.cfg.default_fuel if fuel_mass is None else float(fuel_mass)
        self.state.ground_contact = False

        m_tot, _, _ = self.get_mass_properties(self.state.fuel_mass)
        self.state.mass_total = float(m_tot)
        self.state.tilt_rad = float(tilt_angle_from_vertical(self.state.quat))

        return self.state
