"""Booster parameters, physics constants, and landing thresholds for Kepler."""
from dataclasses import dataclass
import numpy as np


@dataclass
class BoosterConfig:
    """Booster physical, propulsive, and aerodynamic specifications."""

    # Geometry
    height: float = 50.0            # Total height (meters)
    diameter: float = 3.66          # Cylinder diameter (meters)
    radius: float = 1.83            # Radius (meters)
    cross_section_area: float = np.pi * (1.83 ** 2)  # Base frontal area (~10.52 m^2)

    # Mass Properties
    dry_mass: float = 25000.0       # Structural dry mass (kg)
    max_landing_fuel: float = 12000.0  # Max fuel allocated for landing burn (kg)
    default_fuel: float = 8000.0    # Typical starting fuel for descent (kg)

    # Moments of Inertia (approx cylinder around CM)
    # J_axis = 0.5 * m * r^2, J_trans = (1/12) * m * (3*r^2 + h^2)
    j_roll_factor: float = 0.5 * (1.83 ** 2)
    j_pitch_factor: float = (1.0 / 12.0) * (3.0 * (1.83 ** 2) + (50.0 ** 2))

    # Center of Mass & Pressure (meters from engine gimbal at base z=0)
    cm_empty: float = 23.0          # Center of mass when dry (meters above engine)
    cm_full: float = 17.0           # Center of mass when landing fuel full (meters)
    cp_nominal: float = 28.0        # Center of aerodynamic pressure (stabilized by grid fins)

    # Propulsion (Merlin 1D Vacuum/Sea-Level Class)
    max_thrust: float = 845000.0    # 845 kN max thrust at sea level
    min_throttle: float = 0.40      # Minimum engine throttle (40%)
    max_throttle: float = 1.00      # 100%
    isp_sea_level: float = 282.0    # Specific impulse at sea level (seconds)
    isp_vacuum: float = 311.0       # Specific impulse in vacuum (seconds)
    g0: float = 9.80665             # Standard Earth gravity (m/s^2)

    # Thrust Vector Control (Gimbal)
    max_gimbal_angle: float = np.radians(10.0)  # Max gimbal deflection (+/- 10 deg)
    gimbal_slew_rate: float = np.radians(25.0)  # Gimbal angular velocity limit (rad/s)

    # Reaction Control System (Cold-gas thrusters at rocket top)
    rcs_thrust: float = 2000.0      # Thrust per RCS nozzle (N)
    rcs_arm: float = 46.0           # Distance from engine base to RCS nozzles (m)

    # Aerodynamics
    cd_base: float = 0.82           # Drag coefficient along cylinder axis (retro-propulsion)
    cd_cross: float = 1.20          # Crossflow drag coefficient (broadside)

    # Landing Gear & Pad
    gear_span: float = 18.0         # Landing leg footprint diameter (m)
    gear_height: float = 3.5        # Height of engine above ground when legs compressed (m)
    pad_radius: float = 15.0        # Target landing pad radius (m)
    bullseye_radius: float = 3.0    # Bullseye precision target radius (m)

    # Success Criteria
    max_touchdown_vz: float = 2.0   # Max allowable vertical descent velocity (m/s)
    max_touchdown_vh: float = 1.6   # Max allowable horizontal velocity (m/s)
    max_touchdown_tilt: float = np.radians(5.0)  # Max allowable tilt from vertical (deg)
    max_touchdown_omega: float = np.radians(5.0) # Max allowable angular velocity (rad/s)
    max_touchdown_dist: float = 10.0  # Max distance from pad center (m)


@dataclass
class SimConfig:
    """Simulation time-step and physics integration constants."""
    dt: float = 0.02                # Simulation timestep (50 Hz)
    substeps: int = 2               # Substeps per step (100 Hz internal RK4)
    max_flight_time: float = 60.0   # Max episode duration in seconds
