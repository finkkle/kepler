"""Atmospheric density, pressure, and thermodynamic properties."""
import numpy as np


class Atmosphere:
    """U.S. Standard Atmosphere model for tropospheric and stratospheric altitudes."""

    RHO_0 = 1.225         # Sea level density (kg/m^3)
    P_0 = 101325.0        # Sea level pressure (Pa)
    T_0 = 288.15          # Sea level temperature (K)
    LAPSE_RATE = 0.0065   # Temperature lapse rate (K/m)
    SCALE_HEIGHT = 8430.0 # Scale height (m)
    GAMMA = 1.4           # Heat capacity ratio of air
    R_SPECIFIC = 287.05   # Specific gas constant for air (J/(kg*K))

    @classmethod
    def density(cls, altitude: float) -> float:
        """Calculate air density at geometric altitude z (meters)."""
        z = max(0.0, float(altitude))
        if z < 11000.0:
            temp = cls.T_0 - cls.LAPSE_RATE * z
            pressure = cls.P_0 * (temp / cls.T_0) ** (cls.GAMMA / (cls.GAMMA - 1.0))
            return max(1e-6, pressure / (cls.R_SPECIFIC * temp))
        else:
            return cls.RHO_0 * np.exp(-z / cls.SCALE_HEIGHT)

    @classmethod
    def speed_of_sound(cls, altitude: float) -> float:
        """Calculate local speed of sound in m/s."""
        z = max(0.0, float(altitude))
        temp = max(200.0, cls.T_0 - cls.LAPSE_RATE * min(z, 11000.0))
        return np.sqrt(cls.GAMMA * cls.R_SPECIFIC * temp)
