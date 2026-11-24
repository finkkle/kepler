"""Atmospheric boundary layer wind and stochastic Dryden gust model."""
import numpy as np


class WindField:
    """Atmospheric wind profile with logarithmic shear and stochastic turbulence."""

    def __init__(
        self,
        base_speed: float = 5.0,
        direction_deg: float = 45.0,
        turbulence_intensity: float = 0.15,
        shear_exponent: float = 0.14,
        ref_height: float = 10.0,
        seed: int | None = None
    ):
        self.base_speed = float(base_speed)
        self.direction_rad = np.radians(float(direction_deg))
        self.turbulence_intensity = float(turbulence_intensity)
        self.shear_exponent = float(shear_exponent)
        self.ref_height = float(ref_height)

        self.rng = np.random.default_rng(seed)
        # Gust filter state (Dryden continuous Markov model)
        self.gust = np.zeros(3, dtype=np.float64)
        self.time_constant = 2.0  # Correlation time (seconds)

    def reset(self, base_speed: float | None = None, direction_deg: float | None = None):
        """Reset wind state and randomized direction."""
        if base_speed is not None:
            self.base_speed = float(base_speed)
        if direction_deg is not None:
            self.direction_rad = np.radians(float(direction_deg))
        self.gust = np.zeros(3, dtype=np.float64)

    def get_wind(self, altitude: float, dt: float = 0.02) -> np.ndarray:
        """Compute wind velocity vector [w_x, w_y, w_z] at given altitude."""
        z = max(1.0, float(altitude))
        # Power-law / logarithmic boundary layer wind speed
        mean_speed = self.base_speed * (z / self.ref_height) ** self.shear_exponent

        mean_wx = mean_speed * np.cos(self.direction_rad)
        mean_wy = mean_speed * np.sin(self.direction_rad)
        mean_wz = 0.0

        # Dryden-like low-pass filtered Gaussian noise for turbulence gusts
        sigma = self.turbulence_intensity * max(1.0, mean_speed)
        noise = self.rng.normal(0.0, sigma, size=3)
        decay = np.exp(-dt / self.time_constant)
        self.gust = decay * self.gust + np.sqrt(1.0 - decay**2) * noise

        return np.array([mean_wx + self.gust[0], mean_wy + self.gust[1], mean_wz + self.gust[2]], dtype=np.float64)
