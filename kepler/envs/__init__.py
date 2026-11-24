"""Kepler Gymnasium Environments."""
from gymnasium.envs.registration import register
from kepler.envs.landing_env import KeplerLandingEnv

register(
    id="KeplerLanding-v1",
    entry_point="kepler.envs.landing_env:KeplerLandingEnv",
    max_episode_steps=3000,
)

__all__ = ["KeplerLandingEnv"]
