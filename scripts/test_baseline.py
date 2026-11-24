"""Test script for evaluating flight computer baseline."""
import sys
import numpy as np

sys.path.insert(0, "C:/Users/PC/Projects/kepler")
from kepler.envs.landing_env import KeplerLandingEnv
from kepler.controllers.flight_computer import KeplerFlightComputer

env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
fc = KeplerFlightComputer()

successes = 0
total_flights = 10

for ep in range(total_flights):
    fc.reset()
    obs, info = env.reset(seed=100 + ep)
    done = False
    steps = 0
    total_r = 0.0

    while not done and steps < 2000:
        action = fc.get_action(obs)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        steps += 1
        done = term or trunc

    if info.get("landing_success", False):
        successes += 1
    vz = info['vel'][2]
    dist = np.linalg.norm(info['pos'][:2])
    print(f"Flight {ep+1:02d}: Success={info.get('landing_success')}, Vz={vz:+.2f} m/s, Dist={dist:.2f} m, Tilt={info['tilt_deg']:.2f} deg, Fuel Left={info['fuel_mass']:.1f} kg")

print(f"\nBaseline Touchdown Success Rate: {successes}/{total_flights} ({100 * successes / total_flights:.1f}%)")
