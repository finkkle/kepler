"""Test script for evaluating terminal guidance."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from kepler.envs.landing_env import KeplerLandingEnv
from kepler.controllers.flight_computer import KeplerFlightComputer

env = KeplerLandingEnv(difficulty="terminal_flare", wind_enabled=False)
fc = KeplerFlightComputer()

successes = 0
for ep in range(10):
    fc.reset()
    obs, info = env.reset(seed=100 + ep)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(fc.get_action(obs))
        done = term or trunc

    dist = np.linalg.norm(info['pos'][:2])
    vz = info['vel'][2]
    vh = np.linalg.norm(info['vel'][:2])
    tilt = info['tilt_deg']
    s = info.get('landing_success', False)
    if s:
        successes += 1
    print(f"Flight {ep+1:02d}: Success={s} | Vz={vz:+.2f} m/s | Vh={vh:.2f} m/s | Dist={dist:5.2f} m | Tilt={tilt:.2f} deg")

print(f"\nTerminal Flare Success Rate: {successes}/10 ({100.0 * successes / 10:.1f}%)")
