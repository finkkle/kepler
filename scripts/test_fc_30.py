import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/Users/PC/Projects/kepler")
sys.path.insert(0, str(PROJECT_ROOT))

from kepler.envs.landing_env import KeplerLandingEnv
from kepler.controllers.flight_computer import KeplerFlightComputer

env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
fc = KeplerFlightComputer()

successes = 0
results = []

for i in range(30):
    fc.reset()
    obs, info = env.reset(seed=5000 + i)
    done = False
    step = 0
    while not done and step < 2500:
        act = fc.get_action(obs)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        step += 1

    s = bool(info.get("landing_success", False))
    if s:
        successes += 1
    vz = info["vel"][2]
    dist = np.linalg.norm(info["pos"][:2])
    tilt = info["tilt_deg"]
    fuel = info["fuel_mass"]
    stat = "SUCCESS" if s else "FAILED"
    print(f"Flight {i+1:02d} | {stat:7s} | Alt_init={env.initial_altitude:5.1f}m | Vz={vz:+.2f} m/s | Dist={dist:5.2f}m | Tilt={tilt:4.2f}° | Fuel={fuel:4.0f}kg | Step={step}")

print("=" * 60)
print(f"FC Scorecard: {successes}/30 ({100.0 * successes / 30:.1f}%)")
