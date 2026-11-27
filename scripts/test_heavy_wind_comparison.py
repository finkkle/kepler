import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kepler.sim3d.server import session

print("=" * 80)
print("  ADVERSE CROSSWIND & GUST STRESS AUDIT (8 m/s, 10 m/s, 12 m/s, 15 m/s)")
print("=" * 80)

wind_tests = [
    ("Wind  8 m/s + 30m offset", 600.0, 30.0, 20.0, 8.0),
    ("Wind 10 m/s + 20m offset", 600.0, -20.0, 15.0, 10.0),
    ("Wind 12 m/s + 25m offset", 600.0, 25.0, -20.0, 12.0),
    ("Wind 15 m/s Extreme Gust", 600.0, 15.0, 10.0, 15.0),
]

controllers = [
    ("Kepler-K3 (Frontier Champion)", "k3_champion"),
    ("Kepler-K3² (High-Performance Runner-Up)", "k3_runnerup"),
    ("Kepler-K2 (Zero-Hover Policy)", "rl_k2"),
    ("Kepler-K1 Neural Agent", "rl_k1"),
]

for controller_name, mode_id in controllers:
    session.mode = mode_id
    succs = 0
    print(f"\n--- Testing {controller_name} (Mode: {mode_id}) ---")
    for name, alt, x, y, w in wind_tests:
        session.init_alt = alt
        session.init_x = x
        session.init_y = y
        session.init_wind = w
        session.reset()
        for step in range(2500):
            telem = session.step()
            st = telem['status']
            if st in ["SUCCESSFUL_LANDING", "CRASH_IMPACT"]:
                is_ok = (st == "SUCCESSFUL_LANDING")
                if is_ok:
                    succs += 1
                dist = telem['dist_to_pad']
                vz = telem['vel'][2]
                vh = (telem['vel'][0]**2 + telem['vel'][1]**2)**0.5
                tilt = telem['tilt_deg']
                fuel = telem['fuel_mass']
                res = "PASS" if is_ok else "FAIL"
                print(f"  {name:25s} | {res:4s} | Dist={dist:5.2f}m | Vz={vz:+5.2f}m/s | Vh={vh:4.2f}m/s | Tilt={tilt:4.2f}° | Fuel={fuel:.0f}kg", flush=True)
                break
    print(f"  Total Score: {succs}/{len(wind_tests)} ({100.0*succs/len(wind_tests):.1f}%)")
