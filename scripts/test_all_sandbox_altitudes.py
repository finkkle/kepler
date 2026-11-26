import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/Users/PC/Projects/kepler")
sys.path.insert(0, str(PROJECT_ROOT))

from kepler.sim3d.server import session

test_cases = [
    ("Drop 600m with Wind", 600.0, 30.0, 20.0, 5.0),
    ("Drop 800m with Wind", 800.0, 40.0, -25.0, 6.0),
    ("Drop 400m Offset",    400.0, -35.0, 15.0, 4.0),
    ("Drop 250m Normal",    250.0, 15.0, 15.0, 3.0),
    ("Drop 120m Flare",     120.0, -10.0, 8.0, 2.0),
]

for mode_name, mode_id in [("Flight Computer (Autonomous FCS)", "autopilot"), ("Kepler-K2 Neural Agent", "rl")]:
    session.mode = mode_id
    succ = 0
    print(f"\n{'='*75}")
    print(f"  EVALUATING: {mode_name}")
    print(f"{'='*75}")
    for name, alt_init, x_init, y_init, w_init in test_cases:
        session.init_alt = alt_init
        session.init_x = x_init
        session.init_y = y_init
        session.init_wind = w_init
        session.reset()

        for step in range(2500):
            telem = session.step()
            alt = telem['pos'][2]
            vz = telem['vel'][2]
            vh = (telem['vel'][0]**2 + telem['vel'][1]**2)**0.5
            tilt = telem['tilt_deg']
            dist = telem['dist_to_pad']
            st = telem['status']
            if st in ["SUCCESSFUL_LANDING", "CRASH_IMPACT"]:
                is_ok = (st == "SUCCESSFUL_LANDING")
                if is_ok:
                    succ += 1
                res_str = "PASS" if is_ok else "FAIL"
                print(f"  {name:20s} | {res_str:4s} | Dist={dist:5.2f}m | Vz={vz:+5.2f}m/s | Vh={vh:4.2f}m/s | Tilt={tilt:4.2f}° | Fuel={telem['fuel_mass']:.0f}kg", flush=True)
                break
    print(f"  Total Score: {succ}/{len(test_cases)} ({100.0*succ/len(test_cases):.1f}%)\n")
