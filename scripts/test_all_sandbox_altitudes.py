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

session.mode = "rl"

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
        tilt = telem['tilt_deg']
        dist = telem['dist_to_pad']
        st = telem['status']
        if st in ["SUCCESSFUL_LANDING", "CRASH_IMPACT"]:
            print(f"{name:20s} | Result: {st:18s} | Vz={vz:+.2f} m/s | Dist={dist:5.2f} m | Tilt={tilt:4.2f}° | Step={step}", flush=True)
            break
