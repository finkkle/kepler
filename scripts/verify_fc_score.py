import sys
from pathlib import Path
PROJECT_ROOT = Path("C:/Users/PC/Projects/kepler")
sys.path.insert(0, str(PROJECT_ROOT))

from kepler.controllers.flight_computer import KeplerFlightComputer
from scripts.verify_goal_metrics import audit_controller

fc = KeplerFlightComputer()
ok = audit_controller("Kepler Flight Computer (Space-Grade)", fc, is_model=False)
print(f"Audit Result: {'PASSED 30/30' if ok else 'FAILED'}")
