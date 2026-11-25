"""Auditing Kepler-K1 on Adverse Atmospheric & Drift Conditions across Difficulties."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/Users/PC/Projects/kepler")
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from kepler.envs.landing_env import KeplerLandingEnv

def test_adverse():
    model_path = PROJECT_ROOT / "checkpoints" / "kepler_k1.zip"
    print(f"Loading {model_path}...")
    model = PPO.load(str(model_path))

    scenarios = [
        ("Easy + Wind (0-3 m/s)", "easy", True, 10),
        ("Normal + Wind (2-8 m/s)", "normal", True, 10),
    ]

    total_pass = 0
    total_flights = 0

    print("=" * 75)
    print("  KEPLER-K1 ADVERSE CONDITION ROBUSTNESS EVALUATION")
    print("=" * 75)

    for sc_name, diff, wind, num_trials in scenarios:
        env = KeplerLandingEnv(difficulty=diff, wind_enabled=wind)
        passes = 0
        print(f"\n--- Scenario: {sc_name} ({num_trials} flights) ---")
        for i in range(num_trials):
            obs, info = env.reset(seed=8000 + i * 17)
            done = False
            steps = 0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
                steps += 1

            s = bool(info.get("landing_success", False))
            if s:
                passes += 1
                total_pass += 1
            total_flights += 1

            d = float(np.linalg.norm(info["pos"][:2]))
            vz = float(info["vel"][2])
            vh = float(np.linalg.norm(info["vel"][:2]))
            tilt = float(info["tilt_deg"])
            fuel = float(info["fuel_used"])
            res = "PASS" if s else "FAIL"
            print(f"  Trial {i+1:02d} | {res:4s} | Dist={d:5.2f}m | Vz={vz:+5.2f}m/s | Vh={vh:4.2f}m/s | Tilt={tilt:4.2f}° | Fuel={fuel:4.0f}kg | Steps={steps}")

        print(f"  >> Score: {passes}/{num_trials} ({100.0*passes/num_trials:.1f}%)")

    print("\n" + "=" * 75)
    print(f"  TOTAL ADVERSE ROBUSTNESS SCORE: {total_pass}/{total_flights} ({100.0*total_pass/total_flights:.1f}%)")
    print("=" * 75)
    return total_pass, total_flights

if __name__ == "__main__":
    test_adverse()
