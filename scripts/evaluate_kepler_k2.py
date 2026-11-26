"""Comprehensive Benchmark Evaluation Script for Kepler-K2 Neural Flight Controller."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from kepler.envs.landing_env import KeplerLandingEnv
from kepler.config import BoosterConfig

def evaluate_kepler_k2(model_path: str = None, num_flights: int = 30):
    if model_path is None:
        target_path = PROJECT_ROOT / "checkpoints" / "kepler_k2.zip"
    else:
        target_path = Path(model_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path

    print("=" * 75)
    print("  KEPLER-K2 CONTINUOUS HOVERSLAM BENCHMARK SCORECARD")
    print("=" * 75)
    print(f"  Model:   {target_path}")
    print(f"  Flights: {num_flights} randomized test scenarios (Seeds 5000–{5000+num_flights-1})")
    print("=" * 75)

    model = PPO.load(str(target_path))
    env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)

    results = []
    successes = 0

    for i in range(num_flights):
        obs, info = env.reset(seed=5000 + i)
        done = False
        steps = 0
        hover_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            steps += 1
            done = term or trunc

            z = env.dynamics.state.pos[2]
            vz = env.dynamics.state.vel[2]
            if z <= 6.0 and abs(vz) < 0.5:
                hover_steps += 1

        pos = info["pos"]
        vel = info["vel"]
        vz = vel[2]
        vh = np.linalg.norm(vel[:2])
        dist = np.linalg.norm(pos[:2])
        tilt = info["tilt_deg"]
        fuel_left = info["fuel_mass"]
        fuel_used = info["fuel_used"]
        hover_time = hover_steps * 0.02
        flight_time = steps * 0.02
        success = bool(info.get("landing_success", False))

        if success:
            successes += 1

        results.append({
            "flight": i + 1,
            "seed": 5000 + i,
            "success": success,
            "vz": vz,
            "vh": vh,
            "dist": dist,
            "tilt": tilt,
            "fuel_left": fuel_left,
            "fuel_used": fuel_used,
            "hover_time": hover_time,
            "flight_time": flight_time,
            "steps": steps,
        })

        status_str = "SUCCESS" if success else "CRASH/MISS"
        print(f" Flight {i+1:02d} | {status_str:10s} | Vz: {vz:+.2f} m/s | Vh: {vh:4.2f} m/s | Dist: {dist:5.2f} m | Tilt: {tilt:4.2f}° | Hover: {hover_time:4.2f}s | Fuel: {fuel_left:5.0f} kg ({fuel_used:.0f} kg used)")

    success_rate = 100.0 * successes / num_flights
    mean_vz = np.mean([r["vz"] for r in results])
    mean_vh = np.mean([r["vh"] for r in results])
    mean_dist = np.mean([r["dist"] for r in results])
    mean_tilt = np.mean([r["tilt"] for r in results])
    mean_hover = np.mean([r["hover_time"] for r in results])
    mean_flight_time = np.mean([r["flight_time"] for r in results])
    mean_fuel_left = np.mean([r["fuel_left"] for r in results])
    mean_fuel_used = np.mean([r["fuel_used"] for r in results])

    print("\n" + "=" * 75)
    print("  FINAL KEPLER-K2 BENCHMARK SCORECARD")
    print("=" * 75)
    print(f"  Touchdown Success Rate: {success_rate:.1f}% ({successes}/{num_flights})")
    print(f"  Mean Touchdown Speed:   {mean_vz:+.2f} m/s (Target: -1.0 to -2.0 m/s)")
    print(f"  Mean Horizontal Drift:  {mean_vh:.2f} m/s (Target: < 1.6 m/s)")
    print(f"  Mean Radial Dispersion: {mean_dist:.2f} m (Target Pad: < 15.0 m)")
    print(f"  Mean Upright Tilt:      {mean_tilt:.2f} deg (Target: < 5.0 deg)")
    print(f"  Mean Hover Duration:    {mean_hover:.2f} s (Target: < 0.5 s)")
    print(f"  Mean Total Flight Time: {mean_flight_time:.1f} s")
    print(f"  Mean Fuel Conserved:    {mean_fuel_left:.1f} kg (Burned {mean_fuel_used:.1f} kg / 8000 kg)")
    print("=" * 75)

    return results

if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_kepler_k2(model_arg)
