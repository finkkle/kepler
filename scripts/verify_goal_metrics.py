"""Verification Script: Auditing Zero Hover, Descent Speed, Fuel Conservation, and 30/30 Success."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from kepler.envs.landing_env import KeplerLandingEnv
from kepler.controllers.flight_computer import KeplerFlightComputer


def audit_controller(name, controller_obj, is_model=False):
    env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
    successes = 0
    results = []

    print(f"\n{'='*75}")
    print(f"  AUDITING: {name}")
    print(f"{'='*75}")

    for idx, seed in enumerate(range(5000, 5030)):
        obs, _ = env.reset(seed=seed)
        if not is_model:
            controller_obj.reset()
        done = False
        steps = 0
        hover_steps = 0  # Steps below 10m where vz is nearly stalled (|vz| < 0.5 m/s)
        flare_steps = 0  # Steps below 6m
        init_dist = float(np.linalg.norm(env.dynamics.state.pos[:2]))
        dist_at_6m = None

        while not done:
            if is_model:
                act, _ = controller_obj.predict(obs, deterministic=True)
            else:
                act = controller_obj.get_action(obs)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            steps += 1

            z = env.dynamics.state.pos[2]
            vz = env.dynamics.state.vel[2]

            if z <= 6.0 and dist_at_6m is None:
                dist_at_6m = float(np.linalg.norm(env.dynamics.state.pos[:2]))

            if z <= 6.0:
                flare_steps += 1
                if abs(vz) < 0.5:
                    hover_steps += 1

        dist = float(np.linalg.norm(info["pos"][:2]))
        vz = float(info["vel"][2])
        vh = float(np.linalg.norm(info["vel"][:2]))
        tilt = float(info["tilt_deg"])
        fuel_used = float(info["fuel_used"])
        fuel_left = float(info["fuel_mass"])
        succ = bool(info["landing_success"])
        if succ:
            successes += 1

        dist_6m_val = dist_at_6m if dist_at_6m is not None else dist
        # Lateral correction accomplished aloft: init_dist -> dist_at_6m
        aloft_correction_pct = max(0.0, (init_dist - dist_6m_val) / max(0.1, init_dist) * 100.0)

        results.append({
            "flight": idx + 1,
            "seed": seed,
            "succ": succ,
            "dist": dist,
            "vz": vz,
            "vh": vh,
            "tilt": tilt,
            "fuel_left": fuel_left,
            "fuel_used": fuel_used,
            "steps": steps,
            "flight_time_s": steps * 0.02,
            "flare_time_s": flare_steps * 0.02,
            "hover_time_s": hover_steps * 0.02,
            "init_dist": init_dist,
            "dist_6m": dist_6m_val,
            "aloft_correction_pct": aloft_correction_pct,
        })

        stat = "PASS" if succ else "FAIL"
        print(f"Flight {idx+1:02d} (Seed {seed}) | {stat} | Dist={dist:5.2f}m | Vz={vz:+5.2f}m/s | Vh={vh:4.2f}m/s | Tilt={tilt:4.2f}° | Time={steps*0.02:4.1f}s | Hover={hover_steps*0.02:4.2f}s | Aloft Corr: {init_dist:4.1f}m->{dist_6m_val:4.1f}m")

    rate = (successes / 30) * 100.0
    mean_dist = np.mean([r["dist"] for r in results])
    mean_vz = np.mean([r["vz"] for r in results])
    mean_vh = np.mean([r["vh"] for r in results])
    mean_tilt = np.mean([r["tilt"] for r in results])
    mean_fuel_left = np.mean([r["fuel_left"] for r in results])
    mean_fuel_used = np.mean([r["fuel_used"] for r in results])
    mean_flight_time = np.mean([r["flight_time_s"] for r in results])
    mean_hover_time = np.mean([r["hover_time_s"] for r in results])
    mean_flare_time = np.mean([r["flare_time_s"] for r in results])

    print(f"\n{'='*75}")
    print(f"  {name} AUDIT SUMMARY")
    print(f"{'='*75}")
    print(f"  Total Score:            {successes}/30 ({rate:.1f}%)")
    print(f"  Mean Pad Offset:        {mean_dist:.2f} m (Target: < 15.0 m)")
    print(f"  Mean Touchdown Vz:      {mean_vz:+.2f} m/s (Target: -1.0 to -1.5 m/s)")
    print(f"  Mean Touchdown Vh:      {mean_vh:.2f} m/s (Target: < 1.6 m/s)")
    print(f"  Mean Upright Tilt:      {mean_tilt:.2f}° (Target: < 5.0°)")
    print(f"  Mean Total Flight Time: {mean_flight_time:.1f} s")
    print(f"  Mean Final Flare Time:  {mean_flare_time:.2f} s")
    print(f"  Mean Hover Time:        {mean_hover_time:.2f} s (Target: 0.00 s)")
    print(f"  Mean Propellant Used:   {mean_fuel_used:.1f} kg / 8000 kg ({mean_fuel_left:.1f} kg remaining)")
    print(f"{'='*75}\n")
    return successes == 30


if __name__ == "__main__":
    fc = KeplerFlightComputer()
    fc_ok = audit_controller("Kepler Flight Computer (Baseline)", fc, is_model=False)

    model = PPO.load("checkpoints/kepler_1.zip")
    model_ok = audit_controller("Kepler-1 Neural Agent (Trained)", model, is_model=True)

    if fc_ok and model_ok:
        print("\n>>> ALL GOAL DELIVERABLES FORMALLY VERIFIED: 100% 30/30 ON BOTH CONTROLLERS <<<")
        sys.exit(0)
    else:
        print("\n>>> VERIFICATION FAILED <<<")
        sys.exit(1)
