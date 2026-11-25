import numpy as np
from kepler.envs.landing_env import KeplerLandingEnv
from kepler.controllers.flight_computer import KeplerFlightComputer
from stable_baselines3 import PPO

def test_controller(name, controller_fn, is_model=False):
    env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
    successes = 0
    print(f"\n==================== Testing {name} ====================")
    for idx, seed in enumerate(range(5000, 5030)):
        obs, _ = env.reset(seed=seed)
        if not is_model:
            controller_fn.reset()
        done = False
        steps = 0
        while not done:
            if is_model:
                act, _ = controller_fn.predict(obs, deterministic=True)
            else:
                act = controller_fn.get_action(obs)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            steps += 1
        dist = float(np.linalg.norm(info["pos"][:2]))
        vh = float(np.linalg.norm(info["vel"][:2]))
        vz = float(info["vel"][2])
        tilt = float(info["tilt_deg"])
        fuel = float(info["fuel_used"])
        succ = info["landing_success"]
        if succ:
            successes += 1
        status = "PASS" if succ else "FAIL"
        print(f"Flight {idx+1:02d} (Seed {seed}) | {status} | Dist: {dist:5.2f}m | Vz: {vz:+5.2f}m/s | Vh: {vh:4.2f}m/s | Tilt: {tilt:4.2f}° | Steps: {steps} | Fuel: {fuel:5.0f}kg")
    print(f">> Total {name}: {successes}/30 ({successes/30*100:.1f}%)")

if __name__ == "__main__":
    fc = KeplerFlightComputer()
    test_controller("Kepler Flight Computer", fc, is_model=False)
