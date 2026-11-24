"""End-to-End Kepler-1 Autonomous Neural Flight Controller Training Pipeline."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from kepler.envs.landing_env import KeplerLandingEnv
from kepler.controllers.flight_computer import KeplerFlightComputer


def train_kepler1():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("  KEPLER-1 AUTONOMOUS NEURAL FLIGHT CONTROLLER: TRAINING PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Expert Flight Demonstration Harvesting
    print("\n[STAGE 1/4] Harvesting 25 Expert Landings via Autonomous FCS...", flush=True)
    env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
    fc = KeplerFlightComputer()

    all_obs = []
    all_acts = []
    collected_flights = 0
    ep = 0

    while collected_flights < 25:
        fc.reset()
        obs, info = env.reset(seed=2000 + ep)
        ep += 1
        done = False
        ep_obs, ep_acts = [], []

        while not done:
            act = fc.get_action(obs)
            ep_obs.append(obs)
            ep_acts.append(act)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc

        if info.get("landing_success", False):
            collected_flights += 1
            all_obs.extend(ep_obs)
            all_acts.extend(ep_acts)
            dist = np.linalg.norm(info["pos"][:2])
            print(f"  Harvested Flight {collected_flights:02d}/25 | Vz={info['vel'][2]:+.2f} m/s | Pad Dist={dist:5.2f} m | Fuel={info['fuel_mass']:.0f} kg", flush=True)

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_acts, dtype=np.float32)
    print(f"  >> Total Collected Trajectory Samples: {len(obs_arr):,} state-action pairs", flush=True)

    # 2. Balanced Active-Thrust / Coasting Conditioning
    print("\n[STAGE 2/4] Conditioning Balanced Thrust Dynamics...", flush=True)
    is_thrusting = act_arr[:, 0] > 0.0
    thrust_obs = obs_arr[is_thrusting]
    thrust_act = act_arr[is_thrusting]
    coast_obs = obs_arr[~is_thrusting]
    coast_act = act_arr[~is_thrusting]

    print(f"  >> Active Thrusting Samples: {len(thrust_obs):,}", flush=True)
    print(f"  >> Aerodynamic Coast Samples: {len(coast_obs):,}", flush=True)

    # Balance 50% thrusting / 50% coasting to eliminate under-throttling bias
    n_sample = min(len(thrust_obs), len(coast_obs))
    balanced_obs = np.vstack([thrust_obs, coast_obs[:n_sample]])
    balanced_act = np.vstack([thrust_act, coast_act[:n_sample]])
    print(f"  >> Balanced Training Set: {len(balanced_obs):,} samples (50% burn / 50% coast)", flush=True)

    # 3. Neural Policy Network Pre-Training (Behavioral Cloning)
    print("\n[STAGE 3/4] Pre-Training Deep Kepler-1 Actor-Critic Policy (256x256 Tanh)...", flush=True)
    vec_env = make_vec_env(lambda: KeplerLandingEnv(difficulty="easy", wind_enabled=False), n_envs=4)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=5e-4,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.Tanh),
        verbose=0
    )

    dataset = TensorDataset(torch.from_numpy(balanced_obs), torch.from_numpy(balanced_act))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(
        list(model.policy.action_net.parameters()) + list(model.policy.mlp_extractor.policy_net.parameters()),
        lr=8e-4
    )
    loss_fn = nn.MSELoss()

    for epoch in range(12):
        total_loss, count = 0.0, 0
        for b_obs, b_act in loader:
            features = model.policy.extract_features(b_obs)
            latent = model.policy.mlp_extractor.forward_actor(features)
            preds = model.policy.action_net(latent)
            loss = loss_fn(preds, b_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        print(f"  Epoch {epoch+1:02d}/12 | Policy MSE Loss: {total_loss/count:.5f}", flush=True)

    # Export weights
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "kepler_1.zip"
    model.save(str(model_path))
    model.save(str(ckpt_dir / "kepler_1_bc.zip"))
    print(f"  >> Exported Checkpoint: {model_path}", flush=True)

    # 4. Rigorous 30-Flight Benchmark Evaluation
    print("\n" + "=" * 70, flush=True)
    print("  [STAGE 4/4] RIGOROUS 30-FLIGHT BENCHMARK EVALUATION", flush=True)
    print("=" * 70, flush=True)

    eval_env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
    results = []
    successes = 0

    for i in range(30):
        obs, info = eval_env.reset(seed=9000 + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            done = term or trunc

        s = bool(info.get("landing_success", False))
        if s:
            successes += 1
        vz = info["vel"][2]
        vh = np.linalg.norm(info["vel"][:2])
        dist = np.linalg.norm(info["pos"][:2])
        tilt = info["tilt_deg"]
        fuel_left = info["fuel_mass"]
        fuel_used = info["fuel_used"]

        results.append({
            "flight": i + 1, "success": s, "vz": vz, "vh": vh,
            "dist": dist, "tilt": tilt, "fuel_left": fuel_left, "fuel_used": fuel_used
        })

        stat = "SUCCESS" if s else "CRASH/MISS"
        print(f"  Flight {i+1:02d} | {stat:10s} | Vz={vz:+.2f} m/s | Pad Dist={dist:5.2f} m | Tilt={tilt:4.2f}° | Fuel={fuel_left:5.0f} kg", flush=True)

    rate = 100.0 * successes / 30
    mean_vz = np.mean([r["vz"] for r in results])
    mean_dist = np.mean([r["dist"] for r in results])
    mean_tilt = np.mean([r["tilt"] for r in results])
    mean_fuel = np.mean([r["fuel_left"] for r in results])

    print("\n" + "=" * 70, flush=True)
    print("  KEPLER-1 BENCHMARK SCORECARD", flush=True)
    print("=" * 70, flush=True)
    print(f"  Touchdown Success Rate: {rate:.1f}% ({successes}/30)")
    print(f"  Mean Touchdown Speed:   {mean_vz:+.2f} m/s (Target: < 2.0 m/s)")
    print(f"  Mean Pad Dispersion:    {mean_dist:.2f} m (Target ASDS: < 15.0 m)")
    print(f"  Mean Upright Tilt:      {mean_tilt:.2f} deg (Target: < 5.0 deg)")
    print(f"  Mean Propellant Left:   {mean_fuel:.1f} kg / 8000 kg reserve")
    print(f"  Total Pipeline Time:    {time.time() - start_time:.1f} seconds")
    print("=" * 70, flush=True)

    return results


if __name__ == "__main__":
    train_kepler1()
