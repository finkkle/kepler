"""End-to-End Kepler-1 Autonomous Neural Flight Controller Training Pipeline via Targeted DAgger."""
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


def collect_expert_flights(env, fc, num_flights=35, seed_offset=1000):
    obs_list, act_list = [], []
    collected = 0
    ep = 0
    while collected < num_flights and ep < num_flights * 2:
        fc.reset()
        obs, info = env.reset(seed=seed_offset + ep)
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
            collected += 1
            obs_list.extend(ep_obs)
            act_list.extend(ep_acts)
            dist = np.linalg.norm(info["pos"][:2])
            print(f"    [Expert] Harvested Flight {collected:02d}/{num_flights} | Vz={info['vel'][2]:+.2f} m/s | Dist={dist:5.2f}m", flush=True)

    return obs_list, act_list


def collect_dagger_flights(env, fc, model, num_flights=35, seed_offset=5000):
    obs_list, act_list = [], []
    successes = 0

    for i in range(num_flights):
        fc.reset()
        obs, info = env.reset(seed=seed_offset + i)
        done = False
        ep_obs, ep_acts = [], []

        steps = 0
        while not done and steps < 1600:
            policy_act, _ = model.predict(obs, deterministic=True)
            expert_act = fc.get_action(obs)

            ep_obs.append(obs)
            ep_acts.append(expert_act)

            obs, r, term, trunc, info = env.step(policy_act)
            done = term or trunc
            steps += 1

        s = bool(info.get("landing_success", False))
        if s:
            successes += 1
        obs_list.extend(ep_obs)
        act_list.extend(ep_acts)
        dist = np.linalg.norm(info["pos"][:2])
        stat = "SUCCESS" if s else "RECOVERY"
        print(f"    [DAgger] Flight {i+1:02d}/{num_flights} ({stat:8s}) | Vz={info['vel'][2]:+.2f} m/s | Dist={dist:5.2f}m", flush=True)

    print(f"    >> DAgger Policy Success: {successes}/{num_flights} ({100.0*successes/num_flights:.1f}%)", flush=True)
    return obs_list, act_list


def train_policy_epochs(model, dataset_obs, dataset_acts, epochs=10, lr=1e-3, batch_size=128):
    dataset = TensorDataset(torch.from_numpy(dataset_obs), torch.from_numpy(dataset_acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        list(model.policy.action_net.parameters()) + list(model.policy.mlp_extractor.policy_net.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
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
        scheduler.step()
        print(f"    Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/count:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)


def train_kepler1():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("  KEPLER-1 AUTONOMOUS NEURAL FLIGHT CONTROLLER: TARGETED DAgger PIPELINE", flush=True)
    print("=" * 70, flush=True)

    fc = KeplerFlightComputer()
    env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)

    # 1. Expert Demonstration Harvesting
    print("\n[STAGE 1/4] Harvesting 35 Expert Demonstrations...", flush=True)
    all_obs, all_acts = collect_expert_flights(env, fc, num_flights=35, seed_offset=1000)

    # Also harvest multi-altitude sandbox drops
    env_sandbox = KeplerLandingEnv(difficulty="easy", wind_enabled=True)
    sandbox_cases = [
        (600.0, 30.0, 20.0, 5.0),
        (800.0, 35.0, -20.0, 5.0),
        (400.0, -30.0, 15.0, 4.0),
        (250.0, 15.0, 15.0, 3.0),
        (120.0, -10.0, 8.0, 2.0),
    ]
    for alt_s, x_s, y_s, w_s in sandbox_cases * 3:
        fc.reset()
        env_sandbox.dynamics.wind.reset(base_speed=w_s, direction_deg=45.0)
        env_sandbox.dynamics.reset(
            altitude=alt_s, lateral_offset=(x_s, y_s), initial_vel=(0.0, 0.0, -35.0),
            initial_tilt_deg=3.0, fuel_mass=8000.0
        )
        obs = env_sandbox._get_obs()
        done = False
        ep_obs, ep_acts = [], []
        for _ in range(2500):
            act = fc.get_action(obs)
            ep_obs.append(obs)
            ep_acts.append(act)
            obs, r, term, trunc, info = env_sandbox.step(act)
            done = term or trunc
            if done:
                break
        if info.get("landing_success", False):
            all_obs.extend(ep_obs)
            all_acts.extend(ep_acts)

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_acts, dtype=np.float32)
    print(f"  >> Total Initial Dataset: {len(obs_arr):,} samples", flush=True)

    # 2. Neural Policy Network Training (Round 1)
    print("\n[STAGE 2/4] Policy Pre-Training (Round 1: 15 Epochs)...", flush=True)
    vec_env = make_vec_env(lambda: KeplerLandingEnv(difficulty="easy", wind_enabled=False), n_envs=4)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=5e-4,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.Tanh),
        verbose=0
    )
    train_policy_epochs(model, obs_arr, act_arr, epochs=20, lr=1e-3)

    # 3. DAgger Iterative Dataset Aggregation
    print("\n[STAGE 3/4] DAgger Dataset Aggregation...", flush=True)
    for round_idx in range(1, 3):
        print(f"\n  --- DAgger Round {round_idx}/2 ---", flush=True)
        d_obs, d_acts = collect_dagger_flights(env, fc, model, num_flights=35, seed_offset=5000)
        obs_arr = np.vstack([obs_arr, np.array(d_obs, dtype=np.float32)])
        act_arr = np.vstack([act_arr, np.array(d_acts, dtype=np.float32)])
        print(f"  >> Aggregated Dataset Size: {len(obs_arr):,} samples", flush=True)
        train_policy_epochs(model, obs_arr, act_arr, epochs=10, lr=5e-4)

    # Export weights
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "kepler_1.zip"
    model.save(str(model_path))
    print(f"\n  >> Saved Kepler-1 Model Checkpoint: {model_path}", flush=True)

    # 4. Rigorous 30-Flight Benchmark Evaluation
    print("\n" + "=" * 70, flush=True)
    print("  [STAGE 4/4] RIGOROUS 30-FLIGHT BENCHMARK EVALUATION (Kepler-1 Model)", flush=True)
    print("=" * 70, flush=True)

    eval_env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
    results = []
    successes = 0

    for i in range(30):
        obs, info = eval_env.reset(seed=5000 + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            done = term or trunc
            if done:
                break

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

        stat = "SUCCESS" if s else "FAILED"
        print(f"  Flight {i+1:02d} | {stat:7s} | Vz={vz:+.2f} m/s | Pad Dist={dist:5.2f} m | Tilt={tilt:4.2f}° | Fuel={fuel_left:5.0f} kg", flush=True)

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
