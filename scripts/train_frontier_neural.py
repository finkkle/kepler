"""Train Frontier Scaled Neural Controller via Expert DAgger from UnifiedFrontierFC."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path("C:/Users/PC/Projects/kepler")
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from kepler.envs.landing_env import KeplerLandingEnv
from scripts.test_unified_fcs import UnifiedFrontierFC

SANDBOX_CASES = [
    (600.0, 30.0, 20.0, 5.0),
    (800.0, 40.0, -25.0, 6.0),
    (400.0, -35.0, 15.0, 4.0),
    (250.0, 15.0, 15.0, 3.0),
    (120.0, -10.0, 8.0, 2.0),
    (600.0, -20.0, 15.0, 8.0),
    (600.0, 15.0, -10.0, 10.0),
]

def harvest_expert_trajectories(fc, num_flights=35, seed_start=2000):
    env_easy = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
    env_wind = KeplerLandingEnv(difficulty="easy", wind_enabled=True)
    obs_list, act_list = [], []
    
    for i in range(num_flights):
        env = env_easy if i % 2 == 0 else env_wind
        obs, info = env.reset(seed=seed_start + i)
        fc.reset()
        done = False
        ep_obs, ep_acts = [], []
        while not done:
            act = fc.get_action(obs)
            ep_obs.append(obs)
            ep_acts.append(act)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        if info.get("landing_success", False):
            obs_list.extend(ep_obs)
            act_list.extend(ep_acts)
            
    print(f"  Harvested {len(obs_list):,} state-action transitions from standard drops.", flush=True)

    # Multi-altitude drops
    for alt, x, y, w in SANDBOX_CASES * 4:
        env = KeplerLandingEnv(difficulty="easy", wind_enabled=True)
        fc.reset()
        env.dynamics.wind.reset(base_speed=w, direction_deg=45.0)
        env.dynamics.reset(
            altitude=alt,
            lateral_offset=(x, y),
            initial_vel=(0.0, 0.0, -35.0),
            initial_tilt_deg=2.5,
            fuel_mass=8000.0,
        )
        obs = env._get_obs()
        done = False
        ep_obs, ep_acts = [], []
        steps = 0
        while not done and steps < 2500:
            act = fc.get_action(obs)
            ep_obs.append(obs)
            ep_acts.append(act)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            steps += 1
        if info.get("landing_success", False):
            obs_list.extend(ep_obs)
            act_list.extend(ep_acts)

    print(f"  Total Harvested: {len(obs_list):,} transitions (including sandbox & heavy wind).", flush=True)
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)

def train_epochs(model, dataset_obs, dataset_acts, epochs=15, lr=5e-4, batch_size=256):
    dataset = TensorDataset(torch.from_numpy(dataset_obs), torch.from_numpy(dataset_acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        list(model.policy.action_net.parameters()) + list(model.policy.mlp_extractor.policy_net.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    for epoch in range(epochs):
        total_loss, count = 0.0, 0
        for b_obs, b_act in loader:
            features = model.policy.extract_features(b_obs)
            latent = model.policy.mlp_extractor.forward_actor(features)
            preds = model.policy.action_net(latent)

            # Asymmetric loss on throttle: under-throttling is penalized 2x to guarantee soft touchdown
            diff_th = preds[:, 0] - b_act[:, 0]
            loss_th = torch.where(diff_th < 0.0, 2.5 * diff_th ** 2, diff_th ** 2)

            diff_tp = preds[:, 1] - b_act[:, 1]
            diff_tr = preds[:, 2] - b_act[:, 2]
            diff_rcs = preds[:, 3] - b_act[:, 3]

            loss = torch.mean(4.0 * loss_th + 1.8 * diff_tp ** 2 + 1.8 * diff_tr ** 2 + 0.5 * diff_rcs ** 2)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(b_obs)
            count += len(b_obs)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    [Epoch {epoch+1:02d}/{epochs:02d}] Loss: {total_loss/count:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

def quick_eval(model, num_seeds=30):
    env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
    succs = 0
    dists = []
    vzs = []
    hovers = []
    for seed in range(5000, 5000 + num_seeds):
        obs, info = env.reset(seed=seed)
        done = False
        hover_steps = 0
        steps = 0
        while not done and steps < 2000:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            steps += 1
            z = env.dynamics.state.pos[2]
            vz = env.dynamics.state.vel[2]
            if z <= 6.0 and abs(vz) < 0.5:
                hover_steps += 1
        s = bool(info.get("landing_success", False))
        if s: succs += 1
        dists.append(float(np.linalg.norm(info["pos"][:2])))
        vzs.append(float(info["vel"][2]))
        hovers.append(hover_steps * 0.02)
    return succs, np.mean(dists), np.mean(vzs), np.mean(hovers)

if __name__ == "__main__":
    fc = UnifiedFrontierFC()
    k2_path = PROJECT_ROOT / "checkpoints" / "kepler_k2.zip"
    vec_env = make_vec_env(lambda: KeplerLandingEnv(difficulty="easy", wind_enabled=False), n_envs=4)
    model = PPO.load(str(k2_path), env=vec_env)

    print("=" * 75)
    print("  TRAINING FRONTIER NEURAL POLICY (DAGGER CLONING OF FRONTIER FCS)")
    print("=" * 75)

    obs_arr, act_arr = harvest_expert_trajectories(fc, num_flights=30)
    print("Starting Pre-Training Epochs...")
    train_epochs(model, obs_arr, act_arr, epochs=15, lr=6e-4)

    s, d, v, h = quick_eval(model)
    print(f">> Initial Evaluation: {s}/30 PASS | Mean Dist: {d:.2f}m | Mean Vz: {v:+.2f}m/s | Mean Hover: {h:.2f}s")

    # DAgger refinement rounds
    for r_idx in range(1, 4):
        print(f"\n--- DAgger Aggregation Round {r_idx}/3 ---", flush=True)
        env = KeplerLandingEnv(difficulty="easy", wind_enabled=False)
        d_obs, d_acts = [], []
        for i in range(25):
            seed = 6000 + r_idx * 100 + i
            obs, info = env.reset(seed=seed)
            fc.reset()
            done = False
            steps = 0
            while not done and steps < 2000:
                p_act, _ = model.predict(obs, deterministic=True)
                e_act = fc.get_action(obs)
                d_obs.append(obs)
                d_acts.append(e_act)
                # Step with mixture
                use_expert = (np.random.rand() < 0.4)
                step_act = e_act if use_expert else p_act
                obs, rew, term, trunc, info = env.step(step_act)
                done = term or trunc
                steps += 1
        
        obs_arr = np.vstack([obs_arr, np.array(d_obs, dtype=np.float32)])
        act_arr = np.vstack([act_arr, np.array(d_acts, dtype=np.float32)])
        print(f"  Dataset size now: {len(obs_arr):,} transitions.")
        train_epochs(model, obs_arr, act_arr, epochs=10, lr=3e-4)

        s, d, v, h = quick_eval(model)
        print(f">> Round {r_idx} Eval: {s}/30 PASS | Mean Dist: {d:.2f}m | Mean Vz: {v:+.2f}m/s | Mean Hover: {h:.2f}s")
        if s == 30:
            break

    # Save to kepler_k3_neural.zip and kepler_k3_sq.zip
    out_path = PROJECT_ROOT / "checkpoints" / "kepler_k3_sq.zip"
    model.save(str(out_path))
    print(f"\nSaved Frontier Neural Policy: {out_path}")
