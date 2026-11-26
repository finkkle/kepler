"""Interactive 3D Simulation server using FastAPI, WebSockets, and Three.js."""
import os
import json
import asyncio
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from kepler.config import BoosterConfig, SimConfig
from kepler.physics.dynamics import RocketDynamics
from kepler.physics.quaternion import quat_to_euler
from kepler.controllers.flight_computer import KeplerFlightComputer

app = FastAPI(title="Kepler 3D Rocket Sandbox")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Simulation state manager
class SimulationSession:
    def __init__(self):
        self.cfg = BoosterConfig()
        self.sim_cfg = SimConfig()
        self.dynamics = RocketDynamics(self.cfg, self.sim_cfg)
        self.flight_computer = KeplerFlightComputer(self.cfg)
        self.rl_agent = None  # Loaded if available
        self.k2_agent = None
        self.k1_agent = None

        self.mode = "autopilot"  # 'autopilot', 'rl_k2', 'rl_k1', 'manual', 'freefall'
        self.running = True
        self.paused = False

        # Parameters
        self.init_alt = 600.0
        self.init_x = 30.0
        self.init_y = 20.0
        self.init_wind = 5.0
        self.init_wind_dir = 45.0
        self.init_fuel = 8000.0

        # Manual control inputs
        self.manual_throttle = 0.0
        self.manual_gimbal_y = 0.0
        self.manual_gimbal_z = 0.0
        self.manual_rcs = 0.0

        self.reset()

    def reset(self):
        self.dynamics.wind.reset(base_speed=self.init_wind, direction_deg=self.init_wind_dir)
        self.flight_computer.reset()
        self.dynamics.reset(
            altitude=self.init_alt,
            lateral_offset=(self.init_x, self.init_y),
            initial_vel=(0.0, 0.0, -35.0),
            initial_tilt_deg=3.0,
            fuel_mass=self.init_fuel,
        )

    def load_rl_agent(self, model_path: str):
        try:
            from stable_baselines3 import PPO
            self.rl_agent = PPO.load(model_path)
            print(f"[SimSession] Successfully loaded Kepler-1 model from {model_path}")
            return True
        except Exception as e:
            print(f"[SimSession] Error loading RL model: {e}")
            return False

    def step(self):
        if self.paused:
            return self.get_telemetry()

        s = self.dynamics.state
        if s.ground_contact:
            return self.get_telemetry()

        wind = self.dynamics.wind.get_wind(s.pos[2], dt=0.0)
        obs = np.array([
            s.pos[0] / 100.0,
            s.pos[1] / 100.0,
            s.pos[2] / 500.0,
            s.vel[0] / 50.0,
            s.vel[1] / 50.0,
            s.vel[2] / 50.0,
            s.quat[0], s.quat[1], s.quat[2], s.quat[3],
            s.omega[0], s.omega[1], s.omega[2],
            s.tilt_rad,
            s.fuel_mass / max(1.0, self.init_fuel),
            wind[0] / 20.0, wind[1] / 20.0
        ], dtype=np.float32)

        active_agent = None
        if self.mode in ["rl", "rl_k2"]:
            active_agent = self.k2_agent or self.rl_agent
        elif self.mode == "rl_k1":
            active_agent = self.k1_agent or self.rl_agent

        if active_agent is not None:
            action, _ = active_agent.predict(obs, deterministic=True)
            a_th, a_tp, a_tr, a_rcs = action
        elif self.mode == "autopilot":
            action = self.flight_computer.get_action(obs)
            a_th, a_tp, a_tr, a_rcs = action
        elif self.mode == "manual":
            action = None
            throttle_cmd = self.manual_throttle
            gy_cmd = self.manual_gimbal_y * self.cfg.max_gimbal_angle
            gz_cmd = self.manual_gimbal_z * self.cfg.max_gimbal_angle
            rcs_cmd = self.manual_rcs
        else:  # freefall
            action = None
            throttle_cmd = 0.0
            gy_cmd = 0.0
            gz_cmd = 0.0
            rcs_cmd = 0.0

        if self.mode in ["rl", "rl_k2", "rl_k1", "autopilot"] and action is not None:
            throttle_cmd = 0.0 if a_th <= 0.0 else (self.cfg.min_throttle + (self.cfg.max_throttle - self.cfg.min_throttle) * float(a_th))

            # Inner-loop TVC
            euler = quat_to_euler(s.quat)
            roll, pitch, yaw = euler
            wx, wy, wz = s.omega
            z_curr = max(0.0, s.pos[2])

            max_tilt = np.radians(16.0)
            near_ground = 1.0 if z_curr > 6.0 else max(0.0, z_curr / 6.0)
            target_pitch = float(a_tp * max_tilt * near_ground)
            target_roll = float(a_tr * max_tilt * near_ground)

            kp_att = 4.2
            kd_att = 2.4
            gy_cmd = np.clip(kp_att * (pitch - target_pitch) + kd_att * wy, -self.cfg.max_gimbal_angle, self.cfg.max_gimbal_angle)
            gz_cmd = np.clip(-(kp_att * (roll - target_roll) + kd_att * wx), -self.cfg.max_gimbal_angle, self.cfg.max_gimbal_angle)
            rcs_cmd = float(a_rcs)

            if s.ground_contact or s.pos[2] <= 0.05:
                throttle_cmd = 0.0

        self.dynamics.step(throttle_cmd, gy_cmd, gz_cmd, rcs_cmd)
        return self.get_telemetry()

    def get_telemetry(self) -> dict:
        s = self.dynamics.state
        euler = quat_to_euler(s.quat)
        wind = self.dynamics.wind.get_wind(s.pos[2], dt=0.0)

        # Touchdown outcome check
        status = "IN_FLIGHT"
        if s.ground_contact or s.pos[2] <= 0.0:
            soft = abs(s.vel[2]) <= self.cfg.max_touchdown_vz
            upright = s.tilt_rad <= self.cfg.max_touchdown_tilt
            on_pad = np.linalg.norm(s.pos[:2]) <= self.cfg.pad_radius
            if soft and upright and on_pad:
                status = "SUCCESSFUL_LANDING"
            else:
                status = "CRASH_IMPACT"

        return {
            "pos": [round(float(v), 2) for v in s.pos],
            "vel": [round(float(v), 2) for v in s.vel],
            "quat": [round(float(v), 4) for v in s.quat],
            "euler_deg": [round(float(np.degrees(v)), 2) for v in euler],
            "tilt_deg": round(float(np.degrees(s.tilt_rad)), 2),
            "fuel_mass": round(float(s.fuel_mass), 1),
            "fuel_pct": round(float(100.0 * s.fuel_mass / max(1.0, self.init_fuel)), 1),
            "throttle_pct": round(float(100.0 * s.throttle), 1),
            "gimbal_deg": [round(float(np.degrees(s.gimbal_y)), 2), round(float(np.degrees(s.gimbal_z)), 2)],
            "wind": [round(float(v), 2) for v in wind],
            "dist_to_pad": round(float(np.linalg.norm(s.pos[:2])), 2),
            "status": status,
            "mode": self.mode,
            "legs_deployed": bool(s.pos[2] < 35.0),
        }

session = SimulationSession()

# Check for trained models: prefer next-gen kepler_k2.zip, with kepler_k1.zip and kepler_1.zip fallbacks
checkpoints_dir = Path(__file__).parent.parent.parent / "checkpoints"
k2_model_path = checkpoints_dir / "kepler_k2.zip"
k1_model_path = checkpoints_dir / "kepler_k1.zip"
legacy_model_path = checkpoints_dir / "kepler_1.zip"

from stable_baselines3 import PPO

if k2_model_path.exists():
    try:
        session.k2_agent = PPO.load(str(k2_model_path))
        session.rl_agent = session.k2_agent
        session.mode = "rl_k2"
        print(f"[SimSession] Successfully loaded Kepler-K2 model from {k2_model_path}")
    except Exception as e:
        print(f"[SimSession] Error loading Kepler-K2: {e}")

if k1_model_path.exists():
    try:
        session.k1_agent = PPO.load(str(k1_model_path))
        if session.rl_agent is None:
            session.rl_agent = session.k1_agent
            session.mode = "rl_k1"
        print(f"[SimSession] Successfully loaded Kepler-K1 model from {k1_model_path}")
    except Exception as e:
        print(f"[SimSession] Error loading Kepler-K1: {e}")
elif legacy_model_path.exists() and session.rl_agent is None:
    session.load_rl_agent(str(legacy_model_path))
    session.mode = "rl"

@app.get("/")
def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>Kepler 3D Sandbox is initializing...</h3>")

@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Check for incoming client commands (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.015)
                cmd = json.loads(msg)
                action = cmd.get("action")
                if action == "reset":
                    params = cmd.get("params", {})
                    session.init_alt = float(params.get("altitude", session.init_alt))
                    session.init_x = float(params.get("offsetX", session.init_x))
                    session.init_y = float(params.get("offsetY", session.init_y))
                    session.init_wind = float(params.get("windSpeed", session.init_wind))
                    session.init_fuel = float(params.get("fuel", session.init_fuel))
                    session.reset()
                elif action == "set_mode":
                    session.mode = cmd.get("mode", session.mode)
                elif action == "toggle_pause":
                    session.paused = not session.paused
                elif action == "manual_input":
                    session.manual_throttle = float(cmd.get("throttle", 0.0))
                    session.manual_gimbal_y = float(cmd.get("gimbal_y", 0.0))
                    session.manual_gimbal_z = float(cmd.get("gimbal_z", 0.0))
            except asyncio.TimeoutError:
                pass

            data = session.step()
            await ws.send_text(json.dumps(data))
            await asyncio.sleep(0.02)  # ~50 Hz update
    except WebSocketDisconnect:
        pass
