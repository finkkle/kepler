"""Launch the interactive 3D Rocket Landing Sandbox and WebGL UI."""
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("  KEPLER // 3D AUTONOMOUS ROCKET LANDING SIMULATOR")
    print("=" * 60)
    print("  Starting local 60 FPS WebGL Telemetry Server...")
    print("  Open in your browser: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("kepler.sim3d.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()
