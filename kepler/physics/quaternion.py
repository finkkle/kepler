"""High-performance quaternion math routines for 6-DOF aerospace kinematics."""
import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [qw, qx, qy, qz]."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion [qw, -qx, -qy, -qz]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v using unit quaternion q: v' = q * [0, v] * q*."""
    qw, qx, qy, qz = q
    # Rodrigues-like optimized formula
    q_vec = np.array([qx, qy, qz], dtype=np.float64)
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (qw * uv + uuv)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion to 3x3 rotation matrix (Body to World)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (radians) to quaternion [qw, qx, qy, qz]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw] in radians."""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Compute quaternion time-derivative dq/dt = 0.5 * q * [0, omega]."""
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=np.float64)
    return 0.5 * quat_multiply(q, omega_quat)


def tilt_angle_from_vertical(q: np.ndarray) -> float:
    """Compute angle between rocket longitudinal axis (body +Z) and world +Z (vertical)."""
    # Rocket body z-axis in world coordinates is the 3rd column of rotation matrix
    w, x, y, z = q
    body_z_world = np.array([
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y)
    ], dtype=np.float64)
    # Cosine of angle with world vertical [0, 0, 1] is simply body_z_world[2]
    cos_theta = np.clip(body_z_world[2], -1.0, 1.0)
    return float(np.arccos(cos_theta))
