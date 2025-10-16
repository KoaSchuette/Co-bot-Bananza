# vision_to_robot.py
import numpy as np
import cv2
from math import atan2, pi

# ---------- Load calibration ----------
K    = np.load("K.npy")
dist = np.load("dist.npy")
H    = np.load("H.npy")

# ---------- Taught robot waypoints (fill yours) ----------
# Top-Left (origin) of board in BASE:
TL = dict(x = 138.8, y = 695.4, z = -111.80, rx=2.489, ry=1.918, rz=-0.002)
# Top-Right (+X) of board in BASE:
TR = dict(x = 165.06, y = 800.23, z = -111.80, rx=2.489, ry=1.918, rz=-0.002)

TABLE_Z_B = TL["z"]  # table surface Z in base (mm)

# ---------- Build table->base transform (2D) ----------
o  = np.array([TL["x"], TL["y"]], float)
xv = np.array([TR["x"], TR["y"]], float) - o
ex = xv / np.linalg.norm(xv)                 # +X_table unit in base
ey = np.array([-ex[1], ex[0]])               # +Y_table unit in base
yaw = atan2(ex[1], ex[0])                    # table yaw in base (rad)

def _undistort_pixel(u: float, v: float):
    """Return undistorted pixel coords consistent with K."""
    pts = np.array([[[u, v]]], dtype=np.float32)
    # Reproject with K so the output is in pixel units again
    uv = cv2.undistortPoints(pts, K, dist, None, K)  # shape (1,1,2)
    return float(uv[0,0,0]), float(uv[0,0,1])

def _pixel_to_table_mm(u: float, v: float):
    """Apply homography (expects UNDISTORTED pixels) -> (X,Y) mm on table."""
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    q /= q[2]
    return float(q[0]), float(q[1])

def _table_to_base_mm(X_mm: float, Y_mm: float, z_offset_mm: float):
    """Map table (X,Y) to base (X,Y,Z)."""
    xy_b = o + X_mm*ex + Y_mm*ey
    z_b  = TABLE_Z_B + z_offset_mm
    return float(xy_b[0]), float(xy_b[1]), float(z_b)

def pixel_to_robot_pose(u: float, v: float,
                        z_offset_mm: float = 15.0,
                        rx: float = pi, ry: float = 0.0, rz: float | None = None):
    """
    Convert a pixel (u,v) to robot base pose (X,Y,Z,RX,RY,RZ).
    - u,v: pixel coords from your frame (distorted OK; we undistort internally)
    - z_offset_mm: approach height above table surface
    - rx,ry,rz: tool orientation (defaults to Z-down with yaw aligned to board +X)
    """
    uu, vv = _undistort_pixel(u, v)           # pixel -> undistorted pixel
    X_mm, Y_mm = _pixel_to_table_mm(uu, vv)   # undistorted pixel -> table mm
    BX, BY, BZ = _table_to_base_mm(X_mm, Y_mm, z_offset_mm)  # table -> base
    if rz is None:
        rz = yaw
    return BX, BY, BZ, rx, ry, rz

# ---------- Optional CLI: type "u v" to get pose ----------
if __name__ == "__main__":
    BX, BY, BZ, RX, RY, RZ = pixel_to_robot_pose(386,627)
    print(f"BASE pose -> X:{BX:.2f}  Y:{BY:.2f}  Z:{BZ:.2f}  RX:{RX:.3f}  RY:{RY:.3f}  RZ:{RZ:.3f}")
