import numpy as np
from math import atan2, pi

# ==== FILL THESE FROM YOUR TAUGHT WAYPOINTS (in mm & radians) ====
# Top-Left (origin) corner of the board in BASE:
TL_x, TL_y, TL_z =  138.8, 695.4, -111.80
TL_rx, TL_ry, TL_rz = 2.489,1.918,-0.002

# Top-Right (+X) corner of the board in BASE:
TR_x, TR_y, TR_z =  165.06, 800.23, -111.80
TR_rx, TR_ry, TR_rz = 2.489,1.918,-0.002

# Physical width of your board along +X (mm)
BOARD_W_MM = 189.0
# Table surface Z in BASE (mm) — you said your taught Z is the table:
TABLE_Z_B = TL_z

# ==== BUILD TABLE → BASE TRANSFORM (in-plane) ====
origin_B = np.array([TL_x, TL_y], dtype=float)
xref_B   = np.array([TR_x, TR_y], dtype=float)

# unit +X_T (board) in BASE
v = xref_B - origin_B
dist_meas = float(np.linalg.norm(v))
if dist_meas < 1e-6:
    raise ValueError("TL and TR are the same point — check your taught waypoints.")
ex = v / dist_meas

# unit +Y_T in BASE (90° CCW from ex)
ey = np.array([-ex[1], ex[0]])

# sanity check on scale
scale_err = abs(dist_meas - BOARD_W_MM)
print(f"[check] TR–TL distance = {dist_meas:.2f} mm (expected {BOARD_W_MM:.2f}); error = {scale_err:.2f} mm")

# yaw of table's +X in BASE (for tool orientation if you want to align gripper with board X)
yaw = atan2(ex[1], ex[0])

# Helper to map (X,Y) in table frame → (Xb,Yb,Zb,yaw) in BASE
def tableXY_to_base(X_mm: float, Y_mm: float, approach_z_mm: float = 15.0):
    xy_b = origin_B + X_mm*ex + Y_mm*ey
    z_b  = TABLE_Z_B + approach_z_mm
    return float(xy_b[0]), float(xy_b[1]), float(z_b), float(yaw)

# Build URScript p[] pose string (UR expects meters)
def ur_pose(x_mm, y_mm, z_mm, rx, ry, rz):
    return f"p[{x_mm/1000:.6f},{y_mm/1000:.6f},{z_mm/1000:.6f},{rx:.6f},{ry:.6f},{rz:.6f}]"

# ==== EXAMPLE: use a point from vision (table X,Y in mm) ====
# Suppose your homography gave you:
X_mm, Y_mm = 80.0, 95.0

BX, BY, BZ, RZ_yaw = tableXY_to_base(X_mm, Y_mm, approach_z_mm=15.0)

# Choose tool orientation:
# Option A: keep tool Z down and align gripper with board +X:
RX, RY, RZ = pi, 0.0, RZ_yaw
# Option B: ignore yaw and use your favorite fixed RZ (replace RZ above)

cmd_approach = f"movel({ur_pose(BX, BY, BZ, RX, RY, RZ)}, a=1.2, v=0.25)"
cmd_pick     = f"movel({ur_pose(BX, BY, TABLE_Z_B + 1.0, RX, RY, RZ)}, a=0.5, v=0.05)"  # down to ~1 mm
cmd_close    = "set_digital_out(0, True)"  # example gripper close
cmd_retract  = f"movel({ur_pose(BX, BY, BZ, RX, RY, RZ)}, a=1.2, v=0.25)"

print(cmd_approach)
print(cmd_pick)
print(cmd_close)
print(cmd_retract)
