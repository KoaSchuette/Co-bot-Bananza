# cam_to_table_xy_live_charuco.py
import cv2
import numpy as np

# ================== CAMERA & BOARD ==================
CAMERA_ID = 0

# Your ChArUco board (from your photo/stats)
SQUARES_X = 6                 # squares across (columns)
SQUARES_Y = 9                 # squares down (rows)
SQUARE_LEN_MM = 27.0          # full square size in mm
MARKER_LEN_MM = 20.0          # marker size in mm
DICT = cv2.aruco.DICT_4X4_50

# Board physical "rectangle" for homography (inner corners span)
RECT_WIDTH_MM  = (SQUARES_X - 1) * SQUARE_LEN_MM   # 5 * 27 = 135 mm
RECT_HEIGHT_MM = (SQUARES_Y - 1) * SQUARE_LEN_MM   # 8 * 27 = 216 mm

# =============== ROBOT / TABLE (optional) ==========
TABLE_ORIGIN_BASE_X_MM = 400.0
TABLE_ORIGIN_BASE_Y_MM = -250.0
PICK_Z_MM = 15.0
RX, RY, RZ = 3.14159, 0.0, 0.0  # tool Z down

# =============== DISPLAY / UI OPTIONS ===============
DISP_HEIGHT = 1080                 # downscale height for preview
MAGNIFIER = True                   # show zoomed inset near cursor
MAG_FACTOR = 2                     # 2x or 3x
MAG_SIZE = 220                     # size of magnifier box in px (display space)
# ====================================================

# Load intrinsics
K = np.load("K.npy"); dist = np.load("dist.npy")

# Camera
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# ArUco/ChArUco objects
aru_dict = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN_MM, MARKER_LEN_MM, aru_dict)
detector = cv2.aruco.ArucoDetector(aru_dict, cv2.aruco.DetectorParameters())

# State
H = None
undistort_on = True
clicked_px = []           # clicks in FULL-RES pixel coords
scale_x = scale_y = 1.0   # display->fullres scale
mouse_xy_disp = (0, 0)

def undistort(img):
    return cv2.undistort(img, K, dist)

def px_to_table(u, v, H):
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    q /= q[2]
    return float(q[0]), float(q[1])

def ur_pose(x_mm, y_mm, z_mm, rx, ry, rz):
    return f"p[{x_mm/1000.0:.6f},{y_mm/1000.0:.6f},{z_mm/1000.0:.6f},{rx:.6f},{ry:.6f},{rz:.6f}]"

def on_mouse(event, x, y, flags, param):
    global mouse_xy_disp, clicked_px
    mouse_xy_disp = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        # map display coords back to original full-res pixel coords
        clicked_px.append((x * scale_x, y * scale_y))

def auto_build_H(view_full):
    """
    Detect ChArUco corners and compute homography using MANY correspondences.
    No clicking needed. Returns H or None.
    """
    gray = cv2.cvtColor(view_full, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 3:
        return None, "No ArUco markers detected"

    ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if not ok or ch_corners is None or ch_ids is None or len(ch_ids) < 8:
        return None, "Not enough ChArUco corners"

    # Build 2D-2D pairs: image pixels <-> board XY in mm
    pts_img = ch_corners.reshape(-1, 2).astype(np.float32)
    # board.getChessboardCorners() holds 3D points (x,y,0) in same units we passed (mm)
    pts_tab = board.getChessboardCorners()[ch_ids.flatten(), :2].astype(np.float32)

    H, _ = cv2.findHomography(pts_img, pts_tab, method=cv2.RANSAC)
    if H is None:
        return None, "Homography failed"
    return H, f"H OK | corners used: {len(pts_tab)}"

cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("cam", on_mouse)

help_text = "A=auto-H from ChArUco | P=pick | U=undistort | S=save H | Q=quit"
status = "Move camera to see board; press A to auto-build H"

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    # Work at full resolution internally
    full = undistort(frame) if undistort_on else frame

    # High-quality downscale for display
    aspect = full.shape[1] / full.shape[0]
    disp_w = int(DISP_HEIGHT * aspect)
    hud = cv2.resize(full, (disp_w, DISP_HEIGHT), interpolation=cv2.INTER_LANCZOS4)  # high quality
    scale_x = full.shape[1] / disp_w
    scale_y = full.shape[0] / DISP_HEIGHT

    # Draw clicked points (project to display)
    for i, (xf, yf) in enumerate(clicked_px[-10:]):
        xd, yd = int(xf / scale_x), int(yf / scale_y)
        cv2.circle(hud, (xd, yd), 6, (0, 255, 0), -1)
        cv2.putText(hud, str(i+1), (xd+6, yd-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Magnifier near mouse (optional)
    if MAGNIFIER:
        mx, my = mouse_xy_disp
        # map mouse to full-res ROI
        mf = MAG_FACTOR
        # size in full-res pixels to crop for magnifier
        crop_w = int(MAG_SIZE / mf)
        crop_h = int(MAG_SIZE / mf)
        cx_full = int(mx * scale_x); cy_full = int(my * scale_y)
        x0 = max(0, cx_full - crop_w // 2); x1 = min(full.shape[1], x0 + crop_w)
        y0 = max(0, cy_full - crop_h // 2); y1 = min(full.shape[0], y0 + crop_h)
        roi = full[y0:y1, x0:x1]
        if roi.size > 0:
            mag = cv2.resize(roi, (MAG_SIZE, MAG_SIZE), interpolation=cv2.INTER_NEAREST)  # crisp pixel zoom
            # place it top-left corner of hud
            y_off = 60
            hud[y_off:y_off+mag.shape[0], 10:10+mag.shape[1]] = mag
            cv2.rectangle(hud, (10, y_off), (10+MAG_SIZE, y_off+MAG_SIZE), (255,255,0), 2)
            cv2.putText(hud, f"{mf}x", (10, y_off-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # HUD texts
    cv2.putText(hud, help_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(hud, f"H: {'READY' if H is not None else 'NOT SET'}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0) if H is not None else (0,0,255), 2)
    if status:
        cv2.putText(hud, status, (10, hud.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("cam", hud)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key in (ord('u'), ord('U')):
        undistort_on = not undistort_on
        status = f"Undistort: {'ON' if undistort_on else 'OFF'}"
    elif key in (ord('a'), ord('A')):
        H_try, msg = auto_build_H(full)
        status = msg
        if H_try is not None:
            H = H_try
            np.save("H.npy", H)
    elif key in (ord('s'), ord('S')):
        if H is not None:
            np.save("H.npy", H); status = "H saved to H.npy"
        else:
            status = "H not set"
    elif key in (ord('p'), ord('P')):
        if H is None:
            status = "Build H first (press A)"
            continue
        # Wait for one click
        clicked_px = []  # clear buffer
        status = "Click the target point"
        # block until click or timeout
        for _ in range(1000):  # ~1s
            ok2, frame2 = cap.read()
            if not ok2: continue
            full2 = undistort(frame2) if undistort_on else frame2
            # draw same HUD while waiting
            hud2 = cv2.resize(full2, (disp_w, DISP_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            cv2.putText(hud2, "Click a point...", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("cam", hud2)
            cv2.waitKey(1)
            if len(clicked_px) > 0:
                break
        if len(clicked_px) > 0:
            u, v = clicked_px[-1]
            X_mm, Y_mm = px_to_table(u, v, H)
            BX = TABLE_ORIGIN_BASE_X_MM + X_mm
            BY = TABLE_ORIGIN_BASE_Y_MM + Y_mm
            BZ = PICK_Z_MM
            cmd = f"movel({ur_pose(BX, BY, BZ, RX, RY, RZ)}, a=1.2, v=0.25)"
            status = f"Pixel({u:.1f},{v:.1f}) -> Table({X_mm:.1f},{Y_mm:.1f}) mm | {cmd}"
            print(status)

cap.release()
cv2.destroyAllWindows()
