# live_charuco_calibrate.py
import cv2
import numpy as np
import time
import os

# ========= EDIT THESE TO MATCH YOUR BOARD & CAMERA =========
CAMERA_ID = 0                 # your webcam index or RTSP/URL
TARGET_VIEWS = 20             # how many diverse views to collect before calibrating
AUTO_SNAPSHOT = True          # auto-capture when the board moved enough
SAVE_DIR = "calib_live"       # where snapshots & outputs go

# ChArUco board spec:
SQUARES_X = 6                 # number of squares across (columns)
SQUARES_Y = 9                 # number of squares down (rows)
SQUARE_LEN_MM = 27.0          # mm edge length of a FULL square
MARKER_LEN_MM = 20.0          # mm marker size inside each square
DICT = cv2.aruco.DICT_4X4_50  # dictionary used when printing the board
# ==========================================================

os.makedirs(SAVE_DIR, exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN_MM, MARKER_LEN_MM, dictionary)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check CAMERA_ID or device.")

all_corners, all_ids = [], []
img_size = None
last_save_corners = None
frames_saved = 0
status = "Move board/camera to new angles... (SPACE = save, C = calibrate, Q = quit)"

def corners_motion_metric(corners_a, corners_b):
    """Return mean pixel motion between two sets of ChArUco corners."""
    if corners_a is None or corners_b is None: 
        return 1e9
    n = min(len(corners_a), len(corners_b))
    if n == 0:
        return 1e9
    a = corners_a[:n, 0, :]  # (n, 2)
    b = corners_b[:n, 0, :]
    return float(np.linalg.norm(a - b, axis=1).mean())

def try_save_view(frame, ch_corners, ch_ids):
    global frames_saved, img_size, last_save_corners
    if ch_corners is None or ch_ids is None or len(ch_ids) < 10:
        return False, "Need more detected corners to save"

    # Enforce diversity: require enough movement since last saved
    moved = corners_motion_metric(ch_corners, last_save_corners)
    if last_save_corners is not None and moved < 8.0:  # pixels threshold; raise if you get too many similar shots
        return False, f"Too similar to last view (Δ≈{moved:.1f}px). Move more."

    # Save image and remember points
    ts = int(time.time() * 1000)
    img_path = os.path.join(SAVE_DIR, f"img_{ts}.png")
    cv2.imwrite(img_path, frame)

    all_corners.append(ch_corners)
    all_ids.append(ch_ids)
    img_size = frame.shape[1], frame.shape[0]  # (w, h)
    last_save_corners = ch_corners.copy()
    frames_saved += 1
    return True, f"Saved {img_path}  ({frames_saved}/{TARGET_VIEWS})"

def run_calibration():
    if len(all_corners) < 8:
        return False, "Need at least ~8 good views (10–25 recommended)."

    # Calibrate
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_size, None, None
    )
    np.save(os.path.join(SAVE_DIR, "K.npy"), K)
    np.save(os.path.join(SAVE_DIR, "dist.npy"), dist)

    # Quick undistort preview on last frame
    preview_path = os.path.join(SAVE_DIR, "undistorted_preview.png")
    ok, frame = cap.read()
    if ok:
        und = cv2.undistort(frame, K, dist)
        cv2.imwrite(preview_path, und)

    msg = (f"Calibration done. RMS reprojection error: {ret:.3f}\n"
           f"Saved: {SAVE_DIR}/K.npy, {SAVE_DIR}/dist.npy\n"
           f"Preview: {preview_path}")
    return True, msg

print(status)
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # Refine to ChArUco corners
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if retval and ch_corners is not None and ch_ids is not None and len(ch_ids) > 10:
            cv2.aruco.drawDetectedCornersCharuco(frame, ch_corners, ch_ids, (0,255,0))

            if AUTO_SNAPSHOT and frames_saved < TARGET_VIEWS:
                saved, msg = try_save_view(frame, ch_corners, ch_ids)
                if saved:
                    status = msg
                else:
                    # show motion hint on screen
                    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,255), 2)

    # HUD
    cv2.putText(frame, status, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Views: {frames_saved}/{TARGET_VIEWS}   [SPACE=save, C=calibrate, Q=quit]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.imshow("Live ChArUco Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # manual snapshot
        if ids is not None:
            retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if retval:
                saved, msg = try_save_view(frame, ch_corners, ch_ids)
                status = msg
            else:
                status = "Board not stable/visible. Try again."
        else:
            status = "No markers detected. Try again."
    elif key == ord('c'):
        ok_cal, msg = run_calibration()
        status = msg
        print(msg)
        if ok_cal:
            # Keep window open so you can see preview text; press Q to exit when done
            pass

cap.release()
cv2.destroyAllWindows()
