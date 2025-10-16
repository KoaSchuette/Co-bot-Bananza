import cv2, numpy as np

K = np.load("K.npy")
dist = np.load("dist.npy")
img = cv2.imread("calib_live\img_1760472776483.png")  # pick any calibration image
und = cv2.undistort(img, K, dist)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", und)
cv2.waitKey(0)
