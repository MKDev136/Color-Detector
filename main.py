import cv2
from PIL import Image
import numpy as np

COLORS = {
    "red": [
        ([0, 120, 70], [10, 255, 255]),
        ([170, 120, 70], [179, 255, 255])
    ],
    "green": [
        ([36, 100, 100], [85, 255, 255])
    ],
    "blue": [
        ([90, 100, 100], [130, 255, 255])
    ]
}


def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 340)

cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

cv2.createTrackbar("Red", "TrackBars", 0, 1, empty)
cv2.createTrackbar("Green", "TrackBars", 0, 1, empty)
cv2.createTrackbar("Blue", "TrackBars", 0, 1, empty)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgHSV= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    red = cv2.getTrackbarPos("Red", "TrackBars")
    green = cv2.getTrackbarPos("Green", "TrackBars")
    blue = cv2.getTrackbarPos("Blue", "TrackBars")

    color_mask = np.zeros(imgHSV.shape[:2], dtype=np.uint8)

    lower_manual = np.array([h_min, s_min, v_min])
    upper_manual = np.array([h_max, s_max, v_max])
    manual_mask = cv2.inRange(imgHSV, lower_manual, upper_manual)

    if red:
        for lo, hi in COLORS["red"]:
            color_mask |= cv2.inRange(imgHSV, np.array(lo), np.array(hi))

    if green:
        for lo, hi in COLORS["green"]:
            color_mask |= cv2.inRange(imgHSV, np.array(lo), np.array(hi))

    if blue:
        for lo, hi in COLORS["blue"]:
            color_mask |= cv2.inRange(imgHSV, np.array(lo), np.array(hi))

    mask = manual_mask | color_mask
    mask_ = Image.fromarray(color_mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break