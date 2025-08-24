import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_blue = np.array([100, 150, 50])   
upper_blue = np.array([140, 255, 255]) 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask (Blue Regions)", mask)
    cv2.imshow("Detected Blue Objects", result)

    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()
