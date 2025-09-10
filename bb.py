import cv2
import numpy as np

def test_basic():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_basic()