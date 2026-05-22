import cv2

print("Testing camera connection...")
# Try changing 0 to 1 if you have multiple cameras
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("ERROR: Could not connect to the camera.")
else:
    print("Camera connection opened. Attempting to grab a frame...")
    success, frame = cap.read()
    
    if not success or frame is None:
        print("ERROR: Connected to camera, but the signal is empty (None).")
        print("-> Check Windows Privacy Settings or close other camera apps.")
    else:
        print("SUCCESS! Camera is working. Press 'q' to close.")
        while True:
            success, frame = cap.read()
            if not success: break
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()