import cv2
from ultralytics import YOLO
import pyttsx3

print("--- Initializing Signzy ---")

# 1. Load Model
try:
    model = YOLO('models/trained/best.pt')
    print("✓ Model loaded successfully.")
except Exception as e:
    print(f"✗ Model Error: {e}")
    exit()

# 2. Initialize Camera
cap = cv2.VideoCapture(0) # Try 0 first, then 1
if not cap.isOpened():
    print("✗ Camera Error: Could not open webcam. Check privacy settings.")
    exit()
else:
    print("✓ Camera accessed successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✗ Failed to grab frame.")
        break

    # 3. Detection [cite: 4, 185, 202]
    results = model(frame, verbose=False)
    
    for r in results:
        for box in r.boxes:
            # Draw on screen
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Signzy Debug Window', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- System Stopped ---")