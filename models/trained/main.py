import cv2
import pyttsx3
import time
from ultralytics import YOLO

engine = pyttsx3.init()
engine.setProperty('rate', 150)

model_path = r'C:\Users\User\OneDrive\Desktop\signzy\models\trained\best.pt'

try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

current_text     = ""
last_label       = None
confirm_frames   = 0
committed_label  = None
STABILITY        = 15
LETTER_DELAY     = 2.5
last_action_time = time.time() - LETTER_DELAY

print("Waking up camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)
print("Camera ready. Q to quit.")

def filled_rect(frame, x1, y1, x2, y2, color, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    results = model(frame, conf=0.70, verbose=False)
    detected_label = None
    if results[0].boxes:
        detected_label = model.names[int(results[0].boxes[0].cls[0])]
        frame = results[0].plot()
    if detected_label == last_label:
        confirm_frames += 1
    else:
        confirm_frames = 0
        last_label = detected_label
        committed_label = None
    current_time = time.time()
    cooldown_elapsed = (current_time - last_action_time) >= LETTER_DELAY
    ready_to_act = (confirm_frames >= STABILITY and cooldown_elapsed and detected_label is not None and detected_label != committed_label)
    if ready_to_act:
        label = detected_label
        committed_label = label
        if label == 'up':
            if current_text.strip():
                print(f"Speaking: {current_text.strip()}")
                engine.say(current_text.strip())
                engine.runAndWait()
            current_text = ""
            last_action_time = current_time
        elif label == 'down':
            if current_text:
                current_text = current_text[:-1]
                print(f"Erased -> {current_text}")
            last_action_time = current_time
        else:
            current_text += str(label)
            print(f"Added '{label}' -> {current_text}")
            last_action_time = current_time
    filled_rect(frame, 0, 0, w, 75, (30, 30, 30))
    cv2.putText(frame, f"TEXT: {current_text if current_text else '_ _ _'}", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 120), 3)
    time_since = current_time - last_action_time
    if time_since < LETTER_DELAY:
        filled_rect(frame, w-185, 10, w-10, 65, (0, 60, 200))
        cv2.putText(frame, f"WAIT {LETTER_DELAY-time_since:.1f}s", (w-175, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    else:
        filled_rect(frame, w-185, 10, w-10, 65, (0, 140, 0))
        cv2.putText(frame, "READY", (w-175, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    if detected_label and confirm_frames < STABILITY:
        cv2.rectangle(frame, (0, 72), (int((confirm_frames/STABILITY)*w), 78), (0, 200, 255), -1)
    filled_rect(frame, 0, h-45, w, h, (30, 30, 30))
    cv2.putText(frame, "THUMBS UP: Speak & Clear  |  THUMBS DOWN: Backspace  |  Q: Quit", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    cv2.imshow("Signzy", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
