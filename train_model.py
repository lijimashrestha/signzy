from ultralytics import YOLO

# 1. Initialize the model (This is the missing step!)
model = YOLO('yolov8n.pt') 

# 2. Now you can run your training
results = model.train(
    data='/content/signzy/data.yaml',
    epochs=50,
    imgsz=640,
    device=0
)