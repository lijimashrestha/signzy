from ultralytics import YOLO

# The if __name__ == '__main__' block is highly recommended for Windows 
# so the training doesn't crash your computer.
if __name__ == '__main__':
    
    # 1. Load the pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt') 
    
    print("Starting training...")
    
    # 2. Train the model using the settings from your project report
    results = model.train(
        data='data.yaml',   # Points to your dataset map
        epochs=10,          # The number of times it learns the dataset
        imgsz=416,          # Image size specified in your report
        workers=0           # Keeps Windows stable during training
    )
    
    print("Training complete! The best model is saved in the 'runs/detect/train/weights' folder.")