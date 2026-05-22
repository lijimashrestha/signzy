from ultralytics import YOLO

# Load your trained model
model = YOLO('models/best.pt')

# Run validation on your test dataset
# Make sure your data.yaml file is in the same folder!
metrics = model.val(data='data.yaml', plots=True)

print(f"Mean Average Precision (mAP50): {metrics.box.map50}")