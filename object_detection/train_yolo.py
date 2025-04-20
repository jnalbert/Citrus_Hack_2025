from ultralytics import YOLO
import os
data=os.path.join(os.path.dirname(__file__), 'mnt/data/synthetic_dataset/data.yaml'),

# Load a pre‑trained backbone
model = YOLO('yolov8n.pt')

# Train on your data.yaml
results = model.train(
    data=data,
    epochs=30,
    imgsz=640,
    batch=8,
)