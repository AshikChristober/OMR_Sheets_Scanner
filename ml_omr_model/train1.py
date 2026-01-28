from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' = nano model (smallest and fastest)

# Train the model
results = model.train(
    data="/home/ashik-christober/Desktop/ml projects/ml_omr_model/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=10,
    device="cpu",
    name="omr_yolov8_cpu"
)
