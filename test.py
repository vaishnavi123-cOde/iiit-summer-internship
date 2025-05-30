from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Run prediction on an online image
model("https://ultralytics.com/images/bus.jpg", save=True)
