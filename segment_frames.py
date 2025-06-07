from ultralytics import YOLO
import os

# Load pretrained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')  # Use 'yolov8n-seg.pt' or any other segmentation model

input_folder = 'frames'
output_folder = 'segmented_frames'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all frame images
frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

for frame in frame_files:
    frame_path = os.path.join(input_folder, frame)
    
    # Perform segmentation
    results = model(frame_path)
    
    # Save segmented output image
    # Note: save() saves to default folder, so specify filename explicitly here
    output_path = os.path.join(output_folder, frame)
    results[0].save(output_path)
    
    print(f"Segmented {frame}")

print("Segmentation complete for all frames.")
