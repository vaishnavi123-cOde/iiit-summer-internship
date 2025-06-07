from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO("yolov8n-seg.pt")

input_folder = Path("C:/Users/nagav/Desktop/images")
output_folder = Path("C:/Users/nagav/Desktop/segmented")
output_folder.mkdir(exist_ok=True)

for image_file in input_folder.glob("*"):
    if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    print(f"Segmenting: {image_file.name}")

    results = model(image_file)

    # Get the first result's plotted image (numpy array in BGR format)
    img_with_masks = results[0].plot()

    # Save the image using OpenCV
    save_path = output_folder / image_file.name
    cv2.imwrite(str(save_path), img_with_masks)

print("✅ Done! Check the 'segmented' folder.")
