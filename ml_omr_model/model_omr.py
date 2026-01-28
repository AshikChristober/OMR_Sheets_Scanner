from ultralytics import YOLO
import cv2
import os
import numpy as np

model_path = "/home/ashik-christober/runs/detect/omr_yolov8_cpu3/weights/best.pt"
test_image_folder = "/home/ashik-christober/Desktop/ml projects/ml_omr_model/dataset/region data/test/images"
output_folder = "/home/ashik-christober/Desktop/ml projects/ml_omr_model/cropped_outputs"
conf_threshold = 0.5

os.makedirs(output_folder, exist_ok=True)

model = YOLO(model_path)
print("Model loaded successfully!")
print("Classes:", model.names)

ANSWER_CLASS_NAME = "answer_region"

ANSWER_CLASS_ID = None
for k, v in model.names.items():
    if v == ANSWER_CLASS_NAME:
        ANSWER_CLASS_ID = k
        break

if ANSWER_CLASS_ID is None:
    raise ValueError("Answer region class not found in model classes")

def write_yolo_txt_for_image(txt_path, detections, img_w, img_h):
    with open(txt_path, "w") as f:
        for cls_id, x1, y1, x2, y2, _ in detections:
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            bw = (x2 - x1)
            bh = (y2 - y1)
            f.write(
                f"{cls_id} "
                f"{xc / img_w:.6f} "
                f"{yc / img_h:.6f} "
                f"{bw / img_w:.6f} "
                f"{bh / img_h:.6f}\n"
            )

for img_name in os.listdir(test_image_folder):

    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(test_image_folder, img_name)
    base_name = os.path.splitext(img_name)[0]
    txt_path = os.path.join(test_image_folder, base_name + ".txt")

    print(f"\nProcessing: {img_name}")

    results = model.predict(
        source=img_path,
        conf=conf_threshold,
        save=False,
        show=False
    )

    image = cv2.imread(img_path)
    if image is None:
        print("Could not load image")
        continue

    h, w = image.shape[:2]

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        open(txt_path, "w").close()
        print("No detections found")
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    answer_indices = [
        i for i, cls in enumerate(class_ids)
        if int(cls) == ANSWER_CLASS_ID
    ]

    if len(answer_indices) == 0:
        open(txt_path, "w").close()
        print("No answer region detected")
        continue

    best_idx = max(answer_indices, key=lambda i: confs[i])

    x1, y1, x2, y2 = map(float, boxes[best_idx])
    conf = float(confs[best_idx])

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    detections_for_txt = [
        (ANSWER_CLASS_ID, x1, y1, x2, y2, conf)
    ]

    xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))
    crop = image[yi1:yi2, xi1:xi2]

    crop_name = f"{base_name}_ANSWER.jpg"
    cv2.imwrite(os.path.join(output_folder, crop_name), crop)

    print(f"Saved: {crop_name} | confidence: {conf:.3f}")

    write_yolo_txt_for_image(txt_path, detections_for_txt, w, h)
    print(f"Label written: {txt_path}")

    vis = image.copy()
    cv2.rectangle(vis, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)
    cv2.imshow("Answer Region Detected", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nAll answer regions cropped successfully!")

