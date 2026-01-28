# save as visualize_labels.py
import cv2
import os

DATASET = "/home/ashik-christober/Desktop/ml projects/ml_omr_model/dataset/bubble data"
img_dir = os.path.join(DATASET, "train", "images")
lbl_dir = os.path.join(DATASET, "train", "labels")
out_dir = os.path.join(DATASET, "debug_vis")
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(img_dir):
    if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png")):
        continue
    img_path = os.path.join(img_dir, fname)
    base = os.path.splitext(fname)[0]
    lbl_path = os.path.join(lbl_dir, base + ".txt")
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = parts[:5]
                cls = int(float(cls))
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                # convert to pixel coords
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(img, str(cls), (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.imwrite(os.path.join(out_dir, fname), img)
print("Saved visuals to", out_dir)
