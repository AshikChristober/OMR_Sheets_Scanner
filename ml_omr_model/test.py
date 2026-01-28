# detect_and_show.py
from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import sys

# ---------- USER CONFIG ----------
model_path = "/home/ashik-christober/runs/detect/omr_bubble_run3/weights/best.pt"
test_image_folder = "/home/ashik-christober/Desktop/ml projects/ml_omr_model/dataset/bubble data/test/images"
output_folder = "/home/ashik-christober/Desktop/ml projects/ml_omr_model/cropped_outputs"
annotated_folder = os.path.join(output_folder, "annotated")
conf_threshold = 0.5
# ----------------------------------

os.makedirs(output_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

# load model
model = YOLO(model_path)
print("✅ Model loaded successfully!")
print("Classes:", model.names)

def write_yolo_txt_for_image(txt_path, detections, img_w, img_h):
    with open(txt_path, "w") as f:
        for cls_id, x1, y1, x2, y2, conf in detections:
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            bw = (x2 - x1)
            bh = (y2 - y1)
            # normalize
            xc_n = xc / img_w
            yc_n = yc / img_h
            bw_n = bw / img_w
            bh_n = bh / img_h
            f.write(f"{int(cls_id)} {xc_n:.6f} {yc_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")

def safe_array_from_plot(plot_ret):
    """
    Ultralytics result.plot() might return numpy array or PIL.Image.
    Convert to BGR uint8 array suitable for cv2.imshow/save.
    """
    if isinstance(plot_ret, np.ndarray):
        # usually RGB or BGR? ultralytics usually returns RGB numpy
        arr = plot_ret
        # if shape 3-channels, convert RGB->BGR for cv2
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    else:
        # try converting PIL.Image -> numpy
        try:
            arr = np.array(plot_ret)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception:
            return None

# iterate images sorted for consistent order
p = Path(test_image_folder)
if not p.exists():
    print("[ERR] test image folder not found:", test_image_folder)
    sys.exit(1)

image_files = sorted([x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
if not image_files:
    print("[ERR] No images found in:", test_image_folder)
    sys.exit(1)

for img_path in image_files:
    img_name = img_path.name
    base_name = img_path.stem
    txt_path = str(img_path.with_suffix(".txt"))
    print(f"\n🔍 Processing: {img_name}")

    # run model: using __call__ is fine (predict is alias)
    results = model(str(img_path), conf=conf_threshold)  # returns Results object (sequence)
    # usually one result per image (we provided a single image)
    if len(results) == 0:
        print("⚠️ No result returned for", img_name)
        continue
    res0 = results[0]

    # robust extraction of boxes / classes / confs
    boxes_np = np.zeros((0,4))
    cls_ids = np.array([], dtype=int)
    confs = np.array([], dtype=float)
    try:
        # ultralytics v8 often supports these attributes as tensors
        boxes_np = res0.boxes.xyxy.cpu().numpy() if hasattr(res0.boxes, "xyxy") else np.zeros((0,4))
        cls_ids = res0.boxes.cls.cpu().numpy().astype(int) if hasattr(res0.boxes, "cls") else np.array([], dtype=int)
        confs = res0.boxes.conf.cpu().numpy() if hasattr(res0.boxes, "conf") else np.array([], dtype=float)
    except Exception:
        # fallback: iterate
        bxs = []
        cids = []
        cfs = []
        for b in res0.boxes:
            xyxy = getattr(b, "xyxy", None)
            cls = getattr(b, "cls", None)
            conf = getattr(b, "conf", None)
            if xyxy is None:
                continue
            # xyxy might be tensor-like or list
            arr = np.array(xyxy).astype(float).reshape(-1)[:4]
            bxs.append(arr)
            try:
                cids.append(int(cls[0]) if hasattr(cls, "__len__") else int(cls))
            except Exception:
                cids.append(0)
            try:
                cfs.append(float(conf[0]) if hasattr(conf, "__len__") else float(conf))
            except Exception:
                cfs.append(0.0)
        if len(bxs) > 0:
            boxes_np = np.array(bxs, dtype=float)
            cls_ids = np.array(cids, dtype=int)
            confs = np.array(cfs, dtype=float)

    # load image with cv2 for cropping and dims
    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        print("❌ Could not load image with cv2:", img_path)
        continue
    h, w = img_cv.shape[:2]

    detections_for_txt = []

    if boxes_np.shape[0] == 0:
        # create empty txt so annotation training pipeline won't break
        open(txt_path, "w").close()
        print("⚠️ No detections — wrote empty txt:", txt_path)
    else:
        print(f"✅ {len(boxes_np)} detections in {img_name}")

        # crop each detected box and record detection for txt
        for j, box in enumerate(boxes_np):
            x1, y1, x2, y2 = map(float, box)
            cid = int(cls_ids[j]) if j < len(cls_ids) else 0
            conf = float(confs[j]) if j < len(confs) else 0.0
            # clamp
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(w - 1.0, x2), min(h - 1.0, y2)
            detections_for_txt.append((cid, x1, y1, x2, y2, conf))

            xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))
            if xi2 > xi1 and yi2 > yi1:
                crop_name = f"{base_name}_{cid}_{j}.jpg"
                cv2.imwrite(os.path.join(output_folder, crop_name), img_cv[yi1:yi2, xi1:xi2])
                print(f"💾 Saved crop: {crop_name} ({xi1},{yi1},{xi2},{yi2})")

        # write yolo format label file next to source image
        write_yolo_txt_for_image(txt_path, detections_for_txt, w, h)
        print(f"💾 Written YOLO labels: {txt_path}")

    # create annotated visualization using result.plot()
    try:
        plotted = res0.plot()  # may return np.array (RGB) or PIL.Image
        ann_arr = safe_array_from_plot(plotted)
        if ann_arr is None:
            print("⚠️ Could not convert plotted image to array.")
        else:
            # save annotated image to annotated_folder
            ann_name = f"{base_name}_annotated.jpg"
            ann_path = os.path.join(annotated_folder, ann_name)
            cv2.imwrite(ann_path, ann_arr)
            print(f"🖼️ Saved annotated image -> {ann_path}")

            # show annotated image (if GUI available)
            try:
                cv2.imshow("Annotated", ann_arr)
                print("🖼️ Press any key to continue (close window)...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                # environment may be headless (e.g., remote server) — skip showing
                print("ℹ️ cv2.imshow failed (headless environment?). Open the saved annotated image manually.")
    except Exception as e:
        print("[!] Failed to create/save annotated visualization:", e)
        import traceback; traceback.print_exc()

print("\n✅ All done. Check crops, labels, and annotated images in:", output_folder)
