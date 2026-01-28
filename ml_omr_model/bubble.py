from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
from typing import List, Dict
import traceback


CONF_THRESH = 0.35     # keep detections above this confidence
MIN_BOX_AREA = 50      # filter tiny boxes in px^2
Y_TOLERANCE = 20       # pixels tolerance when grouping rows
EXPECTED_COLUMNS = 5   # number of options per question (A..E)
DARKNESS_ADAPT_FACTOR = 0.6
TIE_RELATIVE = 0.10    # two fills within 10% of max -> tie


def load_grayscale_np(path, target_size=None):
    img = Image.open(path).convert("L")
    if target_size:
        img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    return img, arr

def normalize_image(arr):
    a = arr.copy()
    mn, mx = float(a.min()), float(a.max())
    if mx - mn > 1e-6:
        a = (a - mn) / (mx - mn)
    return a

def bbox_darkness(arr: np.ndarray, bbox):
    x1, y1, x2, y2 = bbox
    h, w = arr.shape
    # clip
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = arr[y1:y2, x1:x2]
    return 1.0 - float(crop.mean())

def parse_bubble_boxes(result, conf_thresh=CONF_THRESH, min_area=MIN_BOX_AREA, class_name='bubble'):
    """
    Parse Ultralytics Results object and return list of boxes:
    [{"bbox": (x1,y1,x2,y2), "cx": cx, "cy": cy, "conf": conf}, ...]
    robust to result.names being list or dict.
    """
    boxes = []
    names = getattr(result, "names", {})
    for b in result.boxes:
        try:
            cls = int(b.cls[0])
        except Exception:
            # fallback if structure differs
            try:
                cls = int(b.cls)
            except Exception:
                cls = 0
        # robust name lookup
        if isinstance(names, (list, tuple)):
            name = names[cls] if cls < len(names) else str(cls)
        else:
            name = names.get(cls, str(cls))
        conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
        if name != class_name:
            continue
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        w, h = (x2 - x1), (y2 - y1)
        if w * h < min_area:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        boxes.append({"bbox": (x1, y1, x2, y2), "cx": cx, "cy": cy, "conf": conf})
    return boxes

def group_by_rows(boxes, y_tolerance=Y_TOLERANCE):
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: b["cy"])
    groups = []
    current = [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        if abs(b["cy"] - current[-1]["cy"]) <= y_tolerance:
            current.append(b)
        else:
            groups.append(sorted(current, key=lambda x: x["cx"]))
            current = [b]
    groups.append(sorted(current, key=lambda x: x["cx"]))
    return groups

def evaluate_rows(groups, img_arr, expected_columns=EXPECTED_COLUMNS, adapt_factor=DARKNESS_ADAPT_FACTOR):
    results = {}
    row_baselines = []
    # compute baseline per-row
    for row in groups:
        darks = [bbox_darkness(img_arr, b["bbox"]) for b in row[:expected_columns]]
        row_baselines.append(np.mean(darks) if darks else 0.0)
    global_median = float(np.median(row_baselines)) if row_baselines else 0.0

    for qno, row in enumerate(groups, start=1):
        # limit to expected_columns (leftmost)
        if len(row) > expected_columns:
            row = row[:expected_columns]
        dark_levels = [bbox_darkness(img_arr, b["bbox"]) for b in row]
        if not dark_levels:
            results[qno] = {"selected": None, "note": "no_bubbles_detected", "darks": []}
            continue

        max_dark = max(dark_levels)
        adaptive_threshold = max(0.15, global_median * adapt_factor)
        if max_dark < adaptive_threshold:
            results[qno] = {"selected": None, "note": "blank_or_light_fill", "darks": dark_levels}
            continue

        # detect ties: values within TIE_RELATIVE of max
        ties = [i for i, v in enumerate(dark_levels) if max_dark - v <= TIE_RELATIVE * max_dark]
        if len(ties) > 1:
            results[qno] = {"selected": None, "note": "multiple_or_tie", "darks": dark_levels}
            continue

        selected_index = int(np.argmax(dark_levels))
        selected_letter = chr(ord('A') + selected_index)
        results[qno] = {"selected": selected_letter, "note": "", "darks": dark_levels}
    return results

def save_csv(all_results: Dict[str, Dict], out_csv: str):
    rows = []
    for image_name, res in all_results.items():
        for q, v in res.items():
            rows.append({
                "Image": image_name,
                "Question": q,
                "Answer": v.get("selected"),
                "Note": v.get("note", ""),
                "Darks": "|".join(f"{d:.3f}" for d in v.get("darks", []))
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved CSV -> {out_csv}")

def annotate_and_save(pil_img: Image.Image, boxes: List[dict], groups: List[List[dict]], results: dict, out_path: str):
    """
    Draw detection boxes and per-row selected letter (or '-' when none) and save annotated image.
    """
    try:
        img_rgb = pil_img.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # draw boxes
        for b in boxes:
            x1, y1, x2, y2 = map(int, b["bbox"])
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        # write selected letter near each row's leftmost bubble
        for qno, row in enumerate(groups, start=1):
            if not row:
                continue
            left = int(row[0]["bbox"][0])
            top = int(min([r["bbox"][1] for r in row]))
            sel = results.get(qno, {}).get("selected")
            note = results.get(qno, {}).get("note", "")
            text = f"{qno}: {sel if sel else '-'} {note}"
            ty = max(0, top - 12)
            draw.text((left, ty), text, fill=(0, 0, 255), font=font)

        # ensure parent dir exists
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

        img_rgb.save(out_path)
        print(f"[+] Annotated image saved -> {out_path}")
    except Exception as e:
        print(f"[!] Failed to annotate/save {out_path}: {e}")
        traceback.print_exc()
        raise

# ---------- main detection function ----------
def process_image(yolo_model: YOLO, img_path: str, annotate_dir: str = None):
    pil_img, arr = load_grayscale_np(img_path)
    norm_arr = normalize_image(arr)

    # run model on image path (ultralytics returns Results)
    results = yolo_model(img_path, imgsz=1280, conf=CONF_THRESH, verbose=False)
    res0 = results[0]

    boxes = parse_bubble_boxes(res0)
    groups = group_by_rows(boxes)

    evals = evaluate_rows(groups, norm_arr)
    # annotated image
    if annotate_dir:
        p = Path(img_path)
        out_name = p.stem + "_annotated" + p.suffix
        out_path = os.path.join(annotate_dir, out_name)
        print(f"[DBG] Saving annotation for {img_path} -> {out_path}")
        annotate_and_save(pil_img, boxes, groups, evals, out_path)
    return evals

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Detect and evaluate OMR bubbles with a trained YOLO model.")
    parser.add_argument(
        "--model",
        required=False,
        default="/home/ashik-christober/runs/detect/omr_bubble_run3/weights/best.pt",
        help="Path to trained model (default used if not provided)"
    )
    parser.add_argument(
        "--input",
        required=False,
        default="/home/ashik-christober/Desktop/ml projects/ml_omr_model/dataset/bubble data/test",
        help="Path to image or folder (default used if not provided)"
    )
    parser.add_argument("--out", default="results.csv", help="Output CSV file")
    parser.add_argument("--annotate_dir", default=None, help="Directory to save annotated images (optional)")
    parser.add_argument("--debug", action="store_true", help="Print extra debug info")
    args = parser.parse_args()

    model_path = args.model
    input_path = args.input
    out_csv = args.out
    annotate_dir = args.annotate_dir
    debug = args.debug

    if annotate_dir:
        os.makedirs(annotate_dir, exist_ok=True)

    print(f"[+] Loading model from: {model_path}")
    yolo_model = YOLO(model_path)

    all_results = {}

    p = Path(input_path)
    print(f"[DBG] input_path = {input_path}")
    print(f"[DBG] exists={p.exists()} is_file={p.is_file()} is_dir={p.is_dir()}")

    if not p.exists():
        raise ValueError(f"[ERR] Input path not found: {input_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if p.is_file():
        images = [str(p)]
    else:
        # recursive search (handles nested image folders)
        images = sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in exts])

    print(f"[DBG] images found: {len(images)}")
    if len(images) > 0:
        print("[DBG] sample:", images[:10])

    if not images:
        print("[!] No images found in the input path - please check the folder and extensions.")
        return

    for img in images:
        print(f"[+] Processing: {img}")
        try:
            res = process_image(yolo_model, img, annotate_dir)
            all_results[os.path.basename(img)] = res
            if debug:
                # print per-row darks for debugging
                for q, v in (res.items()):
                    print(f"[DBG] {os.path.basename(img)} Q{q}: selected={v.get('selected')} note={v.get('note')} darks={v.get('darks')}")
        except Exception as e:
            print(f"[!] Failed on {img}: {e}")
            traceback.print_exc()

    save_csv(all_results, out_csv)

if __name__ == "__main__":
    main()
