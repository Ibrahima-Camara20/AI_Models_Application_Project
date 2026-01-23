import os
import cv2
import time
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from ultralytics import YOLO

# ---------- CONFIG ----------
INPUT_DIR = "data_test"         # ajuste si besoin
OUTPUT_DIR = "tests"            # ajuste si besoin
WEIGHT_PATH = "yolo11n.pt"
BATCH_SIZE = 64                # augmente si GPU peut le tenir (VRAM)
WRITE_WORKERS = 8              # threads pour écrire les crops
MIN_BOX_AREA = 20 * 20         # ignorer très petits boxes (optionnel)

# ---------- Helpers ----------
def get_all_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def safe_read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    # Convert BGR -> RGB for ultralytics
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_crop(path, crop):
    # crop is RGB; convert back to BGR for cv2.imwrite
    try:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        return True
    except Exception:
        return False

# ---------- Main detector ----------
def yolo_detector(input_dir, output_dir, weight_path=WEIGHT_PATH, batch_size=BATCH_SIZE):
    if not os.path.exists(input_dir):
        print(f"[ERROR] Le dossier {input_dir} est introuvable.")
        return

    os.makedirs(output_dir, exist_ok=True)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = "GPU" if "cuda" in device_str else "CPU"
    print(f"Device detected: {device_name} ({device_str})")

    # Load model (weights loaded to CPU/GPU as required)
    model = YOLO(weight_path)
    # model.to(device_str) is optional — we'll pass device to predict call
    # but to be explicit:
    try:
        model.to(device_str)
    except Exception:
        pass

    all_images = get_all_images(input_dir)
    n_images = len(all_images)
    if n_images == 0:
        print("[WARN] Aucune image trouvée.")
        return

    # Stats timers
    t_start = time.time()
    load_time = 0.0
    infer_time = 0.0
    post_time = 0.0
    io_time = 0.0
    total_extracted = 0

    # Thread pool for disk writes (asynchronous)
    writer_pool = ThreadPoolExecutor(max_workers=WRITE_WORKERS)
    futures = []

    # Process in batches: preload images, run inference on arrays
    for i in range(0, n_images, batch_size):
        batch_paths = all_images[i : i + batch_size]

        # 1) LOAD batch into memory (RGB arrays)
        t0 = time.time()
        batch_imgs = []
        valid_paths = []
        for p in batch_paths:
            img = safe_read_rgb(p)
            if img is None:
                # skip unreadable images silently
                continue
            batch_imgs.append(img)
            valid_paths.append(p)
        t1 = time.time()
        load_time += (t1 - t0)

        if len(batch_imgs) == 0:
            continue

        # 2) INFERENCE (pass list of numpy arrays)
        t0 = time.time()
        # Use model.predict on arrays; pass device explicitly
        # set verbose=False to silence prints
        # set save=False (we don't want ultralytics to save outputs)
        # set conf and iou as you like; here we keep defaults
        results = model.predict(batch_imgs, device=device_str, verbose=False)
        t1 = time.time()
        infer_time += (t1 - t0)

        # 3) POSTPROCESS: extract boxes and schedule writes
        t0 = time.time()
        # results is a list of Results objects of same length as batch_imgs
        for r_idx, r in enumerate(results):
            # Use the corresponding original path to name files
            original_path = valid_paths[r_idx]
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            # r.boxes.xyxy -> tensor (N,4). Convert once to numpy
            try:
                if len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()  # shape (N,4)
            except Exception:
                # fallback: try r.boxes.xyxy directly (some ultralytics versions)
                try:
                    xyxy = np.array(r.boxes.xyxy)
                except Exception:
                    continue

            # r.orig_img: sometimes provided, else use batch_imgs[r_idx]
            img_rgb = getattr(r, "orig_img", batch_imgs[r_idx])

            for j, box in enumerate(xyxy):
                # Convert to ints and clip to image bounds
                x1, y1, x2, y2 = box
                x1 = int(max(0, math.floor(x1)))
                y1 = int(max(0, math.floor(y1)))
                x2 = int(min(img_rgb.shape[1], math.ceil(x2)))
                y2 = int(min(img_rgb.shape[0], math.ceil(y2)))
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                if w * h < MIN_BOX_AREA:
                    # ignore tiny detections (optional)
                    continue

                # Crop (numpy slicing is cheap)
                crop = img_rgb[y1:y2, x1:x2]

                # prepare filename
                new_filename = f"{base_name}-person-{j}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                out_path = os.path.join(output_dir, new_filename)

                # schedule write in threadpool
                futures.append(writer_pool.submit(write_crop, out_path, crop))
                total_extracted += 1
        t1 = time.time()
        post_time += (t1 - t0)

        # Optionally: free some memory references
        del batch_imgs, results

    # wait for all IO writes to finish
    t0 = time.time()
    for f in as_completed(futures):
        try:
            f.result()
        except Exception:
            pass
    t1 = time.time()
    io_time += (t1 - t0)

    writer_pool.shutdown(wait=True)
    elapsed = time.time() - t_start

    # Summary
    print("---------- Résumé ----------")
    print(f"Images traitées        : {n_images}")
    print(f"Crops extraits         : {total_extracted}")
    print(f"Temps total            : {elapsed:.2f}s")
    print("---------------------------")

if __name__ == "__main__":
    yolo_detector(INPUT_DIR, OUTPUT_DIR)
