import os
import cv2
import time
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from ultralytics import YOLO
import unicodedata
import re

# ---------- CONFIG ----------
INPUT_DIR = "data_test"         # ajuste si besoin
OUTPUT_DIR = "tests"            # ajuste si besoin
WEIGHT_PATH = "yolo11n.pt"
BATCH_SIZE = 64                # augmente si GPU peut le tenir (VRAM)
WRITE_WORKERS = 8              # threads pour écrire les crops
MIN_BOX_AREA = 20 * 20         # ignorer très petits boxes (optionnel)


def normalize_filename(filename):
    """Normalise un nom de fichier en enlevant les accents et caractères spéciaux."""
    # Enlever l'extension
    name, ext = os.path.splitext(filename)
    
    # Enlever les accents
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ASCII', 'ignore').decode('ASCII')
    
    # Remplacer les caractères spéciaux par des underscores
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Enlever les underscores multiples
    name = re.sub(r'_+', '_', name)
    
    return name + ext


def get_all_images(directory):
    """Récupère tous les chemins d'images dans un dossier."""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def safe_read_rgb(path):
    """Lit une image en RGB de manière sécurisée."""
    img = cv2.imread(path)
    if img is None:
        return None
    # Convert BGR -> RGB for ultralytics
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_crop(path, crop):
    """Écrit un crop RGB sur disque en BGR."""
    # crop is RGB; convert back to BGR for cv2.imwrite
    try:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        return True
    except Exception:
        return False


def yolo_detection_single(image_path, output_dir, return_boxes=False):
    """Détecte les personnes dans une seule image avec YOLO.
    
    Args:
        image_path: Chemin vers l'image à traiter
        output_dir: Dossier de sortie pour les crops
        return_boxes: Si True, retourne aussi les coordonnées des boxes
        
    Returns:
        Si return_boxes=False: Nombre de personnes détectées
        Si return_boxes=True: tuple (count, boxes) où boxes = [(x1, y1, x2, y2, confidence), ...]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # Chargement du modèle
    model = YOLO(WEIGHT_PATH).to(device)
    
    # Prédiction (classes=[0] pour "person" seulement)
    results = model.predict(image_path, classes=[0], verbose=False, device=device)
    
    persons_detected = 0
    boxes_list = []
    img_name = os.path.basename(image_path)
    
    for r in results:
        img = r.orig_img
        
        if r.boxes:
            for j, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                
                # Stocker les coordonnées si demandé
                if return_boxes:
                    boxes_list.append((x1, y1, x2, y2, conf))
                
                # Crop
                human_crop = img[y1:y2, x1:x2]
                
                # Nommage avec normalisation
                base_name = os.path.splitext(img_name)[0]
                new_filename = f"{base_name}-person-{j}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                new_filename = normalize_filename(new_filename)
                
                cv2.imwrite(os.path.join(output_dir, new_filename), human_crop)
                persons_detected += 1
    
    if return_boxes:
        return persons_detected, boxes_list
    return persons_detected


def yolo_detector(input_dir, output_dir, weight_path=WEIGHT_PATH, batch_size=BATCH_SIZE):
    """Détecte les personnes dans toutes les images d'un dossier avec YOLO."""
    if not os.path.exists(input_dir):
        print(f"[ERROR] Le dossier {input_dir} est introuvable.")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = "GPU" if "cuda" in device_str else "CPU"
    print(f"Device detected: {device_name} ({device_str})")

    # Load model
    model = YOLO(weight_path)
    try:
        model.to(device_str)
    except Exception:
        pass

    all_images = get_all_images(input_dir)
    n_images = len(all_images)
    if n_images == 0:
        print("[WARN] Aucune image trouvée.")
        return 0

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
        results = model.predict(batch_imgs, device=device_str, verbose=False, classes=[0])
        t1 = time.time()
        infer_time += (t1 - t0)

        # 3) POSTPROCESS: extract boxes and schedule writes
        t0 = time.time()
        for r_idx, r in enumerate(results):
            original_path = valid_paths[r_idx]
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            
            try:
                if len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()  # shape (N,4)
            except Exception:
                try:
                    xyxy = np.array(r.boxes.xyxy)
                except Exception:
                    continue

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
                    continue

                # Crop (numpy slicing is cheap)
                crop = img_rgb[y1:y2, x1:x2]

                # prepare filename with normalization
                new_filename = f"{base_name}-person-{j}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                new_filename = normalize_filename(new_filename)
                out_path = os.path.join(output_dir, new_filename)

                # schedule write in threadpool
                futures.append(writer_pool.submit(write_crop, out_path, crop))
                total_extracted += 1
        t1 = time.time()
        post_time += (t1 - t0)

        # Free memory
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
    print(f"  - Chargement         : {load_time:.2f}s")
    print(f"  - Inférence          : {infer_time:.2f}s")
    print(f"  - Post-traitement    : {post_time:.2f}s")
    print(f"  - Écriture I/O       : {io_time:.2f}s")
    print("---------------------------")
    
    return total_extracted


if __name__ == "__main__":
    yolo_detector(INPUT_DIR, OUTPUT_DIR)
