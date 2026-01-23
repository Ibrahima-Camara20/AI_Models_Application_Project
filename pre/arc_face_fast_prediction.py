import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import re
import numpy as np
import pandas as pd

import tensorflow as tf
from deepface import DeepFace

# optional: opencv for fast image preprocessing
try:
    import cv2
except Exception:
    cv2 = None

# ========== CONFIG ==========
MODEL_NAME = "ArcFace"
THRESHOLD = 0.68

DB_PATH = "img/celebrity_db_cropped"
TEST_PATH = "faces_dataset"
OUTPUT_CSV = "img/stats/arcface_final_results_optimized.csv"

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# Si tes images sont bien cropées sur le visage : True (beaucoup + rapide)
FAST_PREPROCESS = True
TARGET_SIZE = (112, 112)  # taille attendue par l'implémentation ArcFace
BATCH_SIZE_DB = 128
BATCH_SIZE_TEST = 64

# ========== GPU config ==========
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.backend.set_floatx('float32')
        print(f" GPU détecté : {len(gpus)} device(s).")
    except RuntimeError as e:
        print(f"Erreur GPU config: {e}")
else:
    print(" Pas de GPU détecté. Le script fonctionnera CPU-only.")

# ========== utilitaires ==========
def clean_label(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace("face_", "").replace("pins_", "")
    if "-person" in name:
        name = name.split("-person")[0]
    name = re.sub(r'\d+', '', name)
    name = name.replace("_", " ").replace("-", " ")
    return re.sub(r'\s+', ' ', name).strip().lower()

def list_image_files(folder: str):
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]

def fast_load_and_preprocess(path: str, target_size=TARGET_SIZE):
    """
    Chargement rapide : cv2 -> BGR->RGB resize, normalisation float32.
    Retourne None si lecture échoue.
    """
    try:
        if cv2 is None:
            # fallback via PIL
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img = img.resize(target_size, resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
        else:
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            arr = img.astype(np.float32)
        # normalisation compatible ArcFace implementations courantes
        arr = (arr - 127.5) / 128.0
        return arr
    except Exception as e:
        print(f"[WARN] preprocess failed for {path}: {e}")
        return None

# ========== modèle (chargement) ==========
print("Chargement du modèle DeepFace (ArcFace)...")
model = DeepFace.build_model(MODEL_NAME)
print("Modèle chargé :", type(model))
# Montrer méthodes utiles pour debug (optionnel)
debug_methods = [m for m in dir(model) if m in ("predict", "model", "forward", "infer", "get_embedding", "get_emb")]
print("Méthodes détectées sur l'objet modèle:", debug_methods)

# ========== wrapper adaptatif ==========
def _model_predict_batch(model_obj, X_batch: np.ndarray):
    """
    Appelle le modèle de façon robuste :
    - keras-like .predict
    - wrapper .model.predict
    - .forward / .infer (tentative)
    - .get_embedding (par image, lent)
    Retourne np.array shape (N, D) dtype=float32
    """
    import numpy as _np
    # 1) keras-like
    if hasattr(model_obj, "predict"):
        out = model_obj.predict(X_batch)
        return _np.asarray(out, dtype=_np.float32)

    # 2) wrapper contenant un keras model
    if hasattr(model_obj, "model") and hasattr(model_obj.model, "predict"):
        out = model_obj.model.predict(X_batch)
        return _np.asarray(out, dtype=_np.float32)

    # 3) forward / infer: tenter d'appeler directement
    if hasattr(model_obj, "forward"):
        try:
            out = model_obj.forward(X_batch)
            return _np.asarray(out, dtype=_np.float32)
        except Exception as e:
            # Peut-être que forward attend torch.Tensor ; on tente une conversion si torch dispo
            try:
                import torch
                X_t = torch.from_numpy(X_batch).permute(0,3,1,2).to(next(model_obj.parameters()).device) if hasattr(model_obj, "parameters") else torch.from_numpy(X_batch).to(next(model_obj.parameters()).device)
                out_t = model_obj.forward(X_t)
                out = out_t.cpu().detach().numpy()
                return _np.asarray(out, dtype=_np.float32)
            except Exception:
                raise RuntimeError(f"Erreur appel model.forward: {e}")

    if hasattr(model_obj, "infer"):
        try:
            out = model_obj.infer(X_batch)
            return _np.asarray(out, dtype=_np.float32)
        except Exception as e:
            raise RuntimeError(f"Erreur appel model.infer: {e}")

    # 4) get_embedding / get_emb : appel par image (lent, mais compatible)
    if hasattr(model_obj, "get_embedding") or hasattr(model_obj, "get_emb"):
        emb_list = []
        for img in X_batch:
            # certains clients attendent chemin fichier, d'autres array; on tente array
            if hasattr(model_obj, "get_embedding"):
                emb = model_obj.get_embedding(img)
            else:
                emb = model_obj.get_emb(img)
            emb_list.append(np.asarray(emb, dtype=np.float32))
        return np.vstack(emb_list)

    # 5) fallback: lever erreur (on évite d'utiliser DeepFace.represent en batch ici)
    raise RuntimeError("Impossible d'appeler le modèle en batch : aucune méthode batch-compatible détectée.")

# ========== helper: batch embeddings from file paths ==========
def batch_embeddings_from_paths(paths, batch_size=64, fast_preprocess=True):
    """
    Précharge et prétraite les images par batch, puis appelle _model_predict_batch.
    Retourne (embeddings np.array shape=(N,D), elapsed_seconds)
    """
    embeddings = []
    t0 = time.time()
    batch_imgs = []
    for p in paths:
        if fast_preprocess:
            arr = fast_load_and_preprocess(p)
        else:
            # utiliser la fonction DeepFace (plus lente mais fiable)
            from deepface import DeepFace as _DF
            arr = _DF.functions.preprocess_face(img=p, target_size=TARGET_SIZE, enforce_detection=False)
        if arr is None:
            continue
        batch_imgs.append(arr)
        if len(batch_imgs) >= batch_size:
            X = np.stack(batch_imgs, axis=0).astype(np.float32)
            preds = _model_predict_batch(model, X)
            preds = np.asarray(preds, dtype=np.float32)
            # normaliser immédiatement
            norms = np.linalg.norm(preds, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            preds = preds / norms
            embeddings.append(preds)
            batch_imgs = []
    # leftover
    if batch_imgs:
        X = np.stack(batch_imgs, axis=0).astype(np.float32)
        preds = _model_predict_batch(model, X)
        preds = np.asarray(preds, dtype=np.float32)
        norms = np.linalg.norm(preds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        preds = preds / norms
        embeddings.append(preds)
    elapsed = time.time() - t0
    if len(embeddings) == 0:
        return np.zeros((0, 0), dtype=np.float32), elapsed
    return np.vstack(embeddings), elapsed

# ========== 1) Encode DB ==========
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"DB_PATH introuvable: {DB_PATH}")
db_files = list_image_files(DB_PATH)
db_paths = [os.path.join(DB_PATH, f) for f in db_files]
db_names = [clean_label(f) for f in db_files]

print(f"Encodage DB ({len(db_paths)} images) — batch_size={BATCH_SIZE_DB} ...")
db_embeddings, t_db = batch_embeddings_from_paths(db_paths, batch_size=BATCH_SIZE_DB, fast_preprocess=FAST_PREPROCESS)
if db_embeddings.size == 0:
    raise RuntimeError("Aucune embedding produite pour la DB. Vérifie les images / preprocessing.")
print(f"DB encodée en {t_db:.2f}s — shape={db_embeddings.shape}")

# Normalisation déjà faite dans batch_embeddings_from_paths mais au cas où :
db_norms = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
db_norms[db_norms == 0] = 1.0
db_embeddings_normed = db_embeddings / db_norms
db_embeddings_normed = db_embeddings_normed.astype(np.float32)

# ========== 2) Traitement des images de test (batch + comparaison) ==========
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"TEST_PATH introuvable: {TEST_PATH}")
test_files = list_image_files(TEST_PATH)
test_paths = [os.path.join(TEST_PATH, f) for f in test_files]
total = len(test_paths)
print(f"Traitement {total} images de test — batch_size={BATCH_SIZE_TEST}")

results = []
start_total = time.time()
embed_time = 0.0
compare_time = 0.0
processed = 0

# Process test images by batches (no tqdm)
for i in range(0, total, BATCH_SIZE_TEST):
    batch_paths = test_paths[i:i + BATCH_SIZE_TEST]
    preproc_list = []
    file_list = []
    t_pre = time.time()
    for p in batch_paths:
        if FAST_PREPROCESS:
            arr = fast_load_and_preprocess(p)
        else:
            from deepface import DeepFace as _DF
            arr = _DF.functions.preprocess_face(img=p, target_size=TARGET_SIZE, enforce_detection=False)
        if arr is None:
            continue
        preproc_list.append(arr)
        file_list.append(p)
    t_pre_end = time.time()
    embed_time += (t_pre_end - t_pre)

    if len(preproc_list) == 0:
        continue

    X = np.stack(preproc_list, axis=0).astype(np.float32)
    t0 = time.time()
    preds = _model_predict_batch(model, X)
    preds = np.asarray(preds, dtype=np.float32)
    # normalize
    norms = np.linalg.norm(preds, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    preds = preds / norms
    t1 = time.time()
    embed_time += (t1 - t0)

    # comparaison vectorisée pour chaque embedding du batch
    for j, emb in enumerate(preds):
        fname = os.path.basename(file_list[j])
        true_label = clean_label(fname)

        t_comp0 = time.time()
        sims = np.dot(db_embeddings_normed, emb)  # cos similarities
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        min_dist = float(1.0 - best_sim)
        best_match = db_names[best_idx]
        compare_time += (time.time() - t_comp0)

        is_correct = (best_match == true_label and min_dist <= THRESHOLD)
        if is_correct:
            status = "OK"
        elif best_match == true_label:
            status = "SEUIL"
        else:
            status = "MISS"

        results.append({
            "filename": fname,
            "true_label": true_label,
            "predicted": best_match if min_dist <= THRESHOLD else "unknown",
            "distance": round(min_dist, 4),
            "similarity": round(best_sim, 4),
            "status": status,
            "correct": bool(is_correct)
        })
        processed += 1

    # sauvegarde intermédiaire
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

elapsed_total = time.time() - start_total

# ========== 3) Rapport ==========
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
acc = df['correct'].mean() if len(df) > 0 else 0.0

print("\n" + "="*40)
print(f"FIN. Images traitées: {processed} / {total}")
print(f"Temps total: {elapsed_total:.2f}s")
print(f" - temps preprocessing+embedding (estimé): {embed_time:.2f}s")
print(f" - temps comparaison vectorielle (estimé): {compare_time:.2f}s")
print(f"Accuracy finale (ArcFace): {acc:.2%}")
print(f"Résultats sauvegardés -> {OUTPUT_CSV}")
print("="*40)
