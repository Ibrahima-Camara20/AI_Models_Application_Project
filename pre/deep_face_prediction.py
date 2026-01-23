import os
import pandas as pd
import numpy as np
import re
import time
import tensorflow as tf
from deepface import DeepFace
from tqdm import tqdm

# ================= 0. CONFIGURATION GPU =================
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" GPU ACTIVÉ : {len(gpus)} trouvé(s). Prêt pour ArcFace.")
        except RuntimeError as e:
            print(f"Erreur GPU : {e}")
    else:
        print(" ATTENTION : Pas de GPU détecté.")

# ================= 1. PARAMÈTRES ARCFACE =================
DEFAULT_MODEL = "ArcFace" # Le modèle "État de l'art"
# Le seuil pour ArcFace avec la distance Cosinus. 
# 0.68 est un bon point de départ pour ce dataset.
DEFAULT_THRESHOLD = 0.68 

DEFAULT_DB_PATH = "img/celebrity_db" 
DEFAULT_TEST_PATH = "tests"
DEFAULT_OUTPUT_CSV = "img/stats/arcface_final_results_optimized.csv"

# ================= 2. FONCTIONS =================
def clean_label(filename):
    """Nettoyage des noms"""
    name = os.path.splitext(filename)[0]
    name = name.replace("face_", "").replace("pins_", "")
    if "-person" in name:
        name = name.split("-person")[0]
    name = re.sub(r'\d+', '', name)
    name = name.replace("_", " ").replace("-", " ")
    return re.sub(r'\s+', ' ', name).strip().lower()

def get_embedding(img_path, model_name=DEFAULT_MODEL):
    """Calcul de l'embedding ArcFace"""
    try:
        # enforce_detection=False car on a déjà cropé
        return DeepFace.represent(img_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
    except:
        return None

# ================= 3. CHARGEMENT DB (MATRICE ARCFACE) =================
def load_database(db_path, model_name=DEFAULT_MODEL):
    print(f"\n--- 1. Encodage de la DB ({model_name}) ---")

    if not os.path.exists(db_path):
        print(f" ERREUR : Le dossier DB est introuvable : {db_path}")
        return None

    db_names = []
    db_matrix = []
    db_files = [f for f in os.listdir(db_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Traitement de {len(db_files)} références...")

    # Boucle de chargement
    for f in tqdm(db_files, desc="Encodage DB"):
        name = clean_label(f)
        emb = get_embedding(os.path.join(db_path, f), model_name)
        if emb:
            db_names.append(name)
            db_matrix.append(emb)

    if not db_matrix:
        print(" ERREUR : DB vide.")
        return None

    # Transformation en NumPy (Vectorisation)
    db_matrix = np.array(db_matrix) 
    db_names = np.array(db_names)

    # Pré-calcul des normes pour le Cosinus
    db_norms = np.linalg.norm(db_matrix, axis=1)
    print(f" Matrice ArcFace prête : {db_matrix.shape}")
    
    return db_matrix, db_names, db_norms

# ================= 4. TEST RAPIDE (VECTORISÉ) =================
def predict_arcface(test_path, db_data, output_csv=None, model_name=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD):
    print("\n--- 2. Lancement du Test ArcFace ---")

    if not os.path.exists(test_path):
        print(f" ERREUR : Dossier Test introuvable : {test_path}")
        return None
        
    if db_data is None:
        print(" ERREUR : Données DB invalides.")
        return None
        
    db_matrix, db_names, db_norms = db_data

    all_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total_files = len(all_files)

    results = []
    batch_correct = 0
    start_time = time.time()

    print(f"Analyse de {total_files} images de test...")

    for i, f in enumerate(all_files):
        img_path = os.path.join(test_path, f)
        true_label = clean_label(f)
        
        # A. Extraction GPU
        test_emb = get_embedding(img_path, model_name)
        if test_emb is None: continue
        
        # B. Comparaison Vectorielle (Optimisation)
        test_emb = np.array(test_emb)
        test_norm = np.linalg.norm(test_emb)
        
        if test_norm == 0: continue # Sécurité

        # Produit scalaire matrixiel
        dot_products = np.dot(db_matrix, test_emb)
        
        # Similarités et Distances
        similarities = dot_products / (db_norms * test_norm)
        distances = 1 - similarities
        
        # Meilleur candidat
        best_idx = np.argmin(distances)
        min_dist = distances[best_idx]
        best_match = db_names[best_idx]

        # C. Verdict
        is_correct = (best_match == true_label and min_dist <= threshold)
        
        if is_correct: 
            batch_correct += 1
            status = "OK"
        elif best_match == true_label:
            status = "SEUIL"
        else:
            status = "MISS"

        # D. Stockage
        results.append({
            "filename": f,
            "true_label": true_label,
            "predicted": best_match if min_dist <= threshold else "unknown",
            "distance": round(min_dist, 4),
            "status": status,
            "correct": is_correct
        })

        # E. Logs tous les 100 fichiers
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            acc = batch_correct / (i+1)
            print(f" {i+1}/{total_files} | Acc ArcFace: {acc:.2%}")
            
            # Sauvegarde intermédiaire
            if output_csv:
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                pd.DataFrame(results).to_csv(output_csv, index=False)

    # ================= 5. RAPPORT FINAL =================
    final_df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_df.to_csv(output_csv, index=False)

    if len(final_df) > 0:
        acc_final = final_df['correct'].mean()
        elapsed_total = time.time() - start_time
        
        print("\n" + "="*40)
        print(f" TRAITEMENT TERMINÉ en {elapsed_total/60:.1f} minutes")
        print(f" ACCURACY FINALE (ARCFACE) : {acc_final:.2%}")
        if output_csv:
            print(f" Résultats : {output_csv}")
        print("="*40)
        return final_df
    else:
        print("\n Aucun résultat généré.")
        return None

def main():
    configure_gpu()
    
    # 1. Charger la DB
    db_data = load_database(DEFAULT_DB_PATH)
    
    # 2. Lancer la prédiction
    if db_data:
        predict_arcface(DEFAULT_TEST_PATH, db_data, DEFAULT_OUTPUT_CSV)

if __name__ == "__main__":
    main()