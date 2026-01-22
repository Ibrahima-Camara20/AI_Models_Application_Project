import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import re
import time
import tensorflow as tf
from deepface import DeepFace
from tqdm import tqdm


# ================= 0. CONFIGURATION GPU =================
def configure_gpu():
    """Configure la mémoire GPU pour TensorFlow."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 GPU ACTIVÉ : {len(gpus)} trouvé(s). Mode Turbo enclenché.")
        except RuntimeError as e:
            print(f"Erreur GPU : {e}")
    else:
        print("⚠️ ATTENTION : Pas de GPU détecté. Le traitement sera plus lent.")

# ================= 1. PARAMÈTRES PAR DÉFAUT =================
DEFAULT_MODEL = "VGG-Face"
DEFAULT_THRESHOLD = 0.68
DEFAULT_DB_PATH = "img/celebrity_db"
DEFAULT_TEST_PATH = "tests"
DEFAULT_OUTPUT_CSV = "img/stats/vgg_final_results.csv"

# ================= 2. FONCTIONS UTILITAIRES =================
def clean_label(filename):
    """Nettoie le nom du fichier pour extraire le nom de la célébrité."""
    name = os.path.splitext(filename)[0]
    name = name.replace("face_", "").replace("pins_", "")
    
    if "-person" in name:
        name = name.split("-person")[0]
        
    name = re.sub(r'\d+', '', name) # Enlève les chiffres
    name = name.replace("_", " ").replace("-", " ")
    return re.sub(r'\s+', ' ', name).strip().lower()

def get_embedding(img_path, model_name=DEFAULT_MODEL):
    """Calcule l'embedding VGG d'une image."""
    try:
        # enforce_detection=False car les images sont déjà des crops de visages
        return DeepFace.represent(img_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
    except Exception as e:
        return None

# ================= 3. CHARGEMENT DE LA BASE DE DONNÉES =================
def load_database(db_path, model_name=DEFAULT_MODEL):
    """
    Charge et encode la base de données de référence.
    Retourne : (db_matrix, db_names, db_norms) ou None si échec.
    """
    print(f"\n--- 1. Encodage de la DB ({model_name}) en Matrice ---")

    if not os.path.exists(db_path):
        print(f"❌ ERREUR CRITIQUE : Le dossier DB est introuvable : {db_path}")
        return None

    db_names = []
    db_matrix = []
    db_files = [f for f in os.listdir(db_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Traitement de {len(db_files)} images de référence...")

    for f in tqdm(db_files, desc="Encodage DB"):
        name = clean_label(f)
        emb = get_embedding(os.path.join(db_path, f), model_name)
        if emb:
            db_names.append(name)
            db_matrix.append(emb)

    if not db_matrix:
        print("❌ ERREUR : Aucune image valide trouvée dans la DB.")
        return None

    db_matrix = np.array(db_matrix) 
    db_names = np.array(db_names)
    db_norms = np.linalg.norm(db_matrix, axis=1)

    print(f"✅ Matrice DB prête : {db_matrix.shape}")
    return db_matrix, db_names, db_norms

# ================= 4. PRÉDICTION (MAIN FUNCTION) =================
def vgg_predict(test_path, db_data, output_csv=None, model_name=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD):
    """
    Lance la prédiction sur un dossier de test entier.
    Args:
        test_path: Dossier des images à tester
        db_data: Tuple (db_matrix, db_names, db_norms) retourné par load_database
        output_csv: Chemin pour sauvegarder les résultats (optionnel)
    """
    print("\n--- 2. Lancement du Test Vectorisé ---")

    if not os.path.exists(test_path):
        print(f"❌ ERREUR : Dossier Test introuvable : {test_path}")
        return None

    if db_data is None:
        print("❌ ERREUR : Données DB invalides.")
        return None

    db_matrix, db_names, db_norms = db_data
    
    all_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total_files = len(all_files)

    print(f"Analyse de {total_files} images de test...")

    results = []
    batch_correct = 0
    start_time = time.time()

    for i, f in enumerate(all_files):
        img_path = os.path.join(test_path, f)
        true_label = clean_label(f)
        
        # --- A. Extraction ---
        test_emb = get_embedding(img_path, model_name)
        
        if test_emb is None: continue 
        
        # --- B. Comparaison Vectorielle ---
        test_emb = np.array(test_emb)
        test_norm = np.linalg.norm(test_emb)
        
        if test_norm == 0: continue

        dot_products = np.dot(db_matrix, test_emb)
        similarities = dot_products / (db_norms * test_norm)
        distances = 1 - similarities
        
        best_idx = np.argmin(distances)
        min_dist = distances[best_idx]
        best_match = db_names[best_idx]

        # --- C. Verdict ---
        is_correct = best_match == true_label
        
        if is_correct: 
            batch_correct += 1
            status = "OK"
        else:
            status = "MISS"

        # --- D. Stockage ---
        results.append({
            "filename": f,
            "true_label": true_label,
            "predicted": best_match if min_dist <= threshold else "unknown",
            "distance": round(min_dist, 4),
            "status": status,
            "correct": is_correct
        })

        # --- E. Logs ---
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            acc = batch_correct / (i+1)
            print(f"📊 {i+1}/{total_files} | Accuracy VGG: {acc:.2%}")
            
            if output_csv:
                pd.DataFrame(results).to_csv(output_csv, index=False)

    # ================= 5. RAPPORT FINAL =================
    final_df = pd.DataFrame(results)
    if output_csv:
        final_df.to_csv(output_csv, index=False)

    if len(final_df) > 0:
        acc_final = final_df['correct'].mean()
        elapsed_total = time.time() - start_time
        
        print("\n" + "="*40)
        print(f"🏁 TRAITEMENT TERMINÉ en {elapsed_total/60:.1f} minutes")
        print(f"🏆 ACCURACY FINALE (VGG) : {acc_final:.2%}")
        if output_csv:
            print(f"📁 Résultats sauvegardés dans : {output_csv}")
        print("="*40)
        return final_df
    else:
        print("\n⚠️ Aucun résultat généré.")
        return None

def main():
    configure_gpu()
    
    # 1. Charger la DB
    db_data = load_database(DEFAULT_DB_PATH)
    
    # 2. Lancer la prédiction
    if db_data:
        vgg_predict(DEFAULT_TEST_PATH, db_data, DEFAULT_OUTPUT_CSV)

if __name__ == "__main__":
    main()