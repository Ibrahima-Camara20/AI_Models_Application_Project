import os
import pandas as pd
from deepface import DeepFace

# ================= CONFIGURATION =================
# Dossier contenant les visages extraits à l'étape 1.2 (Inconnus)
INPUT_FACES_DIR = "../faces_dataset"

# Dossier contenant les images de référence (Base de connaissance)
# Vous devez mettre ici une photo de chaque star : ex: "brad_pitt.jpg"
DB_PATH = "../celebrity_db"

# Modèle imposé par le sujet (VGG du Visual Geometry Group)
MODEL_NAME = "VGG-Face"
# =================================================

def recognize_celebrities():
    # Vérification des dossiers
    if not os.path.exists(INPUT_FACES_DIR):
        print(f"[ERREUR] Le dossier {INPUT_FACES_DIR} est vide. Lancez l'étape 1.2.")
        return
    if not os.path.exists(DB_PATH):
        print(f"[ERREUR] Créez le dossier {DB_PATH} et mettez-y des photos de stars nommées.")
        return

    print(f"[INFO] Chargement du modèle {MODEL_NAME} et début de la reconnaissance...")

    # Liste des visages à identifier
    unknown_faces = [f for f in os.listdir(INPUT_FACES_DIR) if f.endswith(('.jpg', '.png'))]
    
    results = []

    for face_file in unknown_faces:
        face_path = os.path.join(INPUT_FACES_DIR, face_file)
        
        try:
            # DeepFace.find compare l'image 'face_path' avec toutes celles dans 'DB_PATH'
            # Il utilise VGG-Face pour extraire les traits
            dfs = DeepFace.find(
                img_path = face_path, 
                db_path = DB_PATH, 
                model_name = MODEL_NAME, 
                enforce_detection = False, # RetinaFace a déjà fait la détection
                silent = True
            )
            
            # DeepFace renvoie une liste de DataFrames. On prend le premier.
            if len(dfs) > 0 and not dfs[0].empty:
                df = dfs[0]
                # La première ligne est la correspondance la plus proche (distance la plus faible)
                best_match_path = df.iloc[0]['identity']
                
                # On extrait le nom de la star depuis le nom du fichier trouvé
                # Ex: "../celebrity_db/brad_pitt.jpg" -> "brad_pitt"
                recognized_name = os.path.basename(best_match_path).split('.')[0]
                
                print(f"[SUCCÈS] {face_file} est identifié comme : {recognized_name}")
                results.append((face_file, recognized_name))
            else:
                print(f"[INCONNU] Aucune correspondance trouvée pour {face_file}")
                results.append((face_file, "Unknown"))

        except Exception as e:
            print(f"[ERREUR] Problème sur {face_file} : {e}")

    # --- Sauvegarde des résultats pour le rapport ---
    # Création d'un tableau récapitulatif
    df_results = pd.DataFrame(results, columns=["Fichier_Source", "Prediction_VGG"])
    output_csv = "resultats_reconnaissance.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\n[INFO] Résultats sauvegardés dans {output_csv}")

if __name__ == "__main__":
    recognize_celebrities()