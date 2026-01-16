import os
import cv2
from retinaface import RetinaFace

# ================= CONFIGURATION =================
# Dossier contenant les images découpées par YOLO (étape 1.1)
INPUT_DIR = "../working" 
# Dossier où sauvegarder les visages extraits (pour l'étape 1.3)
OUTPUT_DIR = "../faces_dataset" 
# Seuil de confiance pour accepter un visage (0.9 est standard pour être sûr)
CONFIDENCE_THRESHOLD = 0.9
# =================================================

def extract_faces():
    # Création du dossier de sortie s'il n'existe pas
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Dossier créé : {OUTPUT_DIR}")

    # Vérification que le dossier d'entrée existe
    if not os.path.exists(INPUT_DIR):
        print(f"[ERREUR] Le dossier {INPUT_DIR} n'existe pas. Lancez l'étape 1.1 d'abord.")
        return

    # Liste des images
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[INFO] {len(files)} images trouvées dans {INPUT_DIR}")

    count = 0

    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        
        # 1. Détection des visages avec RetinaFace
        # On passe directement le chemin de l'image
        try:
            faces = RetinaFace.detect_faces(img_path)
        except Exception as e:
            print(f"[ERREUR] Impossible de traiter {filename}: {e}")
            continue

        # RetinaFace renvoie un dictionnaire ou un tuple vide si rien n'est trouvé
        if not isinstance(faces, dict):
            print(f"[SKIP] Aucun visage détecté dans {filename}")
            continue

        # 2. Traitement de chaque visage trouvé
        img = cv2.imread(img_path)
        
        for key in faces:
            identity = faces[key]
            score = identity["score"]
            
            # On ne garde que les détections fiables
            if score < CONFIDENCE_THRESHOLD:
                continue
            
            # Récupération des coordonnées (x1, y1, x2, y2)
            # facial_area donne [x1, y1, x2, y2]
            facial_area = identity["facial_area"]
            x1, y1, x2, y2 = facial_area

            # Sécurité pour ne pas sortir de l'image (clamping)
            h_img, w_img, _ = img.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            # 3. Extraction (Crop)
            face_crop = img[y1:y2, x1:x2]

            # 4. Sauvegarde
            # On conserve le nom d'origine pour garder la trace de l'identité
            # Format: face_SCORE_NomOriginal.jpg
            save_name = f"face_{key}_{filename}"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            cv2.imwrite(save_path, face_crop)
            count += 1

    print(f"--- Terminé ---")
    print(f"[INFO] {count} visages extraits et sauvegardés dans {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_faces()