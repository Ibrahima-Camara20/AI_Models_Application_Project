import os
import cv2
from retinaface import RetinaFace

def extract_faces(input_dir, output_dir, confidence_threshold):
    try:
        # Création du dossier de sortie s'il n'existe pas
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"[INFO] Dossier créé : {output_dir}")
        except Exception as e:
            raise RuntimeError(f"Impossible de créer le dossier '{output_dir}': {e}")

        # Vérification que le dossier d'entrée existe
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Le dossier {input_dir} n'existe pas. Lancez l'étape 1.1 d'abord.")

        # Liste des images
        try:
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la lecture du dossier {input_dir}: {e}")
            
        print(f"[INFO] {len(files)} images trouvées dans {input_dir}")

        count = 0
        error_count = 0

        for filename in files:
            img_path = os.path.join(input_dir, filename)
            
            try:
                # 1. Détection des visages avec RetinaFace
                try:
                    faces = RetinaFace.detect_faces(img_path)
                except Exception as e:
                    print(f"[ERREUR] Impossible de traiter {filename}: {e}")
                    error_count += 1
                    continue

                # RetinaFace renvoie un dictionnaire ou un tuple vide si rien n'est trouvé
                if not isinstance(faces, dict):
                    print(f"[SKIP] Aucun visage détecté dans {filename}")
                    continue

                # 2. Traitement de chaque visage trouvé
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[ERREUR] Impossible de lire l'image {filename}")
                        error_count += 1
                        continue
                except Exception as e:
                    print(f"[ERREUR] Échec de lecture de {filename}: {e}")
                    error_count += 1
                    continue
                
                for key in faces:
                    try:
                        identity = faces[key]
                        score = identity["score"]
                        
                        # On ne garde que les détections fiables
                        if score < confidence_threshold:
                            continue
                        
                        # Récupération des coordonnées (x1, y1, x2, y2)
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
                        
                        if face_crop.size == 0:
                            print(f"[SKIP] Visage vide détecté dans {filename} (key: {key})")
                            continue
                            
                        save_name = f"face_{key}_{filename}"
                        save_path = os.path.join(output_dir, save_name)
                        
                        try:
                            cv2.imwrite(save_path, face_crop)
                            count += 1
                        except Exception as e:
                            print(f"[ERREUR] Échec de sauvegarde pour {save_name}: {e}")
                            error_count += 1
                            
                    except Exception as e:
                        print(f"[ERREUR] Échec du traitement du visage {key} dans {filename}: {e}")
                        error_count += 1
                        
            except Exception as e:
                print(f"[ERREUR] Erreur inattendue pour {filename}: {e}")
                error_count += 1

        print(f"--- Terminé ---")
        print(f"[INFO] {count} visages extraits et sauvegardés dans {output_dir}")
        if error_count > 0:
            print(f"[ATTENTION] {error_count} erreur(s) rencontrée(s) pendant le traitement.")
            
    except Exception as e:
        print(f"[ERREUR CRITIQUE] Échec de l'extraction des visages: {e}")
        raise

if __name__ == "__main__":
    extract_faces("../working", "../faces_dataset", 0.9)
