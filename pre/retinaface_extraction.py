import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
from retinaface import RetinaFace
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        raise Exception(f"Configuration GPU : {e}")
        

def extract_faces_single(image_path, output_dir="working/", return_boxes=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image {image_path}")

    try:
        obj = RetinaFace.detect_faces(image_path)
    except Exception as e:
        raise Exception(f"Erreur RetinaFace sur {filename}: {e}")
    
    boxes_list = []
    
    if isinstance(obj, dict) and len(obj) > 0:
        best_face_coords = None
        max_area = 0
        
        for key, identity in obj.items():
            x1, y1, x2, y2 = identity["facial_area"]
            area = (x2 - x1) * (y2 - y1)
            if return_boxes:
                boxes_list.append((x1, y1, x2, y2))    
            if area > max_area:
                max_area = area
                best_face_coords = [x1, y1, x2, y2]
        
        if best_face_coords is not None:
            x1, y1, x2, y2 = best_face_coords
            h, w, _ = img.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            face_crop = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(output_dir, filename), face_crop)
            if return_boxes:
                return 1, boxes_list
            return 1
    
    if return_boxes:
        return 0, boxes_list
    return 0


def extract_faces(input_dir="working/", output_dir="faces_extraction/"):
    if not os.path.exists(input_dir):
        return {"success": 0, "no_face": 0}
        raise FileNotFoundError(f"Le dossier {input_dir} n'existe pas ou est mal écrit")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    
    def get_all_images_recursive(directory):
        image_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    files = get_all_images_recursive(input_dir)
    total_files = len(files)

    
    if total_files == 0:
        raise Exception(f"[ATTENTION] Aucune image trouvée dans {input_dir}")
    
    print(f"--- DÉMARRAGE RETINAFACE SUR {total_files} IMAGES ---")

    count_success = 0
    count_no_face = 0
    filenames = []

    for i, img_path in enumerate(files):
        parent_folder = os.path.basename(os.path.dirname(img_path))
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        img = cv2.imread(img_path)
        if img is None: continue

        try:
            # On utilise detect_faces car extract_faces n'existe pas dans cette version
            obj = RetinaFace.detect_faces(img_path)
        except Exception as e:
            print(f"[ERREUR] Sur {filename}: {e}")
            continue

        # Si des visages sont trouvés
        if isinstance(obj, dict) and len(obj) > 0:
            
            best_face_coords = None
            max_area = 0

            # On parcourt tous les candidats pour trouver le PLUS GROS visage (la star)
            for key, identity in obj.items():
                x1, y1, x2, y2 = identity["facial_area"]
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    best_face_coords = [x1, y1, x2, y2]

            # Sauvegarde UNIQUEMENT si on a trouvé un "meilleur visage"
            if best_face_coords is not None:
                x1, y1, x2, y2 = best_face_coords
                h, w, _ = img.shape

                # Correction des bornes
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2) 
                y2 = min(h, y2)

                # Crop manuel
                face_crop = img[y1:y2, x1:x2]
                
                # Nouveau nom de fichier unique
                
                cv2.imwrite(os.path.join(output_dir, filename), face_crop)
                count_success += 1
            else:
                filenames.append(filename)
                count_no_face += 1
        else:
            filenames.append(filename)
            count_no_face += 1

        # Log
        if (i + 1) % 10 == 0:  # Log plus fréquent car RetinaFace est lent
            print(f"Progression : {i + 1}/{total_files} - Visages extraits : {count_success}")

    print("-" * 30)
    print(f"TRAITEMENT TERMINÉ.")
    print(f"Images en entrée : {total_files}")
    print(f"Visages extraits : {count_success}")
    print(f"Pertes : {count_no_face}")

    """
    for filename in filenames:
        print(filename)
    """
    
    return {"success": count_success, "no_face": count_no_face}
if __name__ == "__main__":
    extract_faces("img/celebrity_db", "img/celebrity_db_cropped")