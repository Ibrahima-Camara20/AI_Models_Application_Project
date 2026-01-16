import os
import cv2
from ultralytics import YOLO

def extract_person_bounding_boxes(dataset_path, working_dir):
    try:
        # Vérification du dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Le dossier dataset '{dataset_path}' n'existe pas.")
        
        # Chargement du modèle YOLO
        try:
            model = YOLO('yolo11n.pt')
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle YOLO: {e}")
        
        # Création du répertoire de travail
        try:
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
        except Exception as e:
            raise RuntimeError(f"Impossible de créer le dossier '{working_dir}': {e}")

        print(f"Démarrage de l'extraction avec YOLO depuis : {dataset_path}...")
        
        processed_count = 0
        error_count = 0

        for celebrity_name in os.listdir(dataset_path):
            celeb_path = os.path.join(dataset_path, celebrity_name)
            
            if os.path.isdir(celeb_path):
                for img_name in os.listdir(celeb_path):
                    img_path = os.path.join(celeb_path, img_name)
                    
                    try:
                        img = cv2.imread(img_path)
                        
                        if img is None:
                            print(f"[SKIP] Impossible de lire l'image : {img_name}")
                            continue

                        # Détection de la classe 'person' (index 0)
                        results = model.predict(img_path, classes=[0], verbose=False)
                        
                        for r in results:
                            for j, box in enumerate(r.boxes):
                                try:
                                    # Extraction des coordonnées en pixels
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    
                                    # Découpage de la zone humaine
                                    crop = img[y1:y2, x1:x2]
                                    
                                    # Format de nommage strict
                                    base_name = os.path.splitext(img_name)[0]
                                    final_name = f"{base_name}-person-{j:02d}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                                    
                                    # Sauvegarde dans le répertoire de travail
                                    cv2.imwrite(os.path.join(working_dir, final_name), crop)
                                    processed_count += 1
                                    
                                except Exception as e:
                                    print(f"[ERREUR] Échec du crop pour {img_name} (box {j}): {e}")
                                    error_count += 1
                    
                    except Exception as e:
                        print(f"[ERREUR] Échec de traitement pour {img_path}: {e}")
                        error_count += 1

        print(f"Terminé ! {processed_count} personnes extraites dans '{working_dir}'.")
        if error_count > 0:
            print(f"[ATTENTION] {error_count} erreur(s) rencontrée(s) pendant le traitement.")
            
    except Exception as e:
        print(f"[ERREUR CRITIQUE] Échec de l'extraction YOLO: {e}")
        raise

