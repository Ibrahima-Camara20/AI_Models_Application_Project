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

        for item_name in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item_name)
            
            # Si c'est un dossier, on parcourt son contenu (structure datasets/nom/chemins)
            if os.path.isdir(item_path):
                for img_name in os.listdir(item_path):
                    img_path = os.path.join(item_path, img_name)
                    process_image(model, img_path, img_name, working_dir)

            # Si c'est une image directement dans le dossier racine
            elif item_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                process_image(model, item_path, item_name, working_dir)

        print(f"Terminé ! Extraction effectuée dans '{working_dir}'.")
            
    except Exception as e:
        print(f"[ERREUR CRITIQUE] Échec de l'extraction YOLO: {e}")
        raise

def process_image(model, img_path, img_name, working_dir):
    try:
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"[SKIP] Impossible de lire l'image : {img_name}")
            return

        # Détection de la classe 'person' (index 0)
        results = model.predict(img_path, classes=[0], verbose=False)
        
        for r in results:
            boxes = r.boxes
            num_persons = len(boxes)
            
            if num_persons == 1:
                # Une seule personne détectée : on garde l'image telle quelle mais on la renomme
                try:
                    box = boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    base_name = os.path.splitext(img_name)[0]
                    final_name = f"{base_name}-person-00-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                    
                    cv2.imwrite(os.path.join(working_dir, final_name), img)
                except Exception as e:
                     print(f"[ERREUR] Échec du renommage pour {img_name}: {e}")
                
            elif num_persons > 1:
                for j, box in enumerate(boxes):
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
                        
                    except Exception as e:
                        print(f"[ERREUR] Échec du crop pour {img_name} (box {j}): {e}")

    except Exception as e:
        print(f"[ERREUR] Échec de traitement pour {img_path}: {e}")

if __name__ == "__main__":
    extract_person_bounding_boxes("faces_dataset", "working")