import os
import cv2
from ultralytics import YOLO

def extract_person_bounding_boxes(dataset_path, working_dir):
    """
    Étape 1.1 : Extrait les boîtes englobantes des humains via YOLO11n 
    et les sauvegarde dans le dossier 'working' avec le nommage imposé.
    """
    
    # Chargement du modèle imposé par le projet [cite: 19]
    model = YOLO('yolo11n.pt') 

    # Création du dossier de travail s'il n'existe pas [cite: 15]
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    print(f"Démarrage de l'extraction avec YOLO depuis : {dataset_path}...")

    # Parcours des dossiers de célébrités
    for celebrity_name in os.listdir(dataset_path):
        celeb_path = os.path.join(dataset_path, celebrity_name)
        
        if os.path.isdir(celeb_path):
            for img_name in os.listdir(celeb_path):
                img_path = os.path.join(celeb_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue

                # Détection de la classe 'person' (index 0) [cite: 18, 20]
                results = model.predict(img_path, classes=[0], verbose=False)
                
                for r in results:
                    for j, box in enumerate(r.boxes):
                        # Extraction des coordonnées en pixels [cite: 18]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Découpage de la zone humaine [cite: 15]
                        crop = img[y1:y2, x1:x2]
                        
                        # Format de nommage strict:
                        # original-filename-label-nn-bb-x1-y1-x2y2.jpg
                        base_name = os.path.splitext(img_name)[0]
                        final_name = f"{base_name}-person-{j:02d}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                        
                        # Sauvegarde dans le répertoire de travail [cite: 15]
                        cv2.imwrite(os.path.join(working_dir, final_name), crop)

    print(f"Terminé ! Les résultats sont dans le dossier '{working_dir}'.")

