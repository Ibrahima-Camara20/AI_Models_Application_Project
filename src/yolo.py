import os
import cv2
from ultralytics import YOLO

# Chargement du modèle imposé par le projet 
model = YOLO('yolo11n.pt') 

# Chemins (à adapter selon le nom réel de votre dossier dataset)
dataset_path = "dataset" 
working_dir = "working"

if not os.path.exists(working_dir):
    os.makedirs(working_dir) # Création du dossier imposé 

print("Démarrage de l'extraction avec YOLO...")

# Parcours des dossiers de célébrités
for celebrity_name in os.listdir(dataset_path):
    celeb_path = os.path.join(dataset_path, celebrity_name)
    
    if os.path.isdir(celeb_path):
        for img_name in os.listdir(celeb_path):
            img_path = os.path.join(celeb_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None: continue

            # Détection de la personne (classe 0 chez YOLO) [cite: 20]
            results = model.predict(img_path, classes=[0], verbose=False)
            
            for i, r in enumerate(results):
                for j, box in enumerate(r.boxes):
                    # Coordonnées réelles en pixels [cite: 18]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Découpage de la zone humaine 
                    crop = img[y1:y2, x1:x2]
                    
                    # Nommage strict : original-filename-label-nn-bb-x1-y1-x2y2.jpg 
                    base_name = os.path.splitext(img_name)[0]
                    # On utilise 'person' comme label car c'est ce que YOLO reconnaît [cite: 18]
                    final_name = f"{base_name}-person-{j:02d}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                    
                    cv2.imwrite(os.path.join(working_dir, final_name), crop)

print(f"Terminé ! Vérifiez le dossier '{working_dir}' pour voir les résultats.")