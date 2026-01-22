import os
import cv2
from ultralytics import YOLO
import torch
import unicodedata
import re



def normalize_filename(filename):
    """Normalise un nom de fichier en enlevant les accents et caractères spéciaux."""
    # Enlever l'extension
    name, ext = os.path.splitext(filename)
    
    # Enlever les accents
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ASCII', 'ignore').decode('ASCII')
    
    # Remplacer les caractères spéciaux par des underscores
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Enlever les underscores multiples
    name = re.sub(r'_+', '_', name)
    
    return name + ext

def get_all_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def yolo_detection_single(image_path, output_dir):
    """Détecte les personnes dans une seule image avec YOLO."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # Chargement du modèle
    model = YOLO("yolo11n.pt").to(device)
    
    # Prédiction
    results = model.predict(image_path, classes=[0], verbose=False, device=device)
    
    persons_detected = 0
    img_name = os.path.basename(image_path)
    
    for r in results:
        img = r.orig_img
        
        if r.boxes:
            for j, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop
                human_crop = img[y1:y2, x1:x2]
                
                # Nommage avec normalisation
                base_name = os.path.splitext(img_name)[0]
                new_filename = f"{base_name}-person-{j}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                new_filename = normalize_filename(new_filename)
                
                cv2.imwrite(os.path.join(output_dir, new_filename), human_crop)
                persons_detected += 1
    
    return persons_detected


def yolo_detector(input_dir, output_dir):
    """Détecte les personnes dans toutes les images d'un dossier avec YOLO."""
    if not os.path.exists(input_dir):
        print(f"Le dossier {input_dir} n'est pas correct")
        return 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du périphérique : {'GPU' if device == 0 else 'CPU'}")

    # Chargement du modèle sur le GPU (device=0)
    model = YOLO("yolo11n.pt").to(device)

    # Fonction récursive pour récupérer toutes les images
    all_images = get_all_images(input_dir)
    print(f"Total d'images trouvées : {len(all_images)}")

    if not all_images:
        print(f"[ATTENTION] Aucune image trouvée dans {input_dir}")
        return 0
    # Traitement par lots pour éviter de surcharger la mémoire
    BATCH_SIZE = 32
    total_extracted = 0

    for i in range(0, len(all_images), BATCH_SIZE):
        batch_paths = all_images[i : i + BATCH_SIZE]
        print(f"Traitement du lot {i//BATCH_SIZE + 1}/{(len(all_images) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_paths)} images)...")
        
        # On traite le lot
        try:
            results = model.predict(batch_paths, classes=[0], verbose=False, device=device)

            for r, original_path in zip(results, batch_paths):
                img_name = os.path.basename(original_path)
                img = r.orig_img 
                
                if r.boxes:
                    for j, box in enumerate(r.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Crop
                        human_crop = img[y1:y2, x1:x2]
                        
                        # Nommage avec normalisation
                        base_name = os.path.splitext(img_name)[0]
                        new_filename = f"{base_name}-person-{j}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                        new_filename = normalize_filename(new_filename)
                        
                        cv2.imwrite(os.path.join(output_dir, new_filename), human_crop)
                        total_extracted += 1
        except Exception as e:
            print(f"[ERREUR] Erreur lors du traitement du lot commençant à {batch_paths[0]}: {e}")

    print(f"TERMINÉ ! Total d'images : {total_extracted}")
    return total_extracted

if __name__ == "__main__":
    yolo_detector("datasets", "test")
