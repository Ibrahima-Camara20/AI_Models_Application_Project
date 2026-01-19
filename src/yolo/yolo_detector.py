"""
Module de détection YOLO.
Contient la logique principale de détection de personnes avec YOLO.
"""

import os
import cv2
from ultralytics import YOLO
from utils import extract_celebrity_name, save_metadata
from box_validator import validate_bounding_box, validate_crop, is_abnormal_box
from yolo_stats import (
    create_stats_dict, 
    update_detection_stats, 
    update_box_stats,
    finalize_stats
)


class YOLODetector:
    """
    Classe pour gérer la détection de personnes avec YOLO.
    """
    
    def __init__(self, model_path='yolo11n.pt'):
        """
        Initialise le détecteur YOLO.

        """
        print(f"\n[INFO] Chargement du modèle YOLO: {model_path}...")
        self.model = YOLO(model_path)
    
    def detect_persons(self, img_path):
        """
        Détecte les personnes dans une image.
       
        """
        try:
            # Prédiction YOLO (classe 0 = person)
            results = self.model.predict(img_path, classes=[0], verbose=False)
            
            if not results or len(results) == 0:
                return None, None
            
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return None, None
            
            # Filtrer uniquement les personnes (classe 0)
            person_boxes = [box for box in boxes if int(box.cls[0]) == 0]
            
            if len(person_boxes) == 0:
                return None, None
            
            # Extraire les confidences
            confidences = [float(box.conf[0]) for box in person_boxes]
            
            return person_boxes, confidences
            
        except Exception as e:
            print(f"  [ERREUR] Détection YOLO : {e}")
            return None, None
    
    def process_dataset(self, dataset_path, output_dir, metadata_file="metadata.json"):
        """
        Traite un dataset complet avec YOLO.
       
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        stats = create_stats_dict()
        metadata = {}
        
        print("\n" + "="*70)
        print("  ÉTAPE 1.1 : EXTRACTION DE BOUNDING BOXES AVEC YOLO")
        print("="*70)
        print()
        
        # Parcourir chaque dossier de célébrité
        celeb_folders = sorted([f for f in os.listdir(dataset_path) 
                               if os.path.isdir(os.path.join(dataset_path, f))])
        
        total_celebs = len(celeb_folders)
        
        for celeb_idx, celeb_folder in enumerate(celeb_folders):
            celeb_path = os.path.join(dataset_path, celeb_folder)
            celeb_name = extract_celebrity_name(celeb_folder)
            
            print(f"[{celeb_idx+1}/{total_celebs}] Traitement : {celeb_name}")
            
            image_files = [f for f in os.listdir(celeb_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_files:
                img_path = os.path.join(celeb_path, img_name)
                stats['total_images'] += 1
                
                # Afficher progression
                if stats['total_images'] % 500 == 0:
                    print(f"  → {stats['total_images']} images traitées...")
                
                # Détection YOLO
                person_boxes, confidences = self.detect_persons(img_path)
                
                # Mettre à jour stats de détection
                num_persons = len(person_boxes) if person_boxes else 0
                update_detection_stats(stats, num_persons, confidences or [])
                
                if not person_boxes:
                    continue
                
                # Charger l'image pour cropping
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Traiter chaque détection
                for j, (box, confidence) in enumerate(zip(person_boxes, confidences)):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Valider la bounding box
                    is_valid, x1, y1, x2, y2, w, h = validate_bounding_box(
                        x1, y1, x2, y2, img, img_name
                    )
                    
                    if not is_valid:
                        continue
                    
                    # Mettre à jour les stats de box
                    update_box_stats(stats, confidence, w, h)
                    
                    # Cropper l'image
                    crop = img[y1:y2, x1:x2]
                    
                    # Valider le crop
                    if not validate_crop(crop, img_name):
                        continue
                    
                    # Sauvegarder
                    base_name = os.path.splitext(img_name)[0]
                    final_name = f"{base_name}-person-{j:02d}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
                    
                    output_path = os.path.join(output_dir, final_name)
                    cv2.imwrite(output_path, crop)
                    
                    metadata[final_name] = celeb_name
                    stats['total_crops_saved'] += 1
        
        # Sauvegarder métadonnées et finaliser stats
        save_metadata(metadata, metadata_file)
        stats = finalize_stats(stats)
        
        return stats
