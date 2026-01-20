"""
Module de détection YOLO.
Contient la logique principale de détection de personnes avec YOLO.
"""

import os
import sys
import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import extract_celebrity_name, save_metadata
from yolo.box_validator import validate_bounding_box, validate_crop
from yolo.yolo_stats import (
    create_stats_dict, 
    update_detection_stats, 
    update_box_stats,
    finalize_stats
)


class YOLODetector:
    PERSON_CLASS_ID = 0
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif', '.gif')
    PROGRESS_INTERVAL = 500
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_IMG_SIZE = 320
    DEFAULT_CONFIDENCE = 0.25
    DEFAULT_IOU = 0.5
    
    def __init__(self, model_path='yolo11n.pt', batch_size=None, img_size=None):
        print(f"\n[INFO] Chargement du modèle YOLO: {model_path}...")
        self.model = YOLO(model_path)
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.img_size = img_size or self.DEFAULT_IMG_SIZE
        print(f"[INFO] Configuration: batch_size={self.batch_size}, img_size={self.img_size}")
    
    def detect_persons(self, img_path):
        try:
            results = self.model.predict(
                img_path, 
                classes=[self.PERSON_CLASS_ID], 
                verbose=False,
                imgsz=self.img_size,
                conf=self.DEFAULT_CONFIDENCE,
                iou=self.DEFAULT_IOU
            )
            
            if not results or len(results) == 0:
                return None, None
            
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return None, None
            
            person_boxes = [box for box in boxes if int(box.cls[0]) == self.PERSON_CLASS_ID]
            
            if len(person_boxes) == 0:
                return None, None
            
            confidences = [float(box.conf[0]) for box in person_boxes]
            return person_boxes, confidences
            
        except Exception as e:
            print(f"  [ERREUR] Détection YOLO : {e}")
            return None, None
    
    def detect_persons_batch(self, img_paths):
        try:
            results = self.model.predict(
                img_paths,
                classes=[self.PERSON_CLASS_ID],
                verbose=False,
                imgsz=self.img_size,
                conf=self.DEFAULT_CONFIDENCE,
                iou=self.DEFAULT_IOU,
                stream=False
            )
            
            batch_results = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    batch_results.append((None, None))
                    continue
                
                person_boxes = [box for box in result.boxes if int(box.cls[0]) == self.PERSON_CLASS_ID]
                
                if len(person_boxes) == 0:
                    batch_results.append((None, None))
                    continue
                
                confidences = [float(box.conf[0]) for box in person_boxes]
                batch_results.append((person_boxes, confidences))
            
            return batch_results
            
        except Exception as e:
            print(f"  [ERREUR] Détection YOLO batch : {e}")
            return [(None, None)] * len(img_paths)
    
    def _process_detection(self, box, confidence, img, img_name, detection_idx, 
                          celeb_name, output_dir, stats, metadata):
        """Traite une détection : validation, cropping, sauvegarde."""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        is_valid, x1, y1, x2, y2, w, h = validate_bounding_box(x1, y1, x2, y2, img, img_name)
        if not is_valid:
            return False
        
        update_box_stats(stats, confidence, w, h)
        
        crop = img[y1:y2, x1:x2]
        if not validate_crop(crop, img_name):
            return False
        
        base_name = os.path.splitext(img_name)[0]
        final_name = f"{base_name}-person-{detection_idx:02d}-bb-{x1}-{y1}-{x2}-{y2}.jpg"
        
        output_path = os.path.join(output_dir, final_name)
        cv2.imwrite(output_path, crop)
        
        metadata[final_name] = celeb_name
        stats['total_crops_saved'] += 1
        
        return True
    
    def _process_single_image(self, img_path, img_name, celeb_name, 
                             output_dir, stats, metadata):
        stats['total_images'] += 1
        
        if stats['total_images'] % self.PROGRESS_INTERVAL == 0:
            print(f"  → {stats['total_images']} images traitées...")
        
        person_boxes, confidences = self.detect_persons(img_path)
        num_persons = len(person_boxes) if person_boxes else 0
        update_detection_stats(stats, num_persons, confidences or [])
        
        if not person_boxes:
            return
        
        img = cv2.imread(img_path)
        if img is None:
            return
        
        for detection_idx, (box, confidence) in enumerate(zip(person_boxes, confidences)):
            self._process_detection(
                box, confidence, img, img_name, detection_idx,
                celeb_name, output_dir, stats, metadata
            )
    
    def process_dataset(self, dataset_path, output_dir, metadata_file="metadata.json"):
        """Traite un dataset complet avec YOLO."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        stats = create_stats_dict()
        metadata = {}
        
        self._print_header()
        
        celeb_folders = self._get_celebrity_folders(dataset_path)
        total_celebs = len(celeb_folders)
        
        for celeb_idx, celeb_folder in enumerate(celeb_folders):
            self._process_celebrity_folder(
                dataset_path, celeb_folder, celeb_idx, total_celebs,
                output_dir, stats, metadata
            )
        
        save_metadata(metadata, metadata_file)
        stats = finalize_stats(stats)
        
        return stats
    
    def _print_header(self):
        """Affiche l'en-tête du traitement."""
        print("\n" + "="*70)
        print("  ÉTAPE 1.1 : EXTRACTION DE BOUNDING BOXES AVEC YOLO")
        print("="*70)
        print()
    
    def _get_celebrity_folders(self, dataset_path):
        """Retourne tous les dossiers contenant des images (recherche récursive)."""
        celeb_folders = []
        
        for root, dirs, files in os.walk(dataset_path):
            has_images = any(f.lower().endswith(self.SUPPORTED_FORMATS) for f in files)
            if has_images:
                rel_path = os.path.relpath(root, dataset_path)
                celeb_folders.append(rel_path)
        
        return sorted(celeb_folders)
    
    def _process_celebrity_folder(self, dataset_path, celeb_folder, celeb_idx, 
                                  total_celebs, output_dir, stats, metadata):
        """Traite toutes les images d'un dossier avec batch processing."""
        celeb_path = os.path.join(dataset_path, celeb_folder)
        celeb_name = extract_celebrity_name(celeb_folder)
        
        print(f"[{celeb_idx+1}/{total_celebs}] Traitement : {celeb_name}")
        
        image_files = [
            f for f in os.listdir(celeb_path) 
            if f.lower().endswith(self.SUPPORTED_FORMATS)
        ]
        
        for batch_start in range(0, len(image_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            
            batch_paths = [os.path.join(celeb_path, img_name) for img_name in batch_files]
            batch_results = self.detect_persons_batch(batch_paths)
            
            for img_name, img_path, (person_boxes, confidences) in zip(batch_files, batch_paths, batch_results):
                stats['total_images'] += 1
                
                if stats['total_images'] % self.PROGRESS_INTERVAL == 0:
                    print(f"  → {stats['total_images']} images traitées...")
                
                num_persons = len(person_boxes) if person_boxes else 0
                update_detection_stats(stats, num_persons, confidences or [])
                
                if not person_boxes:
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                for detection_idx, (box, confidence) in enumerate(zip(person_boxes, confidences)):
                    self._process_detection(
                        box, confidence, img, img_name, detection_idx,
                        celeb_name, output_dir, stats, metadata
                    )
