"""
Module de détection RetinaFace optimisé.
Extrait les visages à partir des crops de personnes (dossier working).
"""

import os
import sys
import cv2
from retinaface import RetinaFace as RF_Model
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import load_metadata, save_metadata


class RetinaFaceDetector:
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif', '.gif')
    PROGRESS_INTERVAL = 100
    
    MAX_IMAGE_SIZE = 640
    NUM_WORKERS = 4
    
    def __init__(self, max_size=None, num_workers=None):
        self.max_size = max_size or self.MAX_IMAGE_SIZE
        self.num_workers = num_workers or self.NUM_WORKERS
        print(f"\n[INFO] RetinaFace prêt (max_size={self.max_size}, workers={self.num_workers})")
    
    def _resize_if_needed(self, img):
        h, w = img.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.max_size:
            scale = self.max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            return resized, scale
        return img, 1.0
    
    def detect_faces(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None, None, None
            
            resized_img, scale = self._resize_if_needed(img)
            faces = RF_Model.detect_faces(resized_img)
            
            if not faces or len(faces) == 0:
                return None, None, None
            
            return faces, scale, img
            
        except Exception as e:
            print(f"  [ERREUR] Détection RetinaFace : {e}")
            return None, None, None
    
    def _process_detection(self, face_data, img, img_name, detection_idx,
                          celeb_name, output_dir, scale):
        facial_area = face_data['facial_area']
        x1, y1, x2, y2 = facial_area
        
        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        if crop_width < 10 or crop_height < 10:
            return None
        
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        base_name = os.path.splitext(img_name)[0]
        final_name = f"{base_name}.jpg"
        
        output_path = os.path.join(output_dir, final_name)
        cv2.imwrite(output_path, crop)
        
        return (final_name, celeb_name)
    
    def _process_single_image(self, img_path, img_name, celeb_name, output_dir):
        results = []
        
        faces, scale, img = self.detect_faces(img_path)
        
        if not faces or img is None:
            return results
        
        for detection_idx, (face_key, face_data) in enumerate(faces.items()):
            result = self._process_detection(
                face_data, img, img_name, detection_idx,
                celeb_name, output_dir, scale
            )
            if result:
                results.append(result)
        
        return results
    
    def process_working_folder(self, working_dir, output_dir, 
                               input_metadata="metadata.json", 
                               output_metadata="faces_metadata.json"):
        """
        Traite les crops de personnes dans working/ et extrait les visages.
        Utilise metadata.json pour récupérer le label de chaque image.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Charger les métadonnées des crops YOLO
        person_metadata = load_metadata(input_metadata)
        
        stats = {'total_images': 0, 'total_faces_saved': 0, 'no_face_detected': 0}
        face_metadata = {}
        
        print("\n" + "="*70)
        print("  EXTRACTION DE VISAGES AVEC RETINAFACE")
        print("="*70)
        print(f"\n Entrée  : {working_dir}/")
        print(f" Sortie  : {output_dir}/")
        print()
        
        # Récupérer toutes les images "person" dans working
        image_files = [
            f for f in os.listdir(working_dir)
            if f.lower().endswith(self.SUPPORTED_FORMATS) and 'person' in f.lower()
        ]
        
        total_images = len(image_files)
        print(f"[INFO] {total_images} crops de personnes trouvés")
        
        # Traitement parallèle
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for img_name in image_files:
                img_path = os.path.join(working_dir, img_name)
                celeb_name = person_metadata.get(img_name, "Unknown")
                
                future = executor.submit(
                    self._process_single_image,
                    img_path, img_name, celeb_name, output_dir
                )
                futures[future] = img_name
            
            for future in as_completed(futures):
                stats['total_images'] += 1
                
                if stats['total_images'] % self.PROGRESS_INTERVAL == 0:
                    print(f"  → {stats['total_images']}/{total_images} images traitées...")
                
                results = future.result()
                if results:
                    for final_name, name in results:
                        face_metadata[final_name] = name
                        stats['total_faces_saved'] += 1
                else:
                    stats['no_face_detected'] += 1
        
        save_metadata(face_metadata, output_metadata)
        
        print("\n" + "="*70)
        print("  RÉSUMÉ - RETINAFACE")
        print("="*70)
        print(f"  • Crops de personnes traités : {stats['total_images']}")
        print(f"  • Visages extraits           : {stats['total_faces_saved']}")
        print(f"  • Sans visage détecté        : {stats['no_face_detected']}")
        print("="*70)
        print()
        
        return stats
