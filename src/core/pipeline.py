"""
Module de pipeline pour la reconnaissance faciale en 3 étapes.

Ce module orchestre le processus complet de reconnaissance faciale :
1. YOLO : Détection des personnes dans l'image
2. RetinaFace : Extraction précise des visages
3. Prédiction : Reconnaissance faciale avec DeepFace + SVM

Le pipeline gère automatiquement :
- La création et le nettoyage des dossiers temporaires
- La gestion des erreurs à chaque étape
- Le logging en temps réel via callback
- La normalisation des noms de fichiers

"""

import os
import sys
import re
from pathlib import Path

# ============================================================================
# Configuration du path pour les imports
# ============================================================================
# Ajouter le dossier racine au path pour permettre l'import des modules
# de prétraitement (pre/) et de l'interface (interface/)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================================================
# Imports des modules de prétraitement
# ============================================================================
from pre.yolo_detection import yolo_detection_single
from pre.retinaface_extraction import extract_faces_single

# ============================================================================
# Imports des modules de prédiction et utilitaires
# ============================================================================
from src.core.predictor import get_embedding, predict_identity
from src.core.path_utils import ensure_dir, clean_dir, get_temp_dirs
from src.core.text_utils import list_images_in_dir


def run_pipeline_single(image_path: str, model_data: dict, backend: str, 
                       log_callback=None, status_callback=None, cleanup=True) -> dict:
    """
    Exécute le pipeline complet sur une seule image.
    
    Args:
        cleanup: Si True, vide les dossiers temporaires avant le traitement.
    """
    def log(msg):
        """Helper pour logger les messages si un callback est fourni."""
        if log_callback:
            log_callback(msg)
    
    def update_status(stage, status, count=None):
        """Helper pour mettre à jour le statut du pipeline."""
        if status_callback:
            status_callback(stage, status, count)
    
    # ========================================================================
    # Préparation des dossiers temporaires
    # ========================================================================
    # Récupérer les chemins des dossiers temporaires
    temp_dirs = get_temp_dirs()
    working_dir = temp_dirs["working"]        # Pour les crops de personnes (YOLO)
    faces_dir = temp_dirs["faces_extraction"]  # Pour les visages extraits (RetinaFace)
    
    if cleanup:
        # Nettoyer les dossiers pour éviter les conflits avec d'anciennes données
        clean_dir(working_dir)
        clean_dir(faces_dir)
    
    # Créer les dossiers s'ils n'existent pas
    ensure_dir(working_dir)
    ensure_dir(faces_dir)
    
    # ========================================================================
    # Initialisation du résultat
    # ========================================================================
    result = {
        "success": False,           # Indique si le pipeline s'est exécuté avec succès
        "stage": "init",            # Étape actuelle ou étape où l'erreur s'est produite
        "predicted_name": None,     # Nom prédit par le modèle
        "confidence": None,         # Confiance de la prédiction (0-1)
        "error": None,              # Message d'erreur si échec
        "stats": {
            "persons_detected": 0,  # Nombre de personnes détectées par YOLO
            "faces_detected": 0     # Nombre de visages détectés par RetinaFace
        },
        "boxes": {
            "yolo_boxes": [],       # Bounding boxes YOLO [(x1, y1, x2, y2, conf), ...]
            "retinaface_boxes": []  # Bounding boxes RetinaFace [(x1, y1, x2, y2), ...]
        },
        "predictions": []           # Liste des prédictions : [{"name":, "confidence":, "box":}, ...]
    }
    
    try:
        # ====================================================================
        # ÉTAPE 1 : DÉTECTION DES PERSONNES AVEC YOLO
        # ====================================================================
        log("[PIPELINE] Étape 1/3 : Détection YOLO...")
        result["stage"] = "yolo"
        update_status('yolo', 'running')
        
        # Détecter les personnes dans l'image avec récupération des boxes
        # Les crops des personnes seront sauvegardés dans working/
        persons_detected, yolo_boxes = yolo_detection_single(image_path, working_dir, return_boxes=True)
        result["stats"]["persons_detected"] = persons_detected
        result["boxes"]["yolo_boxes"] = yolo_boxes
        
        log(f"[PIPELINE] ✓ Personnes détectées : {persons_detected}")
        
        # Vérifier qu'au moins une personne a été détectée
        if persons_detected == 0:
            result["error"] = "Aucune personne détectée par YOLO"
            update_status('yolo', 'error')
            return result
        
        update_status('yolo', 'success', persons_detected)
        
        # ====================================================================
        # ÉTAPE 2 & 3 : EXTRACTION (RETINAFACE) + PRÉDICTION (SVM)
        # ====================================================================
        log("[PIPELINE] Étape 2 & 3 : Extraction & Prédiction...")
        result["stage"] = "extraction_prediction"
        update_status('retinaface', 'running')
        
        # Récupérer toutes les images de personnes (crops YOLO)
        person_images = list_images_in_dir(working_dir)
        
        if not person_images:
            result["error"] = "Aucune image de personne trouvée dans working/"
            update_status('retinaface', 'error')
            return result
            
        faces_detected = 0
        all_retinaface_boxes = []
        all_predictions = []
        
        # Regex pour parser les coordonnées du crop YOLO depuis le nom de fichier
        # Format attendu : ...-person-{j}-bb-{x1}-{y1}-{x2}-{y2}.jpg
        coord_pattern = re.compile(r"-bb-(\d+)-(\d+)-(\d+)-(\d+)")
        
        for person_img_path in person_images:
            filename = os.path.basename(person_img_path)
            
            # 1. Récupérer les coordonnées absolues du crop YOLO
            match = coord_pattern.search(filename)
            if not match:
                log(f"[WARN] Impossible de parser les coordonnées de {filename}")
                continue
                
            crop_x1, crop_y1, crop_x2, crop_y2 = map(int, match.groups())
            
            # 2. Extraire le visage dans ce crop (RetinaFace)
            # On demande extraction=True pour sauvegarder le visage dans faces_dir
            face_count, face_boxes = extract_faces_single(person_img_path, faces_dir, return_boxes=True)
            
            if face_count == 0:
                continue
                
            faces_detected += face_count
            
            # RetinaFace retourne les coords RELATIVES au crop YOLO.
            # On doit les convertir en ABSOLUES (crop_x1 + rel_x)
            
            # On suppose que extract_faces_single sauvegarde le fichier sous le même nom
            # Mais attention : si extract_faces_single sauve sous le nom d'entrée, on le retrouve ici.
            # IMPORTANT: La logique actuelle de extract_faces_single (si inchangée) écrase le fichier destination
            # avec le crop du visage. Le nom de fichier reste 'filename'.
            
            # Fichier visage extrait
            face_file_path = os.path.join(faces_dir, filename)
            
            if not os.path.exists(face_file_path):
                log(f"[WARN] Visage extrait introuvable : {face_file_path}")
                continue

            # Pour l'instant on prend le premier visage (le plus grand) du crop
            # Car extract_faces_single ne retourne que le "best face"
            rel_x1, rel_y1, rel_x2, rel_y2 = face_boxes[0]
            
            abs_x1 = crop_x1 + rel_x1
            abs_y1 = crop_y1 + rel_y1
            abs_x2 = crop_x1 + rel_x2
            abs_y2 = crop_y1 + rel_y2
            
            abs_box = (abs_x1, abs_y1, abs_x2, abs_y2)
            all_retinaface_boxes.append(abs_box)
            
            # 3. Prédiction immédiate sur ce visage
            try:
                embedding = get_embedding(face_file_path, backend)
                pred_name, conf = predict_identity(model_data, embedding)
                
                prediction = {
                    "name": pred_name,
                    "confidence": conf,
                    "box": abs_box
                }
                all_predictions.append(prediction)
                log(f"   -> Trouvé : {pred_name} ({conf:.2f})")
                
            except Exception as e:
                log(f"[WARN] Erreur prédiction sur {filename}: {e}")

        result["stats"]["faces_detected"] = faces_detected
        result["boxes"]["retinaface_boxes"] = all_retinaface_boxes
        result["predictions"] = all_predictions
        
        log(f"[PIPELINE] ✓ Visages traités : {faces_detected}")
        
        if faces_detected == 0:
            result["error"] = "Aucun visage détecté par RetinaFace"
            update_status('retinaface', 'error')
            # On ne fail pas forcément tout le pipeline, mais c'est un résultat vide
            return result
        
        update_status('retinaface', 'success', faces_detected)
        update_status('prediction', 'success')
        
        # Résultat global (on prend le meilleur score pour l'affichage principal si besoin)
        if all_predictions:
            # Trier par confiance décroissante
            all_predictions.sort(key=lambda x: x['confidence'] if x['confidence'] else 0, reverse=True)
            result["predicted_name"] = all_predictions[0]["name"]
            result["confidence"] = all_predictions[0]["confidence"]
        else:
            result["predicted_name"] = "Unknown"
            result["confidence"] = 0.0

        result["success"] = True
        log(f"[PIPELINE] ✓ Prédiction terminée ({len(all_predictions)} visages)")
        
    except FileNotFoundError as e:
        # Fichier d'image introuvable
        result["error"] = f"Fichier introuvable : {str(e)}"
    except Exception as e:
        # Toute autre erreur
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return result


def run_pipeline_batch(image_paths: list[str], model_data: dict, backend: str, 
                       progress_callback=None) -> list[dict]:
    """Exécute le pipeline sur un lot d'images."""
    results = []
    total = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        result = run_pipeline_single(img_path, model_data, backend)
        result["image_path"] = img_path
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, total, result)
    
    return results


def cleanup_pipeline_dirs():
    """
    Nettoie les dossiers temporaires du pipeline.
    """
    temp_dirs = get_temp_dirs()
    clean_dir(temp_dirs["working"])
    clean_dir(temp_dirs["faces_extraction"])
