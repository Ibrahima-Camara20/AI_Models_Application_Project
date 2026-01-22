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
from interface.core.predictor import get_embedding, predict_identity
from interface.core.path_utils import ensure_dir, clean_dir, get_temp_dirs
from interface.core.text_utils import list_images_in_dir


def run_pipeline_single(image_path: str, model_data: dict, backend: str, log_callback=None) -> dict:
    """
    Exécute le pipeline complet sur une seule image.
    
    Pipeline :
    1. YOLO détecte les personnes → working/
    2. RetinaFace extrait les visages → faces_extraction/
    3. Prédiction sur le visage extrait
    """
    def log(msg):
        """Helper pour logger les messages si un callback est fourni."""
        if log_callback:
            log_callback(msg)
    
    # ========================================================================
    # Préparation des dossiers temporaires
    # ========================================================================
    # Récupérer les chemins des dossiers temporaires
    temp_dirs = get_temp_dirs()
    working_dir = temp_dirs["working"]        # Pour les crops de personnes (YOLO)
    faces_dir = temp_dirs["faces_extraction"]  # Pour les visages extraits (RetinaFace)
    
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
        }
    }
    
    try:
        # ====================================================================
        # ÉTAPE 1 : DÉTECTION DES PERSONNES AVEC YOLO
        # ====================================================================
        log("[PIPELINE] Étape 1/3 : Détection YOLO...")
        result["stage"] = "yolo"
        
        # Détecter les personnes dans l'image
        # Les crops des personnes seront sauvegardés dans working/
        persons_detected = yolo_detection_single(image_path, working_dir)
        result["stats"]["persons_detected"] = persons_detected
        
        log(f"[PIPELINE] ✓ Personnes détectées : {persons_detected}")
        
        # Vérifier qu'au moins une personne a été détectée
        if persons_detected == 0:
            result["error"] = "Aucune personne détectée par YOLO"
            return result
        
        # ====================================================================
        # ÉTAPE 2 : EXTRACTION DES VISAGES AVEC RETINAFACE
        # ====================================================================
        log("[PIPELINE] Étape 2/3 : Extraction RetinaFace...")
        result["stage"] = "retinaface"
        
        # Récupérer toutes les images de personnes détectées par YOLO
        person_images = list_images_in_dir(working_dir)
        
        if not person_images:
            result["error"] = "Aucune image de personne trouvée dans working/"
            return result
        
        # Extraire les visages de toutes les personnes détectées
        # Pour chaque crop de personne, RetinaFace extrait le visage principal
        faces_detected = 0
        for person_img in person_images:
            faces_count = extract_faces_single(person_img, faces_dir)
            faces_detected += faces_count
        
        result["stats"]["faces_detected"] = faces_detected
        log(f"[PIPELINE] ✓ Visages détectés : {faces_detected}")
        
        # Vérifier qu'au moins un visage a été détecté
        if faces_detected == 0:
            result["error"] = "Aucun visage détecté par RetinaFace"
            return result
        
        # ====================================================================
        # ÉTAPE 3 : PRÉDICTION DE L'IDENTITÉ
        # ====================================================================
        log("[PIPELINE] Étape 3/3 : Prédiction...")
        result["stage"] = "prediction"
        
        # Récupérer les visages extraits par RetinaFace
        face_images = list_images_in_dir(faces_dir)
        
        if not face_images:
            result["error"] = "Aucune image de visage trouvée dans faces_extraction/"
            return result
        
        # Prédire sur le premier visage
        # Note: Si plusieurs visages sont détectés, on prédit sur le premier
        # (qui correspond généralement au plus gros visage détecté)
        face_path = face_images[0]
        
        # Extraire l'embedding du visage avec DeepFace
        embedding = get_embedding(face_path, backend)
        
        # Prédire l'identité avec le modèle SVM
        predicted_name, confidence = predict_identity(model_data, embedding)
        
        # ====================================================================
        # SUCCÈS : Enregistrer les résultats
        # ====================================================================
        result["success"] = True
        result["predicted_name"] = predicted_name
        result["confidence"] = confidence
        log(f"[PIPELINE] ✓ Prédiction terminée")
        
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
