import os
import shutil
import sys
from pre.yolo_detection import yolo_detector
from pre.retinaface_extraction import extract_faces
from pre.svm_prediction import identifier_visage

def clean_temp_dir(directory):
    """Supprime un dossier temporaire et son contenu."""
    if os.path.exists(directory):
        shutil.rmtree(directory)

def main():
    print("=== AI Models Application - CLI Mode ===")
    
    # 1. Demander le fichier ou dossier d'entrée
    while True:
        input_path = input("Entrez le chemin du fichier image ou du dossier d'images : ").strip()
        # Enlever les guillemets si l'utilisateur a fait un drag & drop
        if input_path.startswith('"') and input_path.endswith('"'):
            input_path = input_path[1:-1]
            
        if os.path.exists(input_path):
            break
        print(f"[ERREUR] Le chemin '{input_path}' n'existe pas. Veuillez réessayer.")

    # 2. Demander le modèle
    models = {
        "1": ("ArcFace", "models/svm_arcface.pkl"),
        "2": ("Facenet", "models/svm_facenet_512.pkl"),
        "3": ("VGG-Face", "models/svm_vgg_face.pkl")
    }
    
    print("\nChoisissez un modèle :")
    print("1. ArcFace (Recommandé - Meilleure précision)")
    print("2. Facenet")
    print("3. VGG-Face")
    
    while True:
        choice = input("Votre choix (1/2/3) : ").strip()
        if choice in models:
            model_name, model_path = models[choice]
            break
        print("Choix invalide. Veuillez entrer 1, 2 ou 3.")

    # Configuration des dossiers
    working_dir = "working/"
    faces_dir = "faces_extraction/"
    
    # Gestion du cas fichier unique vs dossier
    is_single_file = os.path.isfile(input_path)
    temp_input_dir = "temp_cli_input/"
    
    processing_input_dir = input_path
    
    try:
        # Si c'est un fichier unique, on le copie dans un dossier temporaire
        if is_single_file:
            print(f"\n[INFO] Mode fichier unique détecté : {input_path}")
            if os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir)
            os.makedirs(temp_input_dir)
            shutil.copy(input_path, temp_input_dir)
            processing_input_dir = temp_input_dir
        else:
            print(f"\n[INFO] Mode dossier détecté : {input_path}")

        print(f"[INFO] Modèle sélectionné : {model_name}")
        print("-" * 30)

        # 3. Exécution du pipeline
        print("\n[ETAPE 1/3] Détection YOLO...")
        yolo_detector(input_dir=processing_input_dir, output_dir=working_dir)
        
        print("\n[ETAPE 2/3] Extraction RetinaFace...")
        extract_faces(input_dir=working_dir, output_dir=faces_dir)
        
        print("\n[ETAPE 3/3] Identification SVM...")
        identifier_visage(input_dir=faces_dir, model_path=model_path, model_name=model_name)
        
    except Exception as e:
        print(f"\n[ERREUR CRITIQUE] Une erreur est survenue : {e}")
        
    finally:
        # Nettoyage
        if is_single_file:
            clean_temp_dir(temp_input_dir)
            
    print("\n[FIN] Programme terminé.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ARRET] Programme interrompu par l'utilisateur.")
        sys.exit(0)