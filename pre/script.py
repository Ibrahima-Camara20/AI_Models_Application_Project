import os
import shutil
import sys

# Import local modules
# Add parent directory to path to allow importing from sibling modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pre.yolo_detection import yolo_detector
from pre.retinaface_extraction import extract_faces
from pre.extract_embedding import embedding_extractor
from pre.svm_training import svm_prediction_bundled # This is actually training

def main():
    print("=== Pipeline de création de Dataset et Modèle ===")
    
    # 1. Configuration
    dataset_name = input("Entrez le nom du dataset (ex: celebrity_db) : ").strip()
    input_dir = f"img/{dataset_name}"
    
    if not os.path.exists(input_dir):
        print(f"[ERREUR] Le dossier {input_dir} n'existe pas.")
        print(f"Veuillez placer vos images dans img/{dataset_name}/")
        return

    # Chemins automatiques
    working_dir = f"img/{dataset_name}_yolo_crops"
    faces_dir = f"img/{dataset_name}_aligned"
    embeddings_file = f"models/{dataset_name}_embeddings.csv"
    model_output = f"models/svm_{dataset_name}.pkl"
    
    # Choix du modèle
    print("\nModèles disponibles pour l'embedding :")
    print("1. ArcFace (Recommandé)")
    print("2. Facenet")
    print("3. VGG-Face")
    
    choice = input("Votre choix (1-3) : ").strip()
    model_map = {
        "1": "ArcFace",
        "2": "Facenet",
        "3": "VGG-Face"
    }
    model_name = model_map.get(choice, "ArcFace")
    
    print("-" * 50)
    print(f"Configuration :")
    print(f"- Entrée : {input_dir}")
    print(f"- Crops YOLO : {working_dir}")
    print(f"- Visages alignés : {faces_dir}")
    print(f"- Embeddings : {embeddings_file}")
    print(f"- Modèle de sortie : {model_output}")
    print(f"- Backend : {model_name}")
    print("-" * 50)
    
    confirm = input("Lancer le traitement ? (o/n) : ").lower()
    if confirm != 'o':
        print("Annulé.")
        return

    try:
        # Étape 1 : YOLO
        print("\n[1/4] Détection des personnes (YOLO)...")
        yolo_detector(input_dir, working_dir)
        
        # Étape 2 : RetinaFace
        print("\n[2/4] Extraction et alignement des visages (RetinaFace)...")
        extract_faces(working_dir, faces_dir)
        
        # Étape 3 : Embedding
        print(f"\n[3/4] Génération des embeddings ({model_name})...")
        embedding_extractor(model_name, faces_dir, embeddings_file)
        
        # Étape 4 : Entraînement SVM
        print("\n[4/4] Entraînement du classifieur SVM...")
        if os.path.exists(embeddings_file):
            svm_prediction_bundled(
                input_csv=embeddings_file,
                model_path=model_output,
                model_type=model_name
            )
            print(f"\n[SUCCÈS] Modèle créé : {model_output}")
        else:
            print("[ERREUR] Le fichier d'embeddings n'a pas été créé.")

    except Exception as e:
        print(f"\n[ERREUR CRITIQUE] {e}")

if __name__ == "__main__":
    main()
