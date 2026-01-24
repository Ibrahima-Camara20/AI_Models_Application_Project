import os
import re
import pandas as pd
from deepface import DeepFace

# --- CONFIGURATION ---
data_path = "/kaggle/working/version-finale/faces_dataset" 
output_csv = "/kaggle/working/version-finale/arcface_extraction.csv" # Nom de fichier mis à jour

def clean_label(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace("face_", "").replace("pins_", "")
    if "-person" in name: 
        name = name.split("-person")[0]
    name = re.sub(r'\d+', '', name)
    name = name.replace("_", " ").replace("-", " ")
    return re.sub(r'\s+', ' ', name).strip().lower()

def embedding_extractor(model_name, data_path=data_path, output_csv=output_csv):
    features_list = []
    labels_list = []
    
    print(f"Analyse du dossier : {data_path}")
    
    if not os.path.exists(data_path):
        print("ERREUR : Le dossier n'existe pas !")
        return

    all_files = os.listdir(data_path)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    
    print(f"Début de l'extraction ArcFace sur {total_images} images...")
    
    for i, img_filename in enumerate(image_files):
        img_path = os.path.join(data_path, img_filename)
        
        try:
            # --- LE SEUL CHANGEMENT EST ICI ---
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name= model_name,  # On appelle ArcFace
                enforce_detection=False,
                align=True
            )
            
            embedding = embedding_objs[0]["embedding"]
            
            label_name = clean_label(img_filename)
            features_list.append(embedding)
            labels_list.append(label_name)
            
            if i % 1000 == 0:
                print(f"Traité {i}/{total_images}")

        except Exception as e:
            print(f"\nErreur sur {img_filename}: {e}")
    
    # --- SAUVEGARDE ---
    print("\nCréation du fichier CSV...")
    
    if len(features_list) > 0:
        df = pd.DataFrame(features_list)
        df['label'] = labels_list
        
        df.to_csv(output_csv, index=False)
        print(f"Terminé ! Sauvegardé dans {output_csv}")
        print(f"Dimensions : {df.shape}") 
        # Tu verras ici que df.shape sera (nombre_images, 513) -> 512 features + 1 label
    else:
        print("Aucune donnée extraite.")

# Lancement
if __name__ == "__main__":
    arcface_extraction("ArcFace", data_path=data_path, output_csv=output_csv)