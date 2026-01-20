import os
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Chemin vers votre dossier de référence sur Drive
DB_PATH = "img/celibrity_db"

def check_reference_images():
    print(f"--- Vérification de la base de données : {DB_PATH} ---")
    
    valid_images = 0
    bad_images = 0
    
    for filename in os.listdir(DB_PATH):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        path = os.path.join(DB_PATH, filename)
        
        try:
            # On demande à DeepFace d'extraire le visage. 
            # S'il n'y arrive pas, il lèvera une erreur.
            embedding = DeepFace.represent(img_path=path, model_name="VGG-Face", enforce_detection=True)
            
            print(f"[OK] {filename} : Visage détecté et valide.")
            valid_images += 1
            
        except ValueError:
            print(f"[ERREUR] {filename} : Aucun visage clair détecté ! CHANGEZ CETTE IMAGE.")
            bad_images += 1
            
            # Afficher l'image problématique pour comprendre
            img = cv2.imread(path)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Image problématique : {filename}")
            plt.axis('off')
            plt.show()

    print("-" * 30)
    print(f"Total valide : {valid_images}")
    print(f"À changer   : {bad_images}")

if __name__ == "__main__":
    check_reference_images()