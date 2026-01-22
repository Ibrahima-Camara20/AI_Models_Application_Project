import os
import random
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DS = os.path.join("datasets", "105_classes_pins_dataset")
TARGET_DIR = "data_test"
SAMPLES_PER_CLASS = 2

def create_random_test_set():
    print("--- CRÉATION DU JEU DE TEST ALÉATOIRE ---")
    
    if not os.path.exists(SOURCE_DS):
        print(f"❌ ERREUR : Dossier source introuvable : {SOURCE_DS}")
        return

    # Création (ou nettoyage) du dossier cible
    if os.path.exists(TARGET_DIR):
        print(f"⚠️ Le dossier '{TARGET_DIR}' existe déjà.")
        # On pourrait le vider, mais par sécurité on évite de supprimer sans demander.
        # shutil.rmtree(TARGET_DIR)
        # os.makedirs(TARGET_DIR)
    else:
        os.makedirs(TARGET_DIR)
        print(f"✅ Dossier créé : {TARGET_DIR}")

    # Lister les dossiers de célébrités
    celeb_dirs = [d for d in os.listdir(SOURCE_DS) if os.path.isdir(os.path.join(SOURCE_DS, d))]
    print(f"Trouvé {len(celeb_dirs)} célébrités.")

    total_copied = 0
    
    for celeb in tqdm(celeb_dirs, desc="Extraction"):
        celeb_path = os.path.join(SOURCE_DS, celeb)
        
        # Récupérer toutes les images
        images = [f for f in os.listdir(celeb_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Sélection aléatoire
        nb_to_take = min(len(images), SAMPLES_PER_CLASS)
        selected_images = random.sample(images, nb_to_take)
        
        # Nettoyage du nom de la célébrité pour le fichier de sortie
        # Ex: "pins_Adriana Lima" -> "Adriana Lima"
        clean_celeb_name = celeb.replace("pins_", "")
        
        for i, img_name in enumerate(selected_images):
            src_file = os.path.join(celeb_path, img_name)
            
            # Nommage à plat : Nom_Index.jpg (pour compatibilité vgg_prediction)
            # ou garder le nom original si besoin. 
            # Le user veut "mettre dans un dossier data_test". 
            # Pour vgg_prediction, il faut que le nom contienne l'identité.
            
            # On va créer un nom sans équivoque
            new_filename = f"{clean_celeb_name}_{i+1}_{img_name}"
            dst_file = os.path.join(TARGET_DIR, new_filename)
            
            shutil.copy2(src_file, dst_file)
            total_copied += 1

    print("\n" + "="*40)
    print(f"✅ TERMINÉ. {total_copied} images copiées dans '{TARGET_DIR}'.")
    print(f"Moyenne : {total_copied / len(celeb_dirs):.1f} images/célébrité")
    print("="*40)

if __name__ == "__main__":
    create_random_test_set()
