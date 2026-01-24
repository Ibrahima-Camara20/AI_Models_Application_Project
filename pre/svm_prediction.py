import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import joblib
import numpy as np
from deepface import DeepFace

# --- CONFIGURATION ---.



def identifier_visage(input_dir, model_path, model_name):
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Le dossier {input_dir} n'existe pas.")
    
    fichiers = os.listdir(input_dir)
    # On filtre pour ne garder que les images (évite les erreurs sur fichiers cachés)
    images = [f for f in fichiers if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        raise Exception(f"Aucune image trouvée dans le dossier {input_dir}.")
    try:
        boite = joblib.load(model_path)
        model = boite["model"]
        label_encoder = boite["encoder"]
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle : {e}")
        
    print(f"Traitement de {len(images)} images...\n")

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        
        try:
            # 1. Extraction
            objs = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                enforce_detection=False, # Mets False si tes images sont déjà rognées (crop)
                align=True
            )
            
            embedding = objs[0]["embedding"]
            vecteur = np.array([embedding])
            
            # 2. Prédiction
            probas = model.predict_proba(vecteur)[0]
            meilleur_index = np.argmax(probas)
            score_confiance = probas[meilleur_index]
            
            # 3. Résultat
            nom_predit = label_encoder.inverse_transform([meilleur_index])[0]
            
            print(f"Image : {img_name}")
            
            print(f"-> Résultat : {nom_predit.upper()} ({score_confiance*100:.1f}%)")
            
            print("-" * 20)
            
        except Exception as e:
            print(f"Erreur sur {img_name} : {e}")
            print("-" * 20)
