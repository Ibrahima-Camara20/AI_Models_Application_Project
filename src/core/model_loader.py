"""
Chargement du modèle SVM depuis un fichier pickle.
"""
import joblib

def load_svm_model(model_path: str) -> dict:
    
    # Utilisation de joblib pour charger (compatible avec pickle aussi souvent)
    model_data = joblib.load(model_path)
    
    if not isinstance(model_data, dict):
         raise ValueError("Le fichier modèle doit contenir un dictionnaire.")

    # Compatibilité : Mapper les nouvelles clés vers les anciennes attendues par l'app
    if "model" in model_data and "encoder" in model_data:
        # Nouveau format (celui du user)
        model_data["svm_model"] = model_data["model"]
        model_data["label_encoder"] = model_data["encoder"]
    
    # Vérification des clés requises par app.py / predictor.py
    if "svm_model" not in model_data:
        raise ValueError("Le modèle chargé ne contient pas la clé 'svm_model' (ou 'model').")
    
    return model_data
