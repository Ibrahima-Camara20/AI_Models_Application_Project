"""
Chargement du modèle SVM depuis un fichier pickle.
"""
import pickle


def load_svm_model(model_path: str) -> dict:
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    if not isinstance(model_data, dict) or "svm_model" not in model_data:
        raise ValueError("Le .pkl doit contenir un dict avec au moins la clé 'svm_model'.")
    
    return model_data
