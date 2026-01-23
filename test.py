import os
import pickle
import numpy as np
from deepface import DeepFace

# Chemins
MODEL_PATH = "svm_arc_face.pkl"
FACES_DIR = "faces_dataset"

def load_svm_model(model_path):
    """Charge le modèle SVM depuis un fichier pickle."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

def get_arcface_embedding(image_path):
    """Extrait l'embedding ArcFace d'une image."""
    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="ArcFace",
        enforce_detection=False
    )
    return np.array(embedding[0]["embedding"])

def predict_identity(model_data, embedding):
    """Prédit l'identité à partir de l'embedding."""
    svm_model = model_data["svm_model"]
    label_encoder = model_data.get("label_encoder", None)
    
    # Reshape pour le SVM
    embedding = embedding.reshape(1, -1)
    
    # Prédiction
    prediction = svm_model.predict(embedding)
    
    # Probabilités si disponibles
    if hasattr(svm_model, "predict_proba"):
        probabilities = svm_model.predict_proba(embedding)
        confidence = np.max(probabilities)
    else:
        confidence = None
    
    # Décodage du label si encoder disponible
    if label_encoder is not None:
        predicted_name = label_encoder.inverse_transform(prediction)[0]
    else:
        predicted_name = prediction[0]
    
    return predicted_name, confidence

def main():
    print("=" * 60)
    print("TEST DU MODÈLE SVM + ARCFACE")
    print("=" * 60)
    
    # Charger le modèle SVM
    print(f"\n[INFO] Chargement du modèle: {MODEL_PATH}")
    model_data = load_svm_model(MODEL_PATH)
    print("[INFO] Modèle chargé avec succès!")
    
    # Afficher les infos du modèle
    if isinstance(model_data, dict):
        print(f"[INFO] Clés du modèle: {list(model_data.keys())}")
    
    
    print("-" * 60)
    
    for img_name in os.listdir(FACES_DIR):
        img_path = os.path.join(FACES_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"[ERREUR] {img_path} n'existe pas!")
            continue
        
        print(f"\n[TEST] Traitement de: {img_name}")
        
        try:
            # Extraire l'embedding ArcFace
            print("  -> Extraction de l'embedding ArcFace...")
            embedding = get_arcface_embedding(img_path)
            print(f"  -> Embedding shape: {embedding.shape}")
            
            # Prédiction avec SVM
            print("  -> Prédiction avec le SVM...")
            identity, confidence = predict_identity(model_data, embedding)
            
            if confidence is not None:
                print(f"  => RÉSULTAT: {identity} (confiance: {confidence:.2%})")
            else:
                print(f"  => RÉSULTAT: {identity}")
                
        except Exception as e:
            print(f"  [ERREUR] {e}")
    
    print("\n" + "=" * 60)
    print("TEST TERMINÉ")
    print("=" * 60)

if __name__ == "__main__":
    main()
