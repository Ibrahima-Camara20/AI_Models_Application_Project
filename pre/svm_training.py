import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib 

# --- CONFIGURATION KAGGLE---
"""
INPUT_CSV = "/kaggle/working/version-finale/vgg_extraction.csv"
MODEL_PATH = "/kaggle/working/version-finale/svm_vgg_face.pkl" # Un seul fichier pour tout
MODEL_TYPE = "VGGFace"
"""

def svm_prediction_bundled(input_csv=INPUT_CSV, model_path=MODEL_PATH, model_type=MODEL_TYPE):
    print("chargement des données...")
    df = pd.read_csv(input_csv)

    print(f"Dataset chargé : {df.shape[0]} images.")

    # Séparation X / y
    X = df.drop('label', axis=1).values
    y_str = df['label'].values

    # Encodage
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    
    print(f"Classes détectées : {len(le.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement
    print("Entraînement du SVM...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Évaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Précision globale : {accuracy * 100:.2f}%")

    # --- C'EST ICI QUE ÇA CHANGE ---
    # On met tout dans une "boîte" (un dictionnaire)
    package_to_save = {
        "model": model,
        "encoder": le,
        "accuracy": accuracy,  # On peut même sauvegarder le score pour info !
        "model_type": model_type
    }

    joblib.dump(package_to_save, model_path)
    
    print("-" * 30)
    print(f"Tout est sauvegardé dans : {model_path}")
    print("Le fichier contient : model, encoder, accuracy.")