import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Pour sauvegarder le modèle

# --- CONFIGURATION ---
INPUT_CSV = "img/stats/vgg_final_results.csv"
MODEL_PATH = "img/mon_modele_svm_vgg.pkl"
ENCODER_PATH = "img/mon_encoder_labels_vgg.pkl"

def svm_prediction(input_csv=INPUT_CSV, model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
    print("chargement des données...")
    # 1. Chargement du CSV
    df = pd.read_csv(input_csv)

    # Vérification rapide
    print(f"Dataset chargé : {df.shape[0]} images, {df.shape[1]-1} caractéristiques par image.")

    # 2. Séparation Features (X) / Labels (y)
    # X = tout sauf la colonne label
    X = df.drop('label', axis=1).values
    # y = juste la colonne label
    y_str = df['label'].values

    # 3. Encodage des Labels (Texte -> Chiffres)
    # Le SVM ne comprend pas "Chris Evans", il veut "Classe 0"
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # On garde les noms en mémoire pour plus tard
    print(f"Classes détectées : {len(le.classes_)}")

    # 4. Création des jeux d'entraînement et de test
    # 80% pour apprendre, 20% pour tester
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Entraînement sur {len(X_train)} images, Test sur {len(X_test)} images.")

    # 5. Entraînement du SVM
    print("Entraînement du modèle en cours... (ça peut prendre quelques secondes)")
    # probability=True est important pour avoir le % de confiance plus tard
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # 6. Évaluation
    print("Évaluation du modèle...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n--- RÉSULTATS ---")
    print(f"Précision globale (Accuracy) : {accuracy * 100:.2f}%")
    print("-" * 30)

    # 7. Sauvegarde du modèle et de l'encodeur
    # On a besoin des deux pour faire des prédictions plus tard
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)

    print(f"Modèle sauvegardé sous : {model_path}")
    print(f"Encodeur sauvegardé sous : {encoder_path}")
    print("Tu es prêt pour la prédiction !")

