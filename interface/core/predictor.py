"""
Extraction d'embeddings et prédiction d'identité avec DeepFace + SVM.
"""
import numpy as np
from deepface import DeepFace


def get_embedding(image_path: str, model_name: str) -> np.ndarray:
    """Extrait l'embedding d'une image avec DeepFace."""
    reps = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,       # "ArcFace" ou "VGG11"
        detector_backend="opencv",
        enforce_detection=False
    )
    return np.array(reps[0]["embedding"], dtype=np.float32)


def predict_identity(model_data: dict, embedding: np.ndarray) -> tuple[str, float | None]:
    """Prédit l'identité à partir d'un embedding avec un modèle SVM."""
    svm_model = model_data["svm_model"]
    label_encoder = model_data.get("label_encoder", None)
    normalizer = model_data.get("normalizer", None)

    x = embedding.reshape(1, -1)

    # Normaliser si un normaliseur est disponible
    if normalizer is not None:
        x = normalizer.transform(x)

    # Prédiction
    pred = svm_model.predict(x)

    # Confiance
    confidence = None
    if hasattr(svm_model, "predict_proba"):
        proba = svm_model.predict_proba(x)
        confidence = float(np.max(proba))

    # Décoder le label si nécessaire
    if label_encoder is not None:
        name = label_encoder.inverse_transform(pred)[0]
    else:
        name = pred[0]

    return str(name), confidence
