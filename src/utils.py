import os
import json


def extract_celebrity_name(folder_name):
    # Enlever le préfixe "pins_" s'il existe
    name = folder_name.replace("pins_", "").strip()
    # Enlever les underscores superflus
    name = name.replace("_", " ").strip()
    return name


def save_metadata(metadata, filepath="metadata.json"):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Métadonnées sauvegardées dans {filepath}")
    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder les métadonnées: {e}")


def load_metadata(filepath="metadata.json"):
    try:
        if not os.path.exists(filepath):
            print(f"[ATTENTION] Fichier {filepath} introuvable")
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"[INFO] Métadonnées chargées depuis {filepath}")
        return metadata
    except Exception as e:
        print(f"[ERREUR] Impossible de charger les métadonnées: {e}")
        return {}


def extract_original_filename(face_filename):
    # Format attendu: "face-{i}-{original_filename}"
    parts = face_filename.split('-')
    
    # Vérifier que le format est correct
    if len(parts) < 3 or parts[0] != 'face':
        print(f"[ATTENTION] Format de nom inattendu: {face_filename}")
        return face_filename
    
    # Retirer "face-{i}-" du début
    original = '-'.join(parts[2:])
    return original


def clean_prediction_name(pred_path):
    # Extraire le nom du dossier parent
    folder_name = pred_path.split(os.sep)[-2]
    # Nettoyer le nom
    return extract_celebrity_name(folder_name)


def print_statistics(title, stats):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 50 + "\n")
