"""
Module d'analyse de dataset d'images.

Ce module permet d'analyser un dataset organisé en classes (sous-dossiers),
et de calculer des statistiques sur la répartition des images par classe.


"""

import os
from collections import Counter

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Détermine le chemin absolu du dossier contenant ce script
dossier_courant = os.path.dirname(os.path.abspath(__file__))

# Construit le chemin vers le dossier de données source
dossier_data = os.path.join(
    dossier_courant,
    "..",
    "data",
    "105_classes_pins_dataset"
)


# ─── FONCTIONS ────────────────────────────────────────────────────────────────

def analyze_dataset(dossier_data):
    """
    Analyse un dataset d'images organisé en classes.

    Cette fonction parcourt un dossier contenant des sous-dossiers (classes),
    compte le nombre d'images dans chaque classe et calcule le total global.

    """
    nombre_de_classes = Counter()
    total_images = 0

    # Parcourt tous les éléments du dossier principal
    for class_name in os.listdir(dossier_data):
        chemin_de_la_classe = os.path.join(dossier_data, class_name)

        # Ignore les fichiers, traite uniquement les dossiers
        if not os.path.isdir(chemin_de_la_classe):
            continue

        # Filtre uniquement les fichiers images avec extensions communes
        images = [
            f for f in os.listdir(chemin_de_la_classe)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Enregistre le nombre d'images pour cette classe
        nombre_de_classes[class_name] = len(images)
        total_images += len(images)

    return nombre_de_classes, total_images



# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
   

    # Effectue l'analyse du dataset
    nombre_de_classes, total_images = analyze_dataset(dossier_data)

    # ─── Affichage des résultats ──────────────────────────────────────────────

    print("\nAnalyse de données")
    print("-" * 40)
    print(f"Nombre de classes (célébrités): {len(nombre_de_classes)}")
    print(f"Nombre total d'images: {total_images}")
    print("-" * 40)

    # Affiche les 10 classes ayant le plus d'images
    print("Nombre d'images par classe (premiers 10):")
    for cls, count in nombre_de_classes.most_common(10):
        print(f"{cls:30s} -> {count}")

    print("-" * 40)

    # Affiche les statistiques de distribution
    print(f"Min images par classe: {min(nombre_de_classes.values())}")
    print(f"Max images par classe: {max(nombre_de_classes.values())}")
    print(
        f"Moyenne images par classe: "
        f"{total_images / len(nombre_de_classes):.2f}"
    )
