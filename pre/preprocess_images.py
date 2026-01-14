"""
Module de prétraitement d'images pour modèles de Deep Learning.

Ce module standardise un dataset d'images en appliquant les transformations suivantes :
- Conversion au format RGB
- Redimensionnement à une taille fixe (224x224 pour VGG)
- Conversion au format JPEG avec compression optimisée
- Organisation en structure de dossiers par classe

Le prétraitement est essentiel pour garantir l'uniformité des données d'entrée
des modèles de classification d'images (VGG, ResNet, etc.).

Auteur  : Projet AI Models Application
Date    : 2026
"""

import os
from pathlib import Path
from PIL import Image


# ─── PARAMÈTRES DE CONFIGURATION ──────────────────────────────────────────────

# Chemins des dossiers d'entrée et sortie
DOSSIER_ENTREE = Path("data/105_classes_pins_dataset")
DOSSIER_SORTIE = Path("data_preprocessed/105_classes_pins_dataset_224")

# Paramètres de traitement des images
TAILLE_IMAGE = (224, 224)      # Taille standard pour VGG16/19 et autres CNN
FORMAT_SORTIE = "JPEG"         # Format homogène pour toutes les images
QUALITE_JPEG = 95              # Qualité élevée (0-100), 95 = excellent compromis
IGNORER_SI_EXISTE = True       # Skip les images déjà traitées (optimisation)


# ─── FONCTIONS ────────────────────────────────────────────────────────────────

def est_fichier_image(chemin: Path) -> bool:
    """Vérifie si le fichier est une image supportée."""
    return chemin.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]


def pretraiter_une_image(chemin_source: Path, chemin_destination: Path) -> tuple[bool, str]:
    """
    Prétraite une seule image selon les paramètres définis.

    Effectue les opérations suivantes dans l'ordre :
    1. Ouverture de l'image source
    2. Conversion en mode RGB (enlève transparence, niveaux de gris, etc.)
    3. Redimensionnement à la taille cible (TAILLE_IMAGE)
    4. Sauvegarde au format JPEG avec compression optimisée

    Args:
        chemin_source (Path): Chemin absolu ou relatif vers l'image source.
        chemin_destination (Path): Chemin où sauvegarder l'image prétraitée.
                                    Le dossier parent est créé automatiquement.

    Returns:
        tuple[bool, str]: Un tuple contenant :
            - bool : True si le traitement a réussi, False sinon
            - str  : Message de statut ("ok" si succès, message d'erreur sinon)

    Raises:
        Aucune exception n'est levée. Toutes les erreurs sont capturées
        et retournées sous forme de tuple (False, message_erreur).

    Example:
        >>> success, msg = pretraiter_une_image(
        ...     Path("source/image.png"),
        ...     Path("sortie/image.jpg")
        ... )
        >>> if success:
        ...     print("Image traitée avec succès")
    """
    try:
        with Image.open(chemin_source) as image:
            image = image.convert("RGB")
            image = image.resize(TAILLE_IMAGE)

            # Créer le dossier destination si nécessaire
            chemin_destination.parent.mkdir(parents=True, exist_ok=True)

            image.save(
                chemin_destination,
                format=FORMAT_SORTIE,
                quality=QUALITE_JPEG,
                optimize=True
            )

        return True, "ok"

    except Exception as erreur:
        return False, str(erreur)


def pretraiter_dataset():
    """
    Prétraite l'ensemble du dataset en parcourant toutes les classes.

    Cette fonction constitue le pipeline complet de prétraitement :
    - Vérifie l'existence du dossier source
    - Crée l'arborescence de sortie
    - Parcourt chaque classe (sous-dossier)
    - Traite chaque image individuellement
    - Affiche un rapport de traitement détaillé
    
    """
    # Vérification de l'existence du dossier source
    if not DOSSIER_ENTREE.exists():
        raise FileNotFoundError(
            f"Dossier d'entrée introuvable : {DOSSIER_ENTREE.resolve()}"
        )

    # Création du dossier de sortie principal
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    # Compteurs de statistiques
    total_images = 0
    images_ok = 0
    images_echouees = 0

    # Parcours de chaque classe (dossier) dans le dataset source
    for dossier_classe in sorted(
        [d for d in DOSSIER_ENTREE.iterdir() if d.is_dir()]
    ):
        # Création du dossier de classe correspondant dans la sortie
        dossier_sortie_classe = DOSSIER_SORTIE / dossier_classe.name
        dossier_sortie_classe.mkdir(parents=True, exist_ok=True)

        # Traitement de chaque image dans la classe
        for chemin_image in dossier_classe.iterdir():
            # Ignore les fichiers non-images et les dossiers
            if not chemin_image.is_file() or not est_fichier_image(chemin_image):
                continue

            total_images += 1

            # Génère le nom de fichier de sortie avec extension .jpg
            nom_sortie = chemin_image.stem + ".jpg"
            chemin_sortie = dossier_sortie_classe / nom_sortie

            # Optimisation : skip si l'image est déjà traitée
            if IGNORER_SI_EXISTE and chemin_sortie.exists():
                images_ok += 1
                continue

            # Traite l'image
            succes, message = pretraiter_une_image(
                chemin_image,
                chemin_sortie
            )

            # Mise à jour des compteurs
            if succes:
                images_ok += 1
            else:
                images_echouees += 1
                print(f"[ERREUR] {chemin_image} → {message}")

    # ─── Affichage du rapport final ───────────────────────────────────────────

    print("\n✅ Prétraitement terminé")
    print(f"Total images traitées  : {total_images}")
    print(f"Images réussies        : {images_ok}")
    print(f"Images échouées        : {images_echouees}")
    print(f"Dossier de sortie      : {DOSSIER_SORTIE.resolve()}")


# ─── POINT D’ENTRÉE ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pretraiter_dataset()
