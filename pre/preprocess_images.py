import os
from pathlib import Path
from PIL import Image

# ─── PARAMÈTRES ──────────────────────────────────────────────────────────────
DOSSIER_ENTREE = Path("data/105_classes_pins_dataset")
DOSSIER_SORTIE = Path("data_preprocessed/105_classes_pins_dataset_224")

TAILLE_IMAGE = (224, 224)      # Taille standard pour VGG
FORMAT_SORTIE = "JPEG"         # Format homogène
QUALITE_JPEG = 95              # Bonne qualité
IGNORER_SI_EXISTE = True       # Ne pas retraiter si déjà fait


# ─── FONCTIONS ────────────────────────────────────────────────────────────────

def est_fichier_image(chemin: Path) -> bool:
    """Vérifie si le fichier est une image supportée."""
    return chemin.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]


def pretraiter_une_image(chemin_source: Path,chemin_destination: Path) -> tuple[bool, str]:
    """
    Prétraite une image :
    - ouverture
    - conversion en RGB
    - redimensionnement
    - sauvegarde en JPEG

    Retourne :
    - succès (bool)
    - message (str)
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
    if not DOSSIER_ENTREE.exists():
        raise FileNotFoundError(
            f"Dossier d'entrée introuvable : {DOSSIER_ENTREE.resolve()}"
        )

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    total_images = 0
    images_ok = 0
    images_echouees = 0

    for dossier_classe in sorted(
        [d for d in DOSSIER_ENTREE.iterdir() if d.is_dir()]
    ):
        dossier_sortie_classe = DOSSIER_SORTIE / dossier_classe.name
        dossier_sortie_classe.mkdir(parents=True, exist_ok=True)

        for chemin_image in dossier_classe.iterdir():
            if not chemin_image.is_file() or not est_fichier_image(chemin_image):
                continue

            total_images += 1

            nom_sortie = chemin_image.stem + ".jpg"
            chemin_sortie = dossier_sortie_classe / nom_sortie

            if IGNORER_SI_EXISTE and chemin_sortie.exists():
                images_ok += 1
                continue

            succes, message = pretraiter_une_image(
                chemin_image,
                chemin_sortie
            )

            if succes:
                images_ok += 1
            else:
                images_echouees += 1
                print(f"[ERREUR] {chemin_image} → {message}")

    print("\n✅ Prétraitement terminé")
    print(f"Total images traitées  : {total_images}")
    print(f"Images réussies        : {images_ok}")
    print(f"Images échouées        : {images_echouees}")
    print(f"Dossier de sortie      : {DOSSIER_SORTIE.resolve()}")


# ─── POINT D’ENTRÉE ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pretraiter_dataset()
