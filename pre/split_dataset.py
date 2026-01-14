"""
Module de division de dataset en ensembles Train/Validation/Test.

Ce module permet de diviser automatiquement un dataset organisé par classes
en trois sous-ensembles distincts pour l'entraînement de modèles de Machine Learning :
- Train (70%) : Données d'entraînement du modèle
- Validation (20%) : Données de validation pendant l'entraînement
- Test (10%) : Données de test final pour évaluation

La division est stratifiée (chaque classe est divisée proportionnellement)
et reproductible grâce à une seed fixe.

"""

import os
import random
import shutil
from pathlib import Path


# ─── PARAMÈTRES DE CONFIGURATION ──────────────────────────────────────────────

# Chemins des dossiers
DOSSIER_SOURCE = Path("data_preprocessed/105_classes_pins_dataset_224")
DOSSIER_SORTIE = Path("data_splits")

# Ratios de division (doivent sommer à 1.0)
RATIO_TRAIN = 0.70  # 70% pour l'entraînement
RATIO_VAL = 0.20    # 20% pour la validation
RATIO_TEST = 0.10   # 10% pour le test final

# Reproductibilité
SEED = 42  # Seed fixe pour reproduction exacte des résultats

# Extensions de fichiers acceptées
EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ─── FONCTIONS UTILITAIRES ───────────────────────────────────────────────────

def est_image(p: Path) -> bool:
    """Vérifie si un fichier est une image supportée."""
    return p.suffix.lower() in EXTENSIONS


def copier_fichier(src: Path, dst: Path):
    """
    Copie un fichier vers une destination en créant les dossiers nécessaires.

    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_une_classe(dossier_classe: Path):
    """
    Divise les images d'une classe en ensembles train/val/test.

    """
    # Récupère toutes les images du dossier
    images = [p for p in dossier_classe.iterdir() if p.is_file() and est_image(p)]
    
    # Mélange aléatoire pour division équitable
    random.shuffle(images)

    # Calcul des tailles de chaque ensemble
    n = len(images)
    n_train = int(n * RATIO_TRAIN)
    n_val = int(n * RATIO_VAL)
    n_test = n - n_train - n_val  # Le reste évite les erreurs d'arrondi

    # Division des images en trois ensembles
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # Vérification de cohérence
    assert len(test_imgs) == n_test

    return train_imgs, val_imgs, test_imgs


# ─── FONCTION PRINCIPALE ──────────────────────────────────────────────────────

def main():
    """
    Fonction principale du script de division du dataset.

    Effectue les opérations suivantes :
    1. Vérifie l'existence du dossier source
    2. Initialise le générateur aléatoire avec la seed
    3. Crée l'arborescence de sortie (train/val/test)
    4. Parcourt chaque classe et divise ses images
    5. Copie les images vers les bons dossiers
    6. Affiche un rapport statistique complet

    """
    # Vérification de l'existence du dossier source
    if not DOSSIER_SOURCE.exists():
        raise FileNotFoundError(f"Dossier source introuvable : {DOSSIER_SOURCE.resolve()}")

    # Initialisation de la seed pour reproductibilité
    random.seed(SEED)

    # Création de la structure des dossiers de sortie
    for split in ["train", "val", "test"]:
        (DOSSIER_SORTIE / split).mkdir(parents=True, exist_ok=True)

    # Compteurs de statistiques
    total_train = total_val = total_test = 0
    nb_classes = 0

    # Parcours de chaque classe (dossier) dans le dataset source
    for dossier_classe in sorted([d for d in DOSSIER_SOURCE.iterdir() if d.is_dir()]):
        nb_classes += 1
        nom_classe = dossier_classe.name

        # Division de cette classe en train/val/test
        train_imgs, val_imgs, test_imgs = split_une_classe(dossier_classe)

        # Copie des images vers les dossiers de destination appropriés
        for p in train_imgs:
            copier_fichier(p, DOSSIER_SORTIE / "train" / nom_classe / p.name)
        for p in val_imgs:
            copier_fichier(p, DOSSIER_SORTIE / "val" / nom_classe / p.name)
        for p in test_imgs:
            copier_fichier(p, DOSSIER_SORTIE / "test" / nom_classe / p.name)

        # Mise à jour des compteurs globaux
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)

    # Calcul du total global
    total = total_train + total_val + total_test

    # ─── Affichage du rapport final ───────────────────────────────────────────

    print("\n✅ Split terminé")
    print(f"Classes              : {nb_classes}")
    print(f"Train images         : {total_train} ({total_train/total*100:.1f}%)")
    print(f"Val images           : {total_val} ({total_val/total*100:.1f}%)")
    print(f"Test images          : {total_test} ({total_test/total*100:.1f}%)")
    print(f"Total images         : {total}")
    print(f"Dossier de sortie    : {DOSSIER_SORTIE.resolve()}")
    print(f"Ratios demandés      : train={RATIO_TRAIN}, val={RATIO_VAL}, test={RATIO_TEST}")
    print(f"Seed                 : {SEED}")


# ─── POINT D'ENTRÉE ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
