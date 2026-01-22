"""
Utilitaires pour la gestion des chemins et dossiers temporaires.
"""
import os
import shutil
from pathlib import Path


def ensure_dir(path: str) -> str:
    """Crée un dossier s'il n'existe pas."""
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def clean_dir(path: str) -> None:
    """Vide un dossier de tous ses fichiers et sous-dossiers."""
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def get_temp_dirs(base_dir: str = ".") -> dict[str, str]:
    """Retourne les chemins vers les dossiers temporaires du pipeline."""
    base = Path(base_dir).resolve()
    return {
        "working": str(base / "working"),
        "faces_extraction": str(base / "faces_extraction")
    }


def cleanup_temp_dirs(base_dir: str = ".") -> None:
    """Nettoie les dossiers temporaires du pipeline."""
    temp_dirs = get_temp_dirs(base_dir)
    for dir_path in temp_dirs.values():
        clean_dir(dir_path)
