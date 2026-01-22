"""
Utilitaires pour la normalisation et la comparaison de texte.

Ce module fournit des fonctions pour :
- Normaliser les noms (enlever accents, uniformiser la casse)
- Extraire les préfixes de noms depuis les noms de fichiers
- Comparer les noms de manière robuste
- Lister les images dans un dossier
"""
import os
import re
import unicodedata


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def norm(s: str) -> str:
    s = strip_accents(s.lower().strip())
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z\s]", " ", s)   # garde lettres + espaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pretty_name(s: str) -> str:
    s = norm(s)
    return " ".join(w.capitalize() for w in s.split())


def extract_name_prefix_from_filename(filename: str, max_words: int = 4) -> str:
    
    base = os.path.splitext(os.path.basename(filename))[0]

    # Enlever suffixes type "-person-..."
    base = base.split("-person")[0]

    # Normaliser séparateurs
    base = base.replace("_", " ").strip()
    base = re.sub(r"\s+", " ", base)

    # Couper dès qu'on voit un chiffre
    m = re.search(r"\d", base)
    if m:
        base = base[:m.start()].strip()

    toks = base.split()
    if not toks:
        return ""

    # Prendre les premiers mots (noms composés possibles)
    prefix = " ".join(toks[:max_words]).strip()
    return prefix


def names_match(predicted: str, filename_prefix: str) -> bool:
    """ Vérifie si deux noms correspondent de manière robuste."""
    p = norm(predicted)
    t = norm(filename_prefix)

    if not p or not t:
        return False

    # Match exact
    if p == t:
        return True

    # Match partiel : tous les mots de predicted sont dans prefix
    p_words = p.split()
    t_set = set(t.split())

    return all(w in t_set for w in p_words)


def list_images_in_dir(folder: str) -> list[str]:
    """ Liste tous les fichiers images dans un dossier. """
    paths = []
    for f in os.listdir(folder):
        if f.lower().endswith(IMG_EXTS):
            paths.append(os.path.join(folder, f))
    return sorted(paths)
