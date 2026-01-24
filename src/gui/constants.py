"""
Constantes de configuration pour l'interface graphique.
"""

# Extensions d'images supportées
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Dimensions de la fenêtre principale
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700

# Dimensions du canvas d'affichage
CANVAS_WIDTH = 680
CANVAS_HEIGHT = 500

# Configuration des modèles d'embedding


# Configuration du threading
QUEUE_POLL_INTERVAL_MS = 120

# Configuration de l'extraction de noms
MAX_WORDS_IN_NAME = 4

# Configuration des polices
# Ça veut dire quoi ?
# c'est quoi FONT_RESULT_NAME ? ici c'est 
FONT_RESULT_NAME = ("Segoe UI", 20, "bold")
FONT_RESULT_INFO = ("Segoe UI", 12)
FONT_SECTION_TITLE = ("Segoe UI", 11, "bold")

# Taille du texte dessiné sur les images
ANNOTATION_FONT_SIZE = 25

# Couleurs
CANVAS_BG_COLOR = "#101010"

# Dimensions des widgets
TEXT_LOG_WIDTH = 48
TEXT_LOG_HEIGHT = 22
MODEL_ENTRY_WIDTH = 90
BACKEND_COMBO_WIDTH = 12

# Configuration des fichiers modèles
# Mappe le nom affiché dans l'UI vers le chemin relatif du fichier .pkl
MODEL_FILES = {
    "VGG-Face": "models/svm_vgg_face.pkl",
    "ArcFace": "models/svm_arcface.pkl"
}
DEFAULT_MODEL = "VGG-Face"
