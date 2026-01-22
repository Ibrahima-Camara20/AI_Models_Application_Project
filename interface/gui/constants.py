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
DEFAULT_BACKEND = "ArcFace"
AVAILABLE_BACKENDS = ["ArcFace", "VGG11"]

# Configuration du threading
QUEUE_POLL_INTERVAL_MS = 120

# Configuration de l'extraction de noms
MAX_WORDS_IN_NAME = 4

# Configuration des polices
FONT_RESULT_NAME = ("Segoe UI", 20, "bold")
FONT_RESULT_INFO = ("Segoe UI", 12)
FONT_SECTION_TITLE = ("Segoe UI", 11, "bold")

# Couleurs
CANVAS_BG_COLOR = "#101010"

# Dimensions des widgets
TEXT_LOG_WIDTH = 48
TEXT_LOG_HEIGHT = 22
MODEL_ENTRY_WIDTH = 90
BACKEND_COMBO_WIDTH = 12
