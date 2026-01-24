"""
Module de construction d'interface pour l'application de reconnaissance faciale.

Fournit des fonctions pour construire les différentes sections de l'UI.
"""
import tkinter as tk
from tkinter import ttk

from src.gui.constants import (
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    CANVAS_BG_COLOR,
    MODEL_ENTRY_WIDTH,
    BACKEND_COMBO_WIDTH,
    FONT_RESULT_NAME,
    FONT_RESULT_INFO,
    FONT_SECTION_TITLE,
    TEXT_LOG_WIDTH,
    TEXT_LOG_HEIGHT
)
from src.gui.pipeline_status_widget import PipelineStatusWidget
from src.gui.batch_stats_widget import BatchStatsWidget


def build_config_section(parent, var_model, on_pick_model):
    """Construit la section de configuration (modèle et backend)."""
    top = ttk.LabelFrame(parent, text="Configuration")
    top.pack(fill="x", padx=10, pady=8)
    
    # Sélection du modèle
    ttk.Label(top, text="Modèle:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
    
    # [MODIFIED] Utilisation d'un Combobox au lieu d'Entry + FileDialog
    from src.gui.constants import MODEL_FILES
    model_names = list(MODEL_FILES.keys())
    
    cb_model = ttk.Combobox(
        top, 
        textvariable=var_model, 
        values=model_names,
        state="readonly",
        width=BACKEND_COMBO_WIDTH + 5
    )
    cb_model.grid(row=0, column=1, sticky="w", padx=6)
    
    # Bind de l'événement de sélection
    cb_model.bind("<<ComboboxSelected>>", lambda e: on_pick_model())
    
    # Hack pour pré-sélectionner si var_model est vide
    if not var_model.get() and model_names:
        cb_model.current(0)
    
    # [REMOVED] Sélection du backend (automatique maintenant)
    
    top.columnconfigure(1, weight=1)
    
    return top


def build_image_section(parent, on_pick_image, on_pick_folder, on_predict, 
                       on_stop, on_toggle_boxes, var_show_boxes,
                       on_prev=None, on_next=None, on_batch=None):
    """Construit la section d'affichage d'image (gauche)."""
    left = ttk.LabelFrame(parent, text="Photo")
    left.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Canvas pour l'image
    canvas = tk.Canvas(left, bg=CANVAS_BG_COLOR, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    canvas.pack(fill="both", expand=True, padx=8, pady=8)
    
    # Zone de navigation (Prev/Next)
    nav_frame = ttk.Frame(left)
    nav_frame.pack(fill="x", padx=8, pady=(0, 6))
    
    # Bouton Prev
    btn_prev = ttk.Button(nav_frame, text="<< Précédent", command=on_prev)
    btn_prev.pack(side="left", fill="x", expand=True, padx=(0, 4))
    
    # Label index (sera mis à jour par l'app)
    # On le place au milieu
    
    # Bouton Next
    btn_next = ttk.Button(nav_frame, text="Suivant >>", command=on_next)
    btn_next.pack(side="left", fill="x", expand=True, padx=(4, 0))
    
    # Boutons d'action
    btns = ttk.Frame(left)
    btns.pack(fill="x", padx=8, pady=(0, 6))
    ttk.Button(btns, text="Image...", command=on_pick_image).pack(side="left")
    ttk.Button(btns, text="Dossier...", command=on_pick_folder).pack(
        side="left", padx=(8, 0)
    )
    ttk.Button(btns, text="Prédire (Cette image)", command=on_predict).pack(
        side="left", padx=(8, 0)
    )
    
    # Nouveau bouton Batch
    if on_batch:
        ttk.Button(btns, text="Traiter tout", command=on_batch).pack(
            side="left", padx=(8, 0)
        )
        
    btn_stop = ttk.Button(btns, text="STOP", command=on_stop, state="disabled")
    btn_stop.pack(side="left", padx=(8, 0))
    
    # Barre de progression
    progress = ttk.Progressbar(left, mode="determinate")
    progress.pack(fill="x", padx=8, pady=(0, 6))
    lbl_prog = ttk.Label(left, text="0 / 0")
    lbl_prog.pack(anchor="w", padx=8)
    
    # Checkbox pour afficher les détections
    chk_boxes = ttk.Checkbutton(
        left,
        text="Afficher les détections (YOLO + RetinaFace)",
        variable=var_show_boxes,
        command=on_toggle_boxes
    )
    chk_boxes.pack(anchor="w", padx=8, pady=(6, 0))
    
    return canvas, progress, lbl_prog, btn_stop, btn_prev, btn_next


def build_results_section(parent):
    """Construit la section de résultats (droite)."""
    right = ttk.Frame(parent)
    right.pack(side="right", fill="both", expand=False)
    
    # Widget de statut du pipeline
    pipeline_widget = PipelineStatusWidget(right)
    pipeline_widget.pack(fill="x", padx=8, pady=8)
    
    # Séparateur
    ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=8)
    
    # Résultats de prédiction
    result_frame = ttk.LabelFrame(right, text="Prédiction")
    result_frame.pack(fill="x", padx=8, pady=(0, 8))
    
    lbl_name = ttk.Label(result_frame, text="—", font=FONT_RESULT_NAME)
    lbl_name.pack(anchor="w", padx=12, pady=(16, 6))
    
    lbl_conf = ttk.Label(result_frame, text="Confiance: —", font=FONT_RESULT_INFO)
    lbl_conf.pack(anchor="w", padx=12, pady=4)
    
    # Barre de confiance visuelle
    confidence_bar = ttk.Progressbar(result_frame, mode="determinate", length=200)
    confidence_bar.pack(fill="x", padx=12, pady=(4, 16))
    
    # Séparateur
    ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=8)
    
    # [NEW] Widget statistiques batch
    stats_widget = BatchStatsWidget(right)
    stats_widget.pack(fill="x", padx=8, pady=(0, 8))
    
    # Séparateur
    ttk.Separator(right, orient="horizontal").pack(fill="x", padx=12, pady=8)
    
    # Logs
    ttk.Label(right, text="Logs:", font=FONT_SECTION_TITLE).pack(anchor="w", padx=12)
    txt_log = tk.Text(right, width=TEXT_LOG_WIDTH, height=TEXT_LOG_HEIGHT, wrap="word")
    txt_log.pack(fill="both", expand=True, padx=12, pady=(6, 12))
    
    return {
        "pipeline_widget": pipeline_widget,
        "lbl_name": lbl_name,
        "lbl_conf": lbl_conf,
        "confidence_bar": confidence_bar,
        "stats_widget": stats_widget,
        "txt_log": txt_log
    }
