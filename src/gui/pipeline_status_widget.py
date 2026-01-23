"""
Widget de visualisation du statut du pipeline de reconnaissance faciale.

Affiche visuellement les 3 étapes du pipeline :
1. YOLO - Détection de personnes
2. RetinaFace - Extraction de visages  
3. Prédiction - Reconnaissance faciale

Chaque étape a un indicateur de statut (idle/running/success/error).
"""
import tkinter as tk
from tkinter import ttk


class PipelineStatusWidget(ttk.Frame):
    """
    Widget pour afficher le statut des 3 étapes du pipeline.
    
    Statuts possibles pour chaque étape :
    - 'idle' : Gris - En attente
    - 'running' : Jaune - En cours
    - 'success' : Vert - Terminé avec succès
    - 'error' : Rouge - Erreur
    """
    
    # Couleurs pour chaque statut
    COLORS = {
        'idle': '#6B7280',      # Gris
        'running': '#F59E0B',   # Jaune/Orange
        'success': '#10B981',   # Vert
        'error': '#EF4444'      # Rouge
    }
    
    # Symboles pour chaque statut
    SYMBOLS = {
        'idle': '○',
        'running': '⏳',
        'success': '✓',
        'error': '✗'
    }
    
    def __init__(self, parent, **kwargs):
        """Initialise le widget."""
        super().__init__(parent, **kwargs)
        
        # État de chaque étape
        self.stages = {
            'yolo': {'status': 'idle', 'count': 0},
            'retinaface': {'status': 'idle', 'count': 0},
            'prediction': {'status': 'idle', 'count': 0}
        }
        
        # Construction de l'interface
        self._build_ui()
    
    def _build_ui(self):
        """Construit l'interface du widget."""
        # Titre
        title = ttk.Label(self, text="Pipeline de Traitement", 
                         font=("Segoe UI", 11, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 12))
        
        # Container pour les étapes
        stages_frame = ttk.Frame(self)
        stages_frame.pack(fill="x", padx=8, pady=(0, 8))
        
        # Créer les 3 étapes
        self.stage_widgets = {}
        
        # Étape 1 : YOLO
        self.stage_widgets['yolo'] = self._create_stage(
            stages_frame, 
            "1. YOLO", 
            "Détection de personnes",
            row=0
        )
        
        # Flèche
        ttk.Label(stages_frame, text="↓", font=("Segoe UI", 14)).grid(
            row=1, column=0, pady=2
        )
        
        # Étape 2 : RetinaFace
        self.stage_widgets['retinaface'] = self._create_stage(
            stages_frame,
            "2. RetinaFace",
            "Extraction de visages",
            row=2
        )
        
        # Flèche
        ttk.Label(stages_frame, text="↓", font=("Segoe UI", 14)).grid(
            row=3, column=0, pady=2
        )
        
        # Étape 3 : Prédiction
        self.stage_widgets['prediction'] = self._create_stage(
            stages_frame,
            "3. Prédiction",
            "Reconnaissance faciale",
            row=4
        )
    
    def _create_stage(self, parent, title, subtitle, row):
        """Crée un widget pour une étape du pipeline."""
        # Frame principal
        frame = ttk.Frame(parent, relief="solid", borderwidth=1)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        parent.columnconfigure(0, weight=1)
        
        # Padding interne
        inner = ttk.Frame(frame)
        inner.pack(fill="both", expand=True, padx=10, pady=8)
        
        # Indicateur de statut (cercle coloré)
        indicator = tk.Canvas(inner, width=20, height=20, 
                            bg=self.winfo_toplevel().cget('bg'),
                            highlightthickness=0)
        indicator.pack(side="left", padx=(0, 10))
        circle = indicator.create_oval(2, 2, 18, 18, 
                                      fill=self.COLORS['idle'],
                                      outline="")
        
        # Texte
        text_frame = ttk.Frame(inner)
        text_frame.pack(side="left", fill="both", expand=True)
        
        # Titre de l'étape
        lbl_title = ttk.Label(text_frame, text=title, 
                             font=("Segoe UI", 10, "bold"))
        lbl_title.pack(anchor="w")
        
        # Sous-titre
        lbl_subtitle = ttk.Label(text_frame, text=subtitle,
                                font=("Segoe UI", 9),
                                foreground="#6B7280")
        lbl_subtitle.pack(anchor="w")
        
        # Label pour le compteur
        lbl_count = ttk.Label(inner, text="", font=("Segoe UI", 9))
        lbl_count.pack(side="right", padx=(10, 0))
        
        return {
            'frame': frame,
            'indicator': indicator,
            'circle': circle,
            'label_count': lbl_count
        }
    
    def update_stage(self, stage_name, status, count=None):
        """Met à jour le statut d'une étape."""
        if stage_name not in self.stages:
            return
        
        self.stages[stage_name]['status'] = status
        if count is not None:
            self.stages[stage_name]['count'] = count
        
        # Mettre à jour l'indicateur visuel
        widgets = self.stage_widgets[stage_name]
        color = self.COLORS[status]
        
        # Changer la couleur du cercle
        widgets['indicator'].itemconfig(widgets['circle'], fill=color)
        
        # Mettre à jour le texte du compteur
        if count is not None and status == 'success':
            if stage_name == 'yolo':
                text = f"{count} pers." if count > 0 else "Aucun"
            elif stage_name == 'retinaface':
                text = f"{count} visage{'s' if count > 1 else ''}" if count > 0 else "Aucun"
            else:  # prediction
                text = "✓"
            widgets['label_count'].config(text=text)
        else:
            widgets['label_count'].config(text="")
    
    def reset(self):
        """Réinitialise toutes les étapes à l'état 'idle'."""
        for stage_name in self.stages:
            self.update_stage(stage_name, 'idle', 0)
    
    def set_error(self, stage_name, error_msg=None):
        """Marque une étape comme erreur."""
        self.update_stage(stage_name, 'error')
        if error_msg:
            widgets = self.stage_widgets[stage_name]
            widgets['label_count'].config(text="❌")


if __name__ == "__main__":
    # Test du widget
    root = tk.Tk()
    root.title("Test Pipeline Status Widget")
    root.geometry("350x400")
    
    widget = PipelineStatusWidget(root)
    widget.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Simulation d'états
    def simulate():
        import time
        
        # Reset
        widget.reset()
        root.update()
        root.after(1000)
        
        # YOLO en cours
        widget.update_stage('yolo', 'running')
        root.update()
        root.after(1000)
        
        # YOLO terminé
        widget.update_stage('yolo', 'success', 3)
        widget.update_stage('retinaface', 'running')
        root.update()
        root.after(1000)
        
        # RetinaFace terminé
        widget.update_stage('retinaface', 'success', 2)
        widget.update_stage('prediction', 'running')
        root.update()
        root.after(1000)
        
        # Prédiction terminée
        widget.update_stage('prediction', 'success')
        root.update()
    
    ttk.Button(root, text="Simuler Pipeline", command=simulate).pack(pady=10)
    
    root.mainloop()
