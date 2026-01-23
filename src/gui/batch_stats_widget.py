"""
Widget pour afficher les statistiques en temps réel du traitement batch.
"""
import tkinter as tk
from tkinter import ttk
import time

class BatchStatsWidget(ttk.LabelFrame):
    """
    Affiche les statistiques du batch en cours :
    - Temps écoulé / restant
    - Taux de réussite
    - Distribution des confidences (graphique)
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Statistiques Batch", **kwargs)
        
        # --- Métriques Temps ---
        time_frame = ttk.Frame(self)
        time_frame.pack(fill="x", padx=8, pady=4)
        
        self.lbl_time = ttk.Label(time_frame, text="Temps: 00:00 / Estimé: --:--")
        self.lbl_time.pack(anchor="w")
        
        self.lbl_speed = ttk.Label(time_frame, text="Vitesse: 0.0 img/s")
        self.lbl_speed.pack(anchor="w")
        
        # --- Métriques Réussite ---
        stats_frame = ttk.Frame(self)
        stats_frame.pack(fill="x", padx=8, pady=4)
        
        self.lbl_ok = ttk.Label(stats_frame, text="✅ OK: 0", foreground="green")
        self.lbl_ok.pack(side="left", padx=(0, 10))
        
        self.lbl_unknown = ttk.Label(stats_frame, text="❓ UNKNOWN: 0", foreground="#D97706") # Orange
        self.lbl_unknown.pack(side="left", padx=(0, 10))
        
        self.lbl_error = ttk.Label(stats_frame, text="❌ ERROR: 0", foreground="red")
        self.lbl_error.pack(side="left")
        
        # --- Graphique Distribution ---
        ttk.Label(self, text="Distribution Confiance:", font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=(8, 2))
        
        self.canvas_height = 60
        self.canvas = tk.Canvas(self, height=self.canvas_height, bg="white", highlightthickness=1, highlightbackground="#E5E7EB")
        self.canvas.pack(fill="x", padx=8, pady=(0, 8))
        
        # Données
        self.start_time = None
        self.confidences = []
        
    def reset(self):
        """Réinitialise les statistiques."""
        self.start_time = time.time()
        self.confidences = []
        self.update_time_labels(0, 0, 0)
        self.update_counts(0, 0, 0)
        self.draw_distribution()
        
    def update_stats(self, processed, total, counts, conf_value=None):
        """
        Met à jour toutes les stats.
        
        Args:
            processed: Nombre d'images traitées
            total: Nombre total d'images
            counts: Dict {'OK': int, 'UNKNOWN': int, 'ERROR': int}
            conf_value: Nouvelle valeur de confiance à ajouter (optionnel)
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        # Mise à jour temps
        elapsed = time.time() - self.start_time
        self.update_time_labels(elapsed, processed, total)
        
        # Mise à jour compteurs
        self.update_counts(counts.get('OK', 0), counts.get('UNKNOWN', 0), counts.get('ERROR', 0))
        
        # Mise à jour graphique
        if conf_value is not None:
            self.confidences.append(conf_value)
            self.draw_distribution()

    def update_time_labels(self, elapsed, processed, total):
        """Met à jour les labels de temps."""
        elapsed_str = time.strftime('%M:%S', time.gmtime(elapsed))
        
        if processed > 0:
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / speed if speed > 0 else 0
            rem_str = time.strftime('%M:%S', time.gmtime(remaining))
            self.lbl_speed.config(text=f"Vitesse: {speed:.1f} img/s")
        else:
            rem_str = "--:--"
            self.lbl_speed.config(text="Vitesse: 0.0 img/s")
            
        self.lbl_time.config(text=f"Temps: {elapsed_str} / Restant: {rem_str}")

    def update_counts(self, ok, unknown, error):
        """Met à jour les compteurs de statut."""
        self.lbl_ok.config(text=f"✅ OK: {ok}")
        self.lbl_unknown.config(text=f"❓ UNKNOWN: {unknown}")
        self.lbl_error.config(text=f"❌ ERROR: {error}")

    def draw_distribution(self):
        """Dessine un histogramme simple des confidences sur le canvas."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        if w < 10: w = 200 # Fallback si pas encore affiché
        h = self.canvas_height
        
        if not self.confidences:
            self.canvas.create_text(w//2, h//2, text="En attente de données...", fill="gray")
            return
            
        # Histogramme simple (10 bins)
        bins = [0] * 10
        for c in self.confidences:
            # c est entre 0 et 1 (normalement)
            if c is None: continue
            idx = min(int(c * 10), 9)
            bins[idx] += 1
            
        max_val = max(bins) if bins else 1
        bar_w = w / 10
        
        # Couleurs dégradées (Rouge -> Jaune -> Vert)
        # 0-5 (0-50%): Rouge, 5-8 (50-80%): Jaune, 8-10 (80-100%): Vert
        colors = ['#EF4444']*5 + ['#F59E0B']*3 + ['#10B981']*2
        
        for i, count in enumerate(bins):
            if count == 0: continue
            
            bar_h = (count / max_val) * (h - 10)
            x0 = i * bar_w
            y0 = h
            x1 = (i + 1) * bar_w - 2 # Petit espace entre les barres
            y1 = h - bar_h
            
            self.canvas.create_rectangle(x0, y1, x1, y0, fill=colors[i], outline="")
            
            # Afficher le count si barre assez large
            if bar_w > 15:
                self.canvas.create_text((x0+x1)/2, y1-5, text=str(count), font=("Arial", 7))
