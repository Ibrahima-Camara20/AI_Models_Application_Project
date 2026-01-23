"""
Application principale de reconnaissance faciale avec DeepFace + SVM.
"""
import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from src.core.text_utils import (
    extract_name_prefix_from_filename,
    names_match,
    pretty_name,
    list_images_in_dir
)
from src.core.model_loader import load_svm_model
from src.core.pipeline import run_pipeline_single
from src.gui.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    DEFAULT_BACKEND,
    QUEUE_POLL_INTERVAL_MS,
    MAX_WORDS_IN_NAME
)
from src.gui.batch_processor import BatchProcessor
from src.gui.ui_builder import (
    build_config_section,
    build_image_section,
    build_results_section
)
from src.gui.image_annotator import draw_all_annotations


class FacePredictApp(tk.Tk):
    """
    Application Tkinter pour la reconnaissance faciale.
    
    Fonctionnalités :
    - Chargement d'un modèle SVM pré-entraîné
    - Prédiction sur une image unique
    - Traitement batch d'un dossier d'images
    - Comparaison automatique avec les noms de fichiers
    """
    
    def __init__(self):
        """Initialise l'application."""
        super().__init__()
        self.title("Face Recognition Demo (DeepFace + SVM)")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # État de l'application
        self.model_data = None
        self.image_path = None
        self.tk_img = None
        
        # Stockage des détections pour visualisation
        self.current_boxes = {'yolo_boxes': [], 'retinaface_boxes': []}
        self.current_result = None
        
        # Stockage du dossier batch
        self.batch_folder = None
        self.batch_images = []

        # Queue pour communication avec le batch processor
        self.q = queue.Queue()

        # Variables Tkinter
        self.var_model = tk.StringVar(value="")
        self.var_backend = tk.StringVar(value=DEFAULT_BACKEND)
        self.var_show_boxes = tk.BooleanVar(value=True)
        
        # Batch processor
        self.batch_processor = None

        # Construction de l'interface
        self._build_ui()
        
        # Démarrage du polling de la queue
        self.after(QUEUE_POLL_INTERVAL_MS, self._poll_queue)

    def _build_ui(self):
        """Construit l'interface graphique via les builders."""
        # Section configuration
        build_config_section(
            self,
            self.var_model,
            self.var_backend,
            self.pick_model
        )
        
        # Section principale
        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=8)
        
        # Section image (gauche)
        self.canvas, self.progress, self.lbl_prog, self.btn_stop = build_image_section(
            main,
            self.pick_image,
            self.pick_folder,
            self.predict_one_clicked,
            self.stop_batch,
            self.on_toggle_boxes,
            self.var_show_boxes
        )
        
        # Section résultats (droite)
        results_widgets = build_results_section(main)
        self.pipeline_widget = results_widgets["pipeline_widget"]
        self.lbl_name = results_widgets["lbl_name"]
        self.lbl_conf = results_widgets["lbl_conf"]
        self.confidence_bar = results_widgets["confidence_bar"]
        self.lbl_status = results_widgets["lbl_status"]
        self.stats_widget = results_widgets["stats_widget"]
        self.txt = results_widgets["txt_log"]
        
        self.log("Prêt. 1) Charger modèle .pkl  2) Choisir image/dossier  3) Prédire")

    # =============================
    # Méthodes utilitaires
    # =============================
    
    def log(self, msg: str):
        """Ajoute un message au log."""
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def show_image(self, path: str, boxes=None):
        """Affiche une image sur le canvas avec annotations optionnelles."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(CANVAS_WIDTH / w, CANVAS_HEIGHT / h, 1.0)
        
        # Appliquer les annotations si demandé
        if self.var_show_boxes.get() and boxes:
            yolo_boxes = boxes.get('yolo_boxes', [])
            retinaface_boxes = boxes.get('retinaface_boxes', [])
            
            pred_name = None
            pred_conf = None
            if self.current_result and self.current_result.get('success'):
                pred_name = self.current_result.get('predicted_name')
                pred_conf = self.current_result.get('confidence')
            
            img = draw_all_annotations(
                img,
                yolo_boxes=yolo_boxes if yolo_boxes else None,
                retinaface_boxes=retinaface_boxes if retinaface_boxes else None,
                predicted_name=pred_name,
                confidence=pred_conf
            )
        
        # Redimensionner et afficher
        disp = img.resize((int(w * scale), int(h * scale)))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(
            CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2,
            image=self.tk_img, anchor="center"
        )

    def set_result(self, name: str, conf: float | None, status: str):
        """Met à jour l'affichage des résultats."""
        self.lbl_name.config(text=name)
        if conf is None:
            self.lbl_conf.config(text="Confiance: n/a")
            self.confidence_bar['value'] = 0
        else:
            self.lbl_conf.config(text=f"Confiance: {conf*100:.4f}%")
            self.confidence_bar['value'] = conf * 100
        self.lbl_status.config(text=f"Status: {status}")
    
    def on_toggle_boxes(self):
        """Callback pour toggle des boxes."""
        if self.image_path and os.path.exists(self.image_path):
            self.show_image(self.image_path, self.current_boxes)
    
    def on_pipeline_status_update(self, stage, status, count=None):
        """Callback pour mises à jour du pipeline."""
        self.pipeline_widget.update_stage(stage, status, count)
    
    def refresh_image_display(self):
        """Rafraîchit l'affichage avec annotations."""
        if self.image_path and os.path.exists(self.image_path):
            self.show_image(self.image_path, self.current_boxes)

    # =============================
    # Sélection de fichiers
    # =============================
    
    def pick_model(self):
        """Dialogue pour choisir le modèle .pkl."""
        p = filedialog.askopenfilename(
            title="Choisir le modèle .pkl",
            filetypes=[("Pickle model", "*.pkl"), ("All files", "*.*")]
        )
        if not p:
            return
        self.var_model.set(p)
        try:
            self.model_data = load_svm_model(p)
            self.log(f"[OK] Modèle chargé: {os.path.basename(p)}")
            self.log(f"[INFO] Clés: {list(self.model_data.keys())}")
        except Exception as e:
            self.model_data = None
            messagebox.showerror("Erreur modèle", str(e))

    def pick_image(self):
        """Dialogue pour choisir une image."""
        p = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not p:
            return
        
        # Réinitialiser le mode batch
        self.batch_folder = None
        self.batch_images = []
        self.progress["value"] = 0
        self.lbl_prog.config(text="0 / 0")
        
        self.image_path = p
        self.show_image(p)
        self.log(f"[OK] Image sélectionnée: {os.path.basename(p)}")

    def pick_folder(self):
        """Dialogue pour choisir un dossier."""
        folder = filedialog.askdirectory(title="Choisir un dossier d'images")
        if not folder:
            return

        images = list_images_in_dir(folder)
        if not images:
            messagebox.showwarning("Dossier", "Aucune image trouvée dans ce dossier.")
            return

        # Stocker pour le batch
        self.batch_folder = folder
        self.batch_images = images
        
        # Afficher la première image
        self.image_path = images[0]
        self.show_image(self.image_path)
        
        # Initialiser la progression
        self.progress["maximum"] = len(images)
        self.progress["value"] = 0
        self.lbl_prog.config(text=f"0 / {len(images)}")
        
        self.log(f"\n[INFO] Dossier sélectionné: {folder}")
        self.log(f"[INFO] {len(images)} images trouvées.")
        self.log(f"[INFO] Cliquez sur 'Prédire' pour lancer le traitement batch.")

    # =============================
    # Logique de prédiction
    # =============================
    
    def decide_display(self, filename: str, predicted_raw: str) -> tuple[str, str, str]:
        """Décide du nom à afficher en comparant avec le filename."""
        true_prefix = extract_name_prefix_from_filename(filename, max_words=MAX_WORDS_IN_NAME)
        ok = names_match(predicted_raw, true_prefix)
        if ok:
            return pretty_name(predicted_raw), "OK", true_prefix
        return "UNKNOWN", "UNKNOWN", true_prefix

    def predict_one_clicked(self):
        """Effectue une prédiction : image unique ou batch."""
        if self.model_data is None:
            messagebox.showwarning("Modèle", "Charge d'abord un modèle .pkl.")
            return
        
        # Mode Batch
        if self.batch_images and len(self.batch_images) > 1:
            self._start_batch()
            return
        
        # Mode Image Unique
        if not self.image_path:
            messagebox.showwarning("Image", "Choisis une image ou un dossier.")
            return

        backend = self.var_backend.get().strip()
        fname = os.path.basename(self.image_path)

        self.log(f"\n[RUN] Pipeline Single Image | Backend={backend} | file={fname}")
        self.pipeline_widget.reset()
        
        # Réinitialiser et initialiser le chrono stats
        self.stats_widget.reset()
        start_time = time.time()

        try:
            result = run_pipeline_single(
                self.image_path,
                self.model_data,
                backend,
                log_callback=self.log,
                status_callback=self.on_pipeline_status_update,
                cleanup=True
            )
            
            self.current_result = result
            self.current_boxes = result.get('boxes', {'yolo_boxes': [], 'retinaface_boxes': []})
            
            # Calculer le temps écoulé
            elapsed = time.time() - start_time
            
            if result["success"]:
                pred_name = result["predicted_name"]
                conf = result["confidence"]
                
                display, status, true_prefix = self.decide_display(fname, pred_name)
                self.set_result(display, conf, status)

                self.log(f"[INFO] filename_prefix={true_prefix}")
                self.log(f"[INFO] predicted={pred_name}")
                self.log(f"[OK] status={status}")
                
                # Mettre à jour les stats pour 1 image
                counts = {'OK': 0, 'UNKNOWN': 0, 'ERROR': 0}
                if status in counts:
                    counts[status] = 1
                self.stats_widget.update_stats(1, 1, counts, conf)
                
                self.refresh_image_display()
            else:
                self.set_result("ERROR", None, "ERROR")
                self.log(f"[ERROR] Pipeline échoué à l'étape '{result['stage']}'")
                self.log(f"[ERROR] {result['error']}")
                messagebox.showerror("Erreur Pipeline", result['error'])
                
                # Mettre à jour stats erreur
                self.stats_widget.update_stats(1, 1, {'OK': 0, 'UNKNOWN': 0, 'ERROR': 1}, None)

        except Exception as e:
            self.set_result("ERROR", None, "ERROR")
            self.log(f"[ERROR] {type(e).__name__}: {e}")
            messagebox.showerror("Erreur prédiction", str(e))
            self.stats_widget.update_stats(1, 1, {'OK': 0, 'UNKNOWN': 0, 'ERROR': 1}, None)
    
    def _start_batch(self):
        """Démarre le traitement batch via BatchProcessor."""
        backend = self.var_backend.get().strip()
        
        self.log(f"\n[RUN] Batch Processing | {len(self.batch_images)} images | Backend={backend}")
        self.btn_stop.config(state="normal")
        
        # Réinitialiser les stats
        self.stats_widget.reset()
        
        # Créer et démarrer le batch processor
        self.batch_processor = BatchProcessor(self)
        self.batch_processor. start(self.batch_images, self.model_data, backend)

    def stop_batch(self):
        """Arrête le traitement batch."""
        if self.batch_processor:
            self.batch_processor.stop()
        self.btn_stop.config(state="disabled")

    def _poll_queue(self):
        """Traite les messages de la queue."""
        try:
            while True:
                kind, payload = self.q.get_nowait()

                if kind == "log":
                    self.log(payload)
                elif kind == "store_result":
                    self.current_result = payload
                elif kind == "store_boxes":
                    self.current_boxes = payload
                elif kind == "show":
                    self.image_path = payload
                    self.show_image(payload, self.current_boxes)
                elif kind == "result":
                    name, conf, status = payload
                    self.set_result(name, conf, status)
                elif kind == "progress":
                    i, total = payload
                    self.progress["value"] = i
                    self.lbl_prog.config(text=f"{i} / {total}")
                elif kind == "pipeline_status":
                    stage, status, count = payload
                    self.on_pipeline_status_update(stage, status, count)
                elif kind == "batch_stats":
                    # Mettre à jour le widget de stats
                    processed, total, counts, conf = payload
                    self.stats_widget.update_stats(processed, total, counts, conf)
                elif kind == "reset_pipeline":
                    self.pipeline_widget.reset()
                elif kind == "done":
                    self.btn_stop.config(state="disabled")
                    self.log("[INFO] Batch terminé / prêt.")
        except queue.Empty:
            pass

        self.after(QUEUE_POLL_INTERVAL_MS, self._poll_queue)
