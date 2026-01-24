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
    QUEUE_POLL_INTERVAL_MS,
    MAX_WORDS_IN_NAME,
    MODEL_FILES,
    DEFAULT_MODEL
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
        self.batch_results = {} # Cache : {path: {'result': ..., 'boxes': ...}}
        self.current_index = 0

        # Queue pour communication avec le batch processor
        self.q = queue.Queue()

        # Variables Tkinter
        self.var_model = tk.StringVar(value="")
        self.var_show_boxes = tk.BooleanVar(value=True)
        
        # Batch processor
        self.batch_processor = None

        # Construction de l'interface
        self._build_ui()
        
        # [NEW] Charger le modèle par défaut au démarrage
        self.load_model_by_name(DEFAULT_MODEL)
        
        # Démarrage du polling de la queue
        self.after(QUEUE_POLL_INTERVAL_MS, self._poll_queue)

    def _build_ui(self):
        """Construit l'interface graphique via les builders."""
        # Section configuration
        build_config_section(
            self,
            self.var_model,
            self.load_model_by_name  # Callback de changement de combobox
        )
        
        # Section principale
        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=8)
        
        # Section image (gauche)
        self.canvas, self.progress, self.lbl_prog, self.btn_stop, self.btn_prev, self.btn_next = build_image_section(
            main,
            self.pick_image,
            self.pick_folder,
            self.predict_one_clicked,
            self.stop_batch,
            self.on_toggle_boxes,
            self.var_show_boxes,
            on_prev=self.prev_image,
            on_next=self.next_image,
            on_batch=self.start_batch_clicked
        )
        
        # Section résultats (droite)
        results_widgets = build_results_section(main)
        self.pipeline_widget = results_widgets["pipeline_widget"]
        self.lbl_name = results_widgets["lbl_name"]
        self.lbl_conf = results_widgets["lbl_conf"]
        self.confidence_bar = results_widgets["confidence_bar"]
        self.stats_widget = results_widgets["stats_widget"]
        self.txt = results_widgets["txt_log"]
        
        self.log("Prêt. 1) Charger modèle .pkl  2) Choisir image/dossier  3) Prédire")

    # =============================
    # Méthodes utilitaires
    # =============================
    
    
    def get_backend(self) -> str:
        """Déduit le backend (ArcFace/VGG-Face) depuis le nom du modèle."""
        model_name = self.var_model.get()
        if "ArcFace" in model_name:
            return "ArcFace"
        return "VGG-Face"

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
            predictions = None
            if self.current_result and self.current_result.get('success'):
                pred_name = self.current_result.get('predicted_name')
                pred_conf = self.current_result.get('confidence')
                predictions = self.current_result.get('predictions')
            
            img = draw_all_annotations(
                img,
                yolo_boxes=yolo_boxes if yolo_boxes else None,
                retinaface_boxes=retinaface_boxes if retinaface_boxes else None,
                predicted_name=pred_name,
                confidence=pred_conf,
                predictions=predictions
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
        # lbl_status retiré
        # self.lbl_status.config(text=f"Status: {status}")
    
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

    def _update_display_for_current_image(self):
        """Met à jour l'affichage selon le cache ou réinitialise."""
        self.image_path = self.batch_images[self.current_index]
        self.lbl_prog.config(text=f"{self.current_index + 1} / {len(self.batch_images)}")
        
        # Vérifier si on a un résultat en cache
        if self.image_path in self.batch_results:
            cached = self.batch_results[self.image_path]
            self.current_result = cached['result']
            self.current_boxes = cached['boxes']
            
            # Afficher l'image annotée
            self.show_image(self.image_path, self.current_boxes)
            
            # Restaurer les infos textuelles
            res = self.current_result
            if res.get("success"):
                predictions = res.get("predictions", [])
                if predictions and len(predictions) > 1:
                     names = sorted(list(set([p["name"] for p in predictions])))
                     display = ", ".join(names)
                     if len(display) > 30: display = f"{len(predictions)} detected: " + display[:25] + "..."
                     status = "OK (Multiple)"
                     conf = None
                else:
                    pred_name = res.get("predicted_name")
                    conf = res.get("confidence")
                    fname = os.path.basename(self.image_path)
                    display, status, _ = self.decide_display(fname, pred_name)
                
                self.set_result(display, conf, status)
            else:
                 self.set_result("ERROR", None, "ERROR")

        else:
            # Pas de résultat en cache : On nettoie TOUT
            self.current_result = None
            self.current_boxes = {}
            self.show_image(self.image_path) # Image nue
            self.set_result("—", None, "—")
            self.confidence_bar['value'] = 0
            self.pipeline_widget.reset()


    def next_image(self):
        """Passe à l'image suivante du dossier."""
        if not self.batch_images:
            return
        
        self.current_index += 1
        if self.current_index >= len(self.batch_images):
            self.current_index = 0  # Boucler
            
        self._update_display_for_current_image()

    def prev_image(self):
        """Passe à l'image précédente du dossier."""
        if not self.batch_images:
            return
            
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.batch_images) - 1  # Boucler
            
        self._update_display_for_current_image()

    # =============================
    # Sélection de fichiers
    # =============================
    
    def load_model_by_name(self, name=None):
        """Charge le modèle sélectionné dans la combobox."""
        # Si appelé par event bind (name=None), on prend la valeur actuelle
        if not name or not isinstance(name, str):
            name = self.var_model.get()
            
        if name not in MODEL_FILES:
            return

        # Mettre à jour la variable si appelée directement
        if self.var_model.get() != name:
            self.var_model.set(name)
            
        path = MODEL_FILES[name]
        
        # Le backend est maintenant déduit automatiquement via get_backend()
            
        try:
            self.model_data = load_svm_model(path)
            self.log(f"[OK] Modèle chargé: {name} ({os.path.basename(path)})")
            # self.log(f"[INFO] Clés: {list(self.model_data.keys())}")
        except FileNotFoundError:
             self.log(f"[ERROR] Fichier modèle introuvable: {path}")
             messagebox.showerror("Erreur modèle", f"Fichier introuvable:\n{path}")
        except Exception as e:
            self.model_data = None
            self.log(f"[ERROR] Impossible de charger le modèle {name}: {e}")
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
        
        # Pour une image unique, on simule une liste de 1 élément
        self.batch_images = [p]
        self.current_index = 0
        self.lbl_prog.config(text="1 / 1")

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
        self.batch_results = {}
        
        # Afficher la première image
        self.image_path = images[0]
        self.show_image(self.image_path)
        
        # Initialiser la progression
        self.progress["maximum"] = len(images)
        self.progress["value"] = 0
        self.current_index = 0
        self.lbl_prog.config(text=f"1 / {len(images)}")
        
        self.log(f"\n[INFO] Dossier sélectionné: {folder}")
        self.log(f"[INFO] {len(images)} images trouvées.")
        self.log(f"[INFO] Naviguez avec Prec/Suiv, ou cliquez sur 'Traiter tout' pour le batch.")

    # =============================
    # Logique de prédiction
    # =============================
    
    def decide_display(self, filename: str, predicted_raw: str) -> tuple[str, str, str]:
        """Décide du nom à afficher (retourne simplement la prédiction)."""
        true_prefix = extract_name_prefix_from_filename(filename, max_words=MAX_WORDS_IN_NAME)
        # On affiche toujours la prédiction brute, même si mismatch
        # Le status "MISMATCH" peut toujours être utile en interne ou pour les stats
        ok = names_match(predicted_raw, true_prefix)
        status = "OK" if ok else "MISMATCH"
        
        return pretty_name(predicted_raw), status, true_prefix

    def predict_one_clicked(self):
        """Effectue une prédiction : image unique ou batch."""
        if self.model_data is None:
            messagebox.showwarning("Modèle", "Charge d'abord un modèle .pkl.")
            return
        
        # Mode Image Unique (TOUJOURS, car le batch a son propre bouton maintenant)
        if not self.image_path:
            messagebox.showwarning("Image", "Choisis une image ou un dossier.")
            return

        backend = self.get_backend()
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
            
            # Mettre en cache immédiatement
            if result["success"]:
                self.batch_results[self.image_path] = {
                    'result': self.current_result,
                    'boxes': self.current_boxes
                }
            
            # Calculer le temps écoulé
            elapsed = time.time() - start_time
            
            if result["success"]:
                predictions = result.get("predictions", [])
                
                # Gestion de l'affichage (unique ou multiple)
                if predictions and len(predictions) > 1:
                    # Plusieur visages : on affiche la liste des noms
                    names = sorted(list(set([p["name"] for p in predictions])))
                    display = ", ".join(names)
                    if len(display) > 30: # Tronquer si trop long
                        display = f"{len(predictions)} detected: " + display[:25] + "..."
                    
                    status = "OK (Multiple)"
                    conf = None # Difficile d'afficher une seule confiance
                    
                    self.log(f"[INFO] Multiple identities: {names}")
                
                else:
                    # Cas standard (0 ou 1 visage)
                    pred_name = result["predicted_name"]
                    conf = result["confidence"]
                    display, status, true_prefix = self.decide_display(fname, pred_name)
                    self.log(f"[INFO] filename_prefix={true_prefix}")
                    self.log(f"[INFO] predicted={pred_name}")
                
                self.set_result(display, conf, status)
                self.log(f"[OK] status={status}")
                
                # Mettre à jour les stats pour 1 image
                counts = {'OK': 0, 'UNKNOWN': 0, 'ERROR': 0}
                if "OK" in status:
                    counts['OK'] = 1
                else:
                    counts[status] = 1 # Fallback
                    
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
    
    def start_batch_clicked(self):
        """Lance le traitement batch sur tout le dossier."""
        self._start_batch()

    def _start_batch(self):
        """Démarre le traitement batch via BatchProcessor."""
        if not self.batch_images:
            messagebox.showwarning("Batch", "Aucune image chargée.")
            return

        backend = self.get_backend()
        
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
                    
                    # Caching du résultat batch
                    if self.current_result and self.current_result.get("success"):
                        self.batch_results[payload] = {
                            "result": self.current_result,
                            "boxes": self.current_boxes
                        }
                    
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
