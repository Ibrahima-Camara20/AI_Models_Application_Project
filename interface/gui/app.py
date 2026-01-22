"""
Application principale de reconnaissance faciale avec DeepFace + SVM.
"""
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from interface.core.text_utils import (
    extract_name_prefix_from_filename,
    names_match,
    pretty_name,
    list_images_in_dir
)
from interface.core.model_loader import load_svm_model
from interface.core.predictor import get_embedding, predict_identity
from interface.gui.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    DEFAULT_BACKEND,
    AVAILABLE_BACKENDS,
    QUEUE_POLL_INTERVAL_MS,
    MAX_WORDS_IN_NAME,
    FONT_RESULT_NAME,
    FONT_RESULT_INFO,
    FONT_SECTION_TITLE,
    CANVAS_BG_COLOR,
    TEXT_LOG_WIDTH,
    TEXT_LOG_HEIGHT,
    MODEL_ENTRY_WIDTH,
    BACKEND_COMBO_WIDTH,
    IMG_EXTS
)


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

        # Threading pour le batch processing
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None

        # Variables Tkinter
        self.var_model = tk.StringVar(value="")
        self.var_backend = tk.StringVar(value=DEFAULT_BACKEND)

        # Construction de l'interface
        self._build_ui()
        
        # Démarrage du polling de la queue
        self.after(QUEUE_POLL_INTERVAL_MS, self._poll_queue)

    # =============================
    # Construction de l'interface
    # =============================
    
    def _build_ui(self):
        """Construit l'interface graphique complète."""
        # --- Section Configuration ---
        top = ttk.LabelFrame(self, text="Configuration")
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Modèle (.pkl):").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(top, textvariable=self.var_model, width=MODEL_ENTRY_WIDTH).grid(
            row=0, column=1, sticky="we", padx=6
        )
        ttk.Button(top, text="Parcourir", command=self.pick_model).grid(row=0, column=2, padx=6)

        ttk.Label(top, text="Embedding:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            top, textvariable=self.var_backend,
            values=AVAILABLE_BACKENDS,
            state="readonly", width=BACKEND_COMBO_WIDTH
        ).grid(row=1, column=1, sticky="w", padx=6)

        top.columnconfigure(1, weight=1)

        # --- Section Principale ---
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        # Panneau gauche : Image
        left = ttk.LabelFrame(main, text="Photo")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(left, bg=CANVAS_BG_COLOR, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Boutons d'action
        btns = ttk.Frame(left)
        btns.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(btns, text="Choisir une image", command=self.pick_image).pack(side="left")
        ttk.Button(btns, text="Choisir un dossier", command=self.pick_folder).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(btns, text="Prédire", command=self.predict_one_clicked).pack(
            side="left", padx=(8, 0)
        )
        self.btn_stop = ttk.Button(btns, text="STOP", command=self.stop_batch, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))

        # Barre de progression
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=(0, 6))
        self.lbl_prog = ttk.Label(left, text="0 / 0")
        self.lbl_prog.pack(anchor="w", padx=8)

        # Panneau droit : Résultats
        right = ttk.LabelFrame(main, text="Résultat")
        right.pack(side="right", fill="both", expand=False)

        self.lbl_name = ttk.Label(right, text="—", font=FONT_RESULT_NAME)
        self.lbl_name.pack(anchor="w", padx=12, pady=(16, 6))

        self.lbl_conf = ttk.Label(right, text="Confiance: —", font=FONT_RESULT_INFO)
        self.lbl_conf.pack(anchor="w", padx=12, pady=4)

        self.lbl_status = ttk.Label(right, text="Status: —", font=FONT_RESULT_INFO)
        self.lbl_status.pack(anchor="w", padx=12, pady=4)

        sep = ttk.Separator(right, orient="horizontal")
        sep.pack(fill="x", padx=12, pady=12)

        ttk.Label(right, text="Logs:", font=FONT_SECTION_TITLE).pack(anchor="w", padx=12)
        self.txt = tk.Text(right, width=TEXT_LOG_WIDTH, height=TEXT_LOG_HEIGHT, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self.log("Prêt. 1) Charger modèle .pkl  2) Choisir image/dossier  3) Prédire")

    # =============================
    # Méthodes utilitaires
    # =============================
    
    def log(self, msg: str):
        """Ajoute un message au log."""
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def show_image(self, path: str):
        """Affiche une image sur le canvas."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(CANVAS_WIDTH / w, CANVAS_HEIGHT / h, 1.0)
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
        else:
            self.lbl_conf.config(text=f"Confiance: {conf*100:.4f}%")
        self.lbl_status.config(text=f"Status: {status}")

    def load_model_from_path(self, path: str):
        """Charge un modèle depuis un fichier."""
        self.model_data = load_svm_model(path)
        self.log(f"[OK] Modèle chargé: {os.path.basename(path)}")
        self.log(f"[INFO] Clés: {list(self.model_data.keys())}")

    # =============================
    # Sélection de fichiers
    # =============================
    
    def pick_model(self):
        """Ouvre un dialogue pour choisir le modèle .pkl."""
        p = filedialog.askopenfilename(
            title="Choisir le modèle .pkl",
            filetypes=[("Pickle model", "*.pkl"), ("All files", "*.*")]
        )
        if not p:
            return
        self.var_model.set(p)
        try:
            self.load_model_from_path(p)
        except Exception as e:
            self.model_data = None
            messagebox.showerror("Erreur modèle", str(e))

    def pick_image(self):
        """Ouvre un dialogue pour choisir une image."""
        p = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not p:
            return
        self.image_path = p
        self.show_image(p)
        self.log(f"[OK] Image sélectionnée: {os.path.basename(p)}")

    def pick_folder(self):
        """Ouvre un dialogue pour choisir un dossier et lance le batch."""
        if self.model_data is None:
            messagebox.showwarning("Modèle", "Charge d'abord un modèle .pkl.")
            return

        folder = filedialog.askdirectory(title="Choisir un dossier d'images")
        if not folder:
            return

        images = list_images_in_dir(folder)
        if not images:
            messagebox.showwarning("Dossier", "Aucune image trouvée dans ce dossier.")
            return

        backend = self.var_backend.get().strip()

        self.log(f"\n[RUN] Batch dossier: {folder}")
        self.log(f"[INFO] {len(images)} images trouvées. Backend={backend}")

        self.stop_flag.clear()
        self.btn_stop.config(state="normal")

        self.progress["maximum"] = len(images)
        self.progress["value"] = 0
        self.lbl_prog.config(text=f"0 / {len(images)}")

        self.worker = threading.Thread(
            target=self._batch_worker,
            args=(images, backend),
            daemon=True
        )
        self.worker.start()

    # =============================
    # Logique de prédiction
    # =============================
    
    def decide_display(self, filename: str, predicted_raw: str) -> tuple[str, str, str]:
        """ Décide du nom à afficher en comparant la prédiction avec le nom de fichier. """
        
        true_prefix = extract_name_prefix_from_filename(filename, max_words=MAX_WORDS_IN_NAME)
        ok = names_match(predicted_raw, true_prefix)
        if ok:
            return pretty_name(predicted_raw), "OK", true_prefix
        return "UNKNOWN", "UNKNOWN", true_prefix

    def predict_one_clicked(self):
        """Effectue une prédiction sur l'image sélectionnée."""
        if self.model_data is None:
            messagebox.showwarning("Modèle", "Charge d'abord un modèle .pkl.")
            return
        if not self.image_path:
            messagebox.showwarning("Image", "Choisis une image.")
            return

        backend = self.var_backend.get().strip()
        fname = os.path.basename(self.image_path)

        self.log(f"\n[RUN] Single | Backend={backend} | file={fname}")

        try:
            emb = get_embedding(self.image_path, backend)
            pred_name, conf = predict_identity(self.model_data, emb)

            display, status, true_prefix = self.decide_display(fname, pred_name)
            self.set_result(display, conf, status)

            self.log(f"[INFO] filename_prefix={true_prefix}")
            self.log(f"[INFO] predicted={pred_name}")
            self.log(f"[OK] status={status}")

        except Exception as e:
            self.set_result("ERROR", None, "ERROR")
            self.log(f"[ERROR] {type(e).__name__}: {e}")
            messagebox.showerror("Erreur prédiction", str(e))

    # =============================
    # Traitement batch
    # =============================
    
    def _batch_worker(self, images: list[str], backend: str):
        """
        Worker thread pour le traitement batch.
        
        Args:
            images: Liste des chemins d'images
            backend: Nom du backend d'embedding
        """
        total = len(images)
        for i, img_path in enumerate(images, start=1):
            if self.stop_flag.is_set():
                self.q.put(("log", "[WARN] Batch interrompu par l'utilisateur."))
                break

            fname = os.path.basename(img_path)
            try:
                emb = get_embedding(img_path, backend)
                pred_name, conf = predict_identity(self.model_data, emb)

                display, status, true_prefix = self.decide_display(fname, pred_name)

                self.q.put(("show", img_path))
                self.q.put(("result", (display, conf, status)))
                self.q.put((
                    "log", 
                    f"[{i}/{total}] {fname} | prefix={true_prefix} | pred={pred_name} | {status}"
                ))

            except Exception as e:
                self.q.put(("log", f"[ERROR] {fname}: {type(e).__name__} - {e}"))

            self.q.put(("progress", (i, total)))

        self.q.put(("done", None))

    def stop_batch(self):
        """Arrête le traitement batch en cours."""
        self.stop_flag.set()
        self.btn_stop.config(state="disabled")

    def _poll_queue(self):
        """Traite les messages de la queue du worker thread."""
        try:
            while True:
                kind, payload = self.q.get_nowait()

                if kind == "log":
                    self.log(payload)
                elif kind == "show":
                    self.image_path = payload
                    self.show_image(payload)
                elif kind == "result":
                    name, conf, status = payload
                    self.set_result(name, conf, status)
                elif kind == "progress":
                    i, total = payload
                    self.progress["value"] = i
                    self.lbl_prog.config(text=f"{i} / {total}")
                elif kind == "done":
                    self.btn_stop.config(state="disabled")
                    self.log("[INFO] Batch terminé / prêt.")
        except queue.Empty:
            pass

        self.after(QUEUE_POLL_INTERVAL_MS, self._poll_queue)
