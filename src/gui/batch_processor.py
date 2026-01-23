"""
Module de traitement batch pour l'application de reconnaissance faciale.

Gère l'exécution asynchrone du traitement batch avec threading et queue.
"""
import queue
import threading
import os
import time

from src.core.pipeline import run_pipeline_single


class BatchProcessor:
    """
    Gestionnaire de traitement batch pour plusieurs images.
    
    Exécute le pipeline de reconnaissance faciale en arrière-plan sur
    une liste d'images et communique les résultats via une queue.
    """
    
    def __init__(self, app):
        """Initialise le processeur batch."""
        self.app = app
        self.worker = None
        self.stop_flag = threading.Event()
    
    def start(self, images: list[str], model_data: dict, backend: str):
        """Démarre le traitement batch."""
        if not images:
            return
        
        self.stop_flag.clear()
        
        self.worker = threading.Thread(
            target=self._worker,
            args=(images, model_data, backend),
            daemon=True
        )
        self.worker.start()
    
    def stop(self):
        """Arrête le traitement batch en cours."""
        self.stop_flag.set()
    
    def _worker(self, images: list[str], model_data: dict, backend: str):
        """Worker thread pour le traitement batch."""
        total = len(images)
        counts = {'OK': 0, 'UNKNOWN': 0, 'ERROR': 0}
        
        for i, img_path in enumerate(images, start=1):
            # Vérifier si arrêt demandé
            if self.stop_flag.is_set():
                self.app.q.put(("log", "[WARN] Batch interrompu par l'utilisateur."))
                break
            
            fname = os.path.basename(img_path)
            
            # Réinitialiser le widget de pipeline pour chaque image
            self.app.q.put(("reset_pipeline", None))
            
            # Callbacks pour communication via queue
            def batch_log(msg):
                self.app.q.put(("log", msg))
            
            def batch_status(stage, status, count=None):
                self.app.q.put(("pipeline_status", (stage, status, count)))
            
            try:
                # Exécuter le pipeline complet
                # Nettoyer SEULEMENT au début du batch (1ère image)
                do_cleanup = (i == 1)
                
                result = run_pipeline_single(
                    img_path,
                    model_data,
                    backend,
                    log_callback=batch_log,
                    status_callback=batch_status,
                    cleanup=do_cleanup
                )
                
                if result["success"]:
                    pred_name = result["predicted_name"]
                    conf = result["confidence"]
                    stats = result["stats"]
                    boxes = result.get("boxes", {"yolo_boxes": [], "retinaface_boxes": []})
                    
                    # Décider du nom à afficher
                    display, status, true_prefix = self.app.decide_display(fname, pred_name)
                    
                    if status in counts:
                        counts[status] += 1
                    
                    # Envoyer les résultats via la queue
                    # IMPORTANT: Stocker AVANT d'afficher
                    self.app.q.put(("store_result", result))
                    self.app.q.put(("store_boxes", boxes))
                    self.app.q.put(("show", img_path))
                    self.app.q.put(("result", (display, conf, status)))
                    
                    # Envoyer les stats
                    self.app.q.put(("batch_stats", (i, total, counts.copy(), conf)))
                    
                    self.app.q.put((
                        "log",
                        f"[{i}/{total}] {fname} | P:{stats['persons_detected']} "
                        f"F:{stats['faces_detected']} | pred={pred_name} | {status}"
                    ))
                else:
                    counts['ERROR'] += 1
                    # Pipeline échoué
                    self.app.q.put(("log", f"[{i}/{total}] {fname} | ÉCHEC: {result['error']}"))
                    self.app.q.put(("result", ("ERROR", None, "ERROR")))
                    
                    # Envoyer les stats (conf=None)
                    self.app.q.put(("batch_stats", (i, total, counts.copy(), None)))
            
            except Exception as e:
                self.app.q.put(("log", f"[ERROR] {fname}: {type(e).__name__} - {e}"))
            
            # Mettre à jour la progression
            self.app.q.put(("progress", (i, total)))
        
        # Traitement terminé
        self.app.q.put(("done", None))
