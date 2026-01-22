import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

""" --- CONFIGURATION de kaggle  ---
INPUT_DIR = "/kaggle/input/pins-face-recognition/105_classes_pins_dataset"
OUTPUT_DIR = "/kaggle/working/version-finale/working"
SAVE_DIR = "/kaggle/working/version-finale"
"""
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def list_images_in_dir(root):
    imgs = []
    for r, d, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                imgs.append(os.path.splitext(f)[0])  # nom sans extension
    return imgs

def evaluate_yolo_performance(INPUT_DIR, OUTPUT_DIR, SAVE_DIR):
    print("--- DÉBUT DE L'ÉVALUATION YOLO ---")

    # 1) Ground truth: tous les noms d'images attendus (sans extension)
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"INPUT_DIR introuvable: {INPUT_DIR}")
    expected_files = set(list_images_in_dir(INPUT_DIR))
    total_expected = len(expected_files)
    print(f"[INFO] Images totales attendues (ground truth) : {total_expected}")

    # 2) Ce que YOLO a généré / détecté (on prend le nom original extrait)
    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"OUTPUT_DIR introuvable: {OUTPUT_DIR}")
    generated_files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
    print(f"[INFO] Fichiers trouvés dans OUTPUT_DIR : {len(generated_files)}")

    # Extraire le nom original : tolérance sur "-person" ou "-person-"
    found_originals = set()
    original_to_file = {}  # pour afficher exemples (garder 1 file par original)
    for fname in generated_files:
        low = fname.lower()
        if "-person" in low:
            original_name = fname.split("-person")[0]
            found_originals.add(original_name)
            # garder la première occurrence utile
            if original_name not in original_to_file:
                # vérifier que c'est bien une image (sinon essayer ajouter extension)
                original_to_file[original_name] = os.path.join(OUTPUT_DIR, fname)

    total_found = len(found_originals)
    print(f"[INFO] Noms originaux détectés par YOLO : {total_found}")

    # 3) Calculs des métriques (au niveau "nom original / image")
    TP_set = expected_files.intersection(found_originals)  # attendues & trouvées
    FN_set = expected_files.difference(found_originals)     # attendues mais pas trouvées
    FP_set = found_originals.difference(expected_files)     # trouvées mais pas attendues (potentiels FP)
    TP = len(TP_set)
    FN = len(FN_set)
    FP = len(FP_set)

    # Construire vecteurs pour la matrice de confusion :
    # On considère l'univers = union(expected_files, found_originals)
    universe = sorted(set(expected_files).union(found_originals))
    y_true = [1 if u in expected_files else 0 for u in universe]
    y_pred = [1 if u in found_originals else 0 for u in universe]

    # Metrics classiques
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Accuracy calculée sur l'univers considéré (TP+TN) / total
    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "="*40)
    print(" RÉSULTATS STATISTIQUES (au niveau 'nom image') ")
    print("="*40)
    print(f"Vrais positifs  (TP) : {TP}")
    print(f"Faux négatifs   (FN) : {FN}")
    print(f"Faux positifs   (FP) : {FP}")
    print("-"*40)
    print(f"Précision (Precision) : {precision:.2%}")
    print(f"Rappel   (Recall)    : {recall:.2%}")
    print(f"F1-score               : {f1:.2%}")
    print(f"Accuracy (sur univers) : {accuracy:.2%}")
    print("="*40)

    # 4) Matrice de confusion (labels [1,0] -> row true=1 then true=0, col pred=1 then pred=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Prédit: Humain', 'Prédit: Non Humain'],
                yticklabels=['Vrai: Humain', 'Vrai: Non Humain'])
    plt.title(f'Matrice de Confusion YOLO\nPrecision: {precision:.2%}  Recall: {recall:.2%}  F1: {f1:.2%}')
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction YOLO')
    cm_path = os.path.join(SAVE_DIR, "yolo_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[INFO] Matrice de confusion sauvegardée -> {cm_path}")
    plt.show()

    # 5) Visualisation qualitative : afficher jusqu'à 5 découpages réels (fichiers générés)
    sample_originals = list(TP_set)[:5]  # succès (TP)
    if len(sample_originals) == 0:
        print("[VISUEL] Aucun succès (TP) trouvé pour afficher d'exemples.")
    else:
        print("\n[VISUEL] Exemples de découpages réussis (TP) :")
        n = len(sample_originals)
        fig, axes = plt.subplots(1, max(1, n), figsize=(4 * max(1, n), 4))
        if n == 1:
            axes = [axes]
        for i, orig in enumerate(sample_originals):
            img_path = original_to_file.get(orig)
            if img_path and os.path.exists(img_path):
                img = plt.imread(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(orig)
            else:
                axes[i].text(0.5, 0.5, "Fichier image introuvable", ha='center')
                axes[i].axis('off')
        example_path = os.path.join(SAVE_DIR, "yolo_exemples_tp.png")
        plt.savefig(example_path)
        print(f"[INFO] Exemples sauvegardés -> {example_path}")
        plt.show()
if __name__ == "__main__":
    evaluate_yolo_performance()