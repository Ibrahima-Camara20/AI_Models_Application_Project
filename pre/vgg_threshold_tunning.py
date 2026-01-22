import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "img/stats/vgg_final_results.csv"
OUTPUT_PLOT = "img/stats/threshold_tuning.png"

def optimize_threshold(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Fichier introuvable : {csv_path}")
        return

    # 1. Chargement et Nettoyage
    df = pd.read_csv(csv_path)
    print(f"📊 Analyse de {len(df)} prédictions...")

    # Détection automatique de la colonne "candidat trouvé"
    if 'best_match' in df.columns:
        col_match = 'best_match'
    elif 'predicted' in df.columns: # Parfois appelé 'predicted' dans les scripts précédents
        col_match = 'predicted'
    elif 'raw_match' in df.columns:
        col_match = 'raw_match'
    else:
        # Fallback : on cherche une colonne qui ressemble
        cols = [c for c in df.columns if 'match' in c or 'pred' in c]
        col_match = cols[0] if cols else 'best_match'
    
    print(f"   ➤ Colonne utilisée pour le match : '{col_match}'")

    # 2. Pré-calcul de la "Vérité Terrain" (Ground Truth)
    # Est-ce que le candidat proposé par l'IA est le bon ? (Indépendamment du seuil)
    # Si Vrai : C'est un potentiel Vrai Positif. Si Faux : C'est un potentiel Faux Positif.
    # Convertit en entiers (1 ou 0)
    y_true_base = (df[col_match] == df['true_label']).astype(int).values
    distances = df['distance'].values

    # 3. Analyse par Balayage (Grid Search) Vectorisé
    thresholds = np.arange(0.30, 0.90, 0.01)
    history_f1 = []
    history_acc = []

    best_thresh = 0
    best_f1 = 0
    best_acc = 0

    # On utilise tqdm pour la barre de progression
    for t in tqdm(thresholds, desc="Optimisation du Seuil"):
        # Vectorisation : On crée le tableau de prédictions en une seule ligne
        # Si la distance est sous le seuil, l'IA dit "OUI" (1), sinon "NON" (0)
        y_pred_at_t = (distances <= t).astype(int)

        # Note: Scikit-learn gère la comparaison entre y_true_base et y_pred_at_t
        f1 = f1_score(y_true_base, y_pred_at_t, zero_division=0)
        acc = accuracy_score(y_true_base, y_pred_at_t)

        history_f1.append(f1)
        history_acc.append(acc)

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_thresh = t

    # 4. Rapport Final
    print("\n" + "="*40)
    print(f"🏆 MEILLEUR SEUIL (Threshold) : {best_thresh:.2f}")
    print(f"📈 F1-Score Max              : {best_f1:.2%}")
    print(f"🎯 Accuracy associée         : {best_acc:.2%}")
    print("="*40)

    # 5. Visualisation Professionnelle
    plt.figure(figsize=(12, 6))
    
    # Courbes
    plt.plot(thresholds, history_f1, label="F1 Score (Compromis)", color='#2196F3', linewidth=2.5)
    plt.plot(thresholds, history_acc, label="Accuracy (Précision)", color='#4CAF50', linestyle='--', linewidth=2)
    
    # Ligne du meilleur seuil
    plt.axvline(x=best_thresh, color='#F44336', linestyle=':', linewidth=2, label=f'Best: {best_thresh:.2f}')
    
    # Zone de confiance (Highlight du sommet)
    plt.axvspan(best_thresh - 0.05, best_thresh + 0.05, color='#F44336', alpha=0.1, label="Zone Optimale")

    # Esthétique
    plt.title(f"Optimisation du Seuil de Reconnaissance\n(Pic de performance à {best_thresh:.2f})", fontsize=14, fontweight='bold')
    plt.xlabel("Valeur du Seuil (Distance)", fontsize=12)
    plt.ylabel("Score de Performance", fontsize=12)
    plt.legend(loc="lower center", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0.30, 0.90)
    plt.ylim(0, 1.05)
    
    # Sauvegarde
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graphique sauvegardé : {OUTPUT_PLOT}")

# --- EXECUTION ---
if __name__ == "__main__":
    optimize_threshold(INPUT_CSV)