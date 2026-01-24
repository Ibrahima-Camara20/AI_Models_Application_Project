import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report

def evaluate_retinaface_performance(input_dir, output_dir):
    print("--- GÉNÉRATION DU RAPPORT SCIKIT-LEARN ---")

    # 1. Préparation des données
    # On récupère la liste des fichiers attendus (Input)
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # On récupère la liste des fichiers réussis (Output)
    output_files = set(os.listdir(OUTPUT_DIR))

    y_true = []
    y_pred = []

    # 2. Construction des vecteurs de prédiction
    for filename in input_files:
        # Vérité Terrain : On sait qu'il y a une personne (car c'est un crop YOLO)
        y_true.append(1) 
        
        # Prédiction : Est-ce que le fichier existe dans le dossier de sortie ?
       
        if filename in output_files:
            y_pred.append(1) # Trouvé
        else:
            y_pred.append(0) # Raté

    # 3. Calcul des Métriques avec Scikit-Learn
    acc = accuracy_score(y_true, y_pred)
    # Le Recall est la métrique la plus importante ici : "Sur tous les humains, combien de visages ai-je trouvés ?"
    rec = recall_score(y_true, y_pred, zero_division=0) 

    print("\n" + "="*40)
    print(f" RÉSULTATS STATISTIQUES (Phase 1.2)")
    print("="*40)
    print(f"Images Totales (YOLO)  : {len(y_true)}")
    print(f"Visages Extraits       : {sum(y_pred)}")
    print(f"Images Perdues         : {len(y_true) - sum(y_pred)}")
    print("-" * 40)
    print(f"Précision (Accuracy)   : {acc:.2%}")
    print(f"Rappel (Recall)        : {rec:.2%}")
    print("="*40)

    # 4. Matrice de Confusion
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

   
    plt.figure(figsize=(6, 5))
    labels = ['Non Détecté', 'Détecté']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels,
                yticklabels=['(Pas d\'humain)', 'Humain présent'])

    plt.title(f'Matrice de Confusion RetinaFace\nRecall: {rec:.1%}')
    plt.xlabel('Prédiction du Modèle')
    plt.ylabel('Vérité Terrain')
    plt.savefig("/kaggle/working/version-finale/retinaface_evaluation.png")
    plt.show()

    # 5. Rapport détaillé texte
    print("\n--- RAPPORT DÉTAILLÉ ---")
    print(classification_report(y_true, y_pred, target_names=['Échec', 'Succès'], labels=[0, 1], zero_division=0))

if __name__ == "__main__":
    evaluate_retinaface_performance()