"""
Module de détection YOLO - Point d'entrée simplifié
===================================================

Utilise le package yolo pour la détection de personnes.

Usage:
    python src/yolo/yolo_main.py

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yolo import YOLODetector
from display_utils import display_yolo_evaluation


def main():
    """
    Point d'entrée du script YOLO.
    """
    # Configuration
    DATASET = "datasets"
    WORKING = "working"
    METADATA = "metadata.json"
    
    print("\n" + "="*70)
    print("  DÉTECTION DE PERSONNES AVEC YOLO")
    print("="*70)
    print(f"\n Entrée  : {DATASET}/")
    print(f" Sortie  : {WORKING}/")
    print(f" Métadonnées : {METADATA}")
    
    # Créer le détecteur et traiter le dataset
    detector = YOLODetector('yolo11n.pt')
    stats = detector.process_dataset(DATASET, WORKING, METADATA)
    
    # Afficher l'évaluation
    display_yolo_evaluation(stats)


if __name__ == "__main__":
    main()
