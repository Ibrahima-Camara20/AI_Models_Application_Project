"""
Package interface pour la reconnaissance faciale avec DeepFace + SVM.

Ce package fournit une application GUI complète pour :
- Charger un modèle SVM pré-entraîné
- Effectuer des prédictions sur des images individuelles
- Traiter des dossiers d'images en batch
- Comparer automatiquement les prédictions avec les noms de fichiers

Usage:
    from interface import FacePredictApp
    
    app = FacePredictApp()
    app.mainloop()
"""

from src.gui.app import FacePredictApp

__version__ = "2.0.0"
__all__ = ["FacePredictApp"]
