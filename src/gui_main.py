#!/usr/bin/env python3
"""
Point d'entrée pour l'application de reconnaissance faciale.

Lance l'interface graphique Tkinter. Dans le dossier principal tapez: python interface.main
"""

from src.gui.app import FacePredictApp


def main():
    """Lance l'application de reconnaissance faciale."""
    app = FacePredictApp()
    app.mainloop()


if __name__ == "__main__":
    main()
