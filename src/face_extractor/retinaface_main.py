"""
Module de détection RetinaFace - Point d'entrée
Extrait les visages à partir des crops de personnes (working/).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from face_extractor import RetinaFaceDetector


def main():
    WORKING = "working"
    FACES = "faces_dataset"
    INPUT_METADATA = "metadata.json"
    OUTPUT_METADATA = "faces_metadata.json"
    
    detector = RetinaFaceDetector()
    stats = detector.process_working_folder(
        WORKING, FACES, 
        INPUT_METADATA, OUTPUT_METADATA
    )


if __name__ == "__main__":
    main()
