"""
Package YOLO pour la détection de personnes.

Contient tous les modules nécessaires à la détection YOLO.
"""

from .yolo_detector import YOLODetector
from .box_validator import validate_bounding_box, validate_crop, is_abnormal_box
from .yolo_stats import (
    create_stats_dict,
    update_detection_stats,
    update_box_stats,
    finalize_stats
)

__all__ = [
    'YOLODetector',
    'validate_bounding_box',
    'validate_crop',
    'is_abnormal_box',
    'create_stats_dict',
    'update_detection_stats',
    'update_box_stats',
    'finalize_stats',
]
