"""
Module de calcul des statistiques YOLO.
Gère l'accumulation et le calcul des métriques.
"""

def create_stats_dict():
    """
    Crée un dictionnaire de statistiques vide pour YOLO.
   
    """
    return {
        'total_images': 0,
        'images_with_detection': 0,
        'images_without_detection': 0,
        'total_persons_detected': 0,
        'total_crops_saved': 0,
        'multiple_persons_images': 0,
        # Métriques avancées
        'confidence_scores': [],
        'box_sizes': [],
        'box_ratios': [],
        'low_confidence_count': 0,
        'abnormal_boxes_count': 0
    }


def update_detection_stats(stats, num_persons, confidence_scores):
    """
    Met à jour les statistiques après une détection.
  
    """
    if num_persons > 0:
        stats['images_with_detection'] += 1
        stats['total_persons_detected'] += num_persons
        
        if num_persons > 1:
            stats['multiple_persons_images'] += 1
        
        stats['confidence_scores'].extend(confidence_scores)
    else:
        stats['images_without_detection'] += 1


def update_box_stats(stats, confidence, box_width, box_height):
    """
    Met à jour les statistiques pour une bounding box.
    
    """
    # Score de confiance
    if confidence < 0.5:
        stats['low_confidence_count'] += 1
    
    # Taille de la box
    box_size = box_width * box_height
    stats['box_sizes'].append(box_size)
    
    # Ratio de la box
    box_ratio = box_width / box_height if box_height > 0 else 0
    stats['box_ratios'].append(box_ratio)
    
    # Détection de boxes anormales
    if box_ratio < 0.3 or box_ratio > 3.0:
        stats['abnormal_boxes_count'] += 1


def finalize_stats(stats):
    """
    Calcule les métriques finales.
    
    """
    if stats['total_images'] > 0:
        stats['detection_rate_%'] = round(
            (stats['images_with_detection'] / stats['total_images']) * 100, 2
        )
        
    if stats['images_with_detection'] > 0:
        stats['avg_persons_per_image'] = round(
            stats['total_persons_detected'] / stats['images_with_detection'], 2
        )
    else:
        stats['avg_persons_per_image'] = 0
    
    return stats
