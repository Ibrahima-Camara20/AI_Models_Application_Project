"""
Module de validation des bounding boxes YOLO.
Contient toutes les validations pour garantir la qualité des détections.
"""

def validate_bounding_box(x1, y1, x2, y2, img, img_name, min_size=10):
    
    # VALIDATION 1 : Vérifier que les coordonnées sont cohérentes
    if x1 >= x2 or y1 >= y2:
        print(f"  [SKIP] Box invalide pour {img_name}: x1={x1} x2={x2} y1={y1} y2={y2}")
        return (False, x1, y1, x2, y2, 0, 0)
    
    # VALIDATION 2 : Vérifier que les coordonnées sont dans l'image
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # VALIDATION 3 : Vérifier que le crop n'est pas trop petit
    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < min_size or crop_height < min_size:
        print(f"  [SKIP] Crop trop petit pour {img_name}: {crop_width}x{crop_height}px")
        return (False, x1, y1, x2, y2, crop_width, crop_height)
    
    return (True, x1, y1, x2, y2, crop_width, crop_height)


def validate_crop(crop, img_name):
   
    if crop.size == 0:
        print(f"  [SKIP] Crop vide pour {img_name}")
        return False
    return True


def is_abnormal_box(width, height, min_ratio=0.3, max_ratio=3.0):
    """
    Détecte si une bounding box a un ratio anormal.
    
    """
    if height == 0:
        return True
    
    ratio = width / height
    return ratio < min_ratio or ratio > max_ratio
