"""
Utilitaire pour annoter les images avec les bounding boxes et prédictions.

Permet de dessiner :
- Bounding boxes YOLO (personnes détectées)
- Bounding boxes RetinaFace (visages détectés)
- Texte de prédiction
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.gui.constants import ANNOTATION_FONT_SIZE


def draw_yolo_boxes(image, boxes, color=(34, 197, 94), thickness=3):
    """Dessine les bounding boxes YOLO (personnes) sur une image."""
    # Convertir en PIL si nécessaire
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Créer une copie pour ne pas modifier l'original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            # Dessiner le rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            
            # Si on a la confiance, l'afficher
            if len(box) >= 5:
                conf = box[4]
                text = f"Person {conf:.2f}"
                # Fond pour le texte
                text_bbox = draw.textbbox((x1, y1 - 20), text)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1 - 20), text, fill=(255, 255, 255))
    
    return img_copy


def draw_retinaface_boxes(image, boxes, color=(59, 130, 246), thickness=2):
    """Dessine les bounding boxes RetinaFace (visages) sur une image."""
    # Convertir en PIL si nécessaire
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Créer une copie pour ne pas modifier l'original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            # Dessiner le rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            
            # Si on a la confiance, l'afficher
            if len(box) >= 5:
                conf = box[4]
                text = f"Face {conf:.2f}"
                # Fond pour le texte
                text_bbox = draw.textbbox((x1, y1 - 20), text)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1 - 20), text, fill=(255, 255, 255))
    
    return img_copy


def annotate_prediction(image, predicted_name, confidence, box=None, show_confidence=True):
    """Ajoute le nom prédit et la confiance sur l'image."""
    # Convertir en PIL si nécessaire
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Texte à afficher
    if show_confidence and confidence is not None:
        text = f"{predicted_name} ({confidence*100:.1f}%)"
    else:
        text = predicted_name
    
    # Déterminer la couleur selon la confiance
    if confidence >= 0.8:
        bg_color = (34, 197, 94)  # Vert
    elif confidence >= 0.5:
        bg_color = (245, 158, 11)  # Jaune/Orange
    else:
        bg_color = (239, 68, 68)   # Rouge
    
    # Calculer la position
    try:
        # Essayer d'utiliser une police plus grande
        font = ImageFont.truetype("arial.ttf", ANNOTATION_FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    img_width, img_height = img_copy.size
    
    
    # Calculer la position
    if box:
        # Afficher juste au-dessus de la boîte du visage
        x1, y1, x2, y2 = box
        x = x1
        y = y1 - text_height - 10
        # S'assurer qu'on ne sort pas de l'image (haut)
        if y < 0:
            y = y2 + 10
    else:
        # Position par défaut (centré haut)
        x = (img_width - text_width) // 2
        y = 20
    
    # Dessiner le fond
    padding = 10
    draw.rectangle(
        [x - padding, y - padding, 
         x + text_width + padding, y + text_height + padding],
        fill=bg_color
    )
    
    # Dessiner le texte
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img_copy


def draw_all_annotations(image, yolo_boxes=None, retinaface_boxes=None, 
                        predicted_name=None, confidence=None, predictions=None):
    """Dessine toutes les annotations sur une image."""
    # Convertir en PIL si nécessaire
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    result = image.copy()
    
    # Dessiner les boxes YOLO (en vert)
    if yolo_boxes:
        result = draw_yolo_boxes(result, yolo_boxes)
    
    # Dessiner les boxes RetinaFace (en bleu)
    if retinaface_boxes:
        result = draw_retinaface_boxes(result, retinaface_boxes)
    
    # Ajouter les prédictions multiples (si disponibles)
    if predictions:
        # Masquer la confiance s'il y a plusieurs personnes pour éviter la surcharge
        show_conf = (len(predictions) == 1)
        
        for pred in predictions:
            name = pred.get("name")
            conf = pred.get("confidence")
            box = pred.get("box")
            if name and conf is not None:
                result = annotate_prediction(result, name, conf, box, show_confidence=show_conf)
    
    # Rétro-compatibilité pour affichage unique (si pas de liste)
    elif predicted_name and confidence is not None:
        result = annotate_prediction(result, predicted_name, confidence)
    
    return result


def cv2_to_pil(cv2_image):
    """Convertit une image OpenCV (BGR) en PIL (RGB)."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(pil_image):
    """Convertit une image PIL (RGB) en OpenCV (BGR)."""
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    # Test de l'annotateur
    import os
    
    # Créer une image de test
    test_img = np.ones((600, 800, 3), dtype=np.uint8) * 240
    test_img = Image.fromarray(test_img)
    
    # Boxes YOLO fictives
    yolo_boxes = [
        (100, 150, 350, 500, 0.95),  # (x1, y1, x2, y2, confidence)
        (450, 100, 700, 450, 0.88),
    ]
    
    # Boxes RetinaFace fictives
    retinaface_boxes = [
        (150, 180, 280, 310, 0.99),
        (500, 130, 630, 260, 0.97),
    ]
    
    # Annoter l'image
    annotated = draw_all_annotations(
        test_img,
        yolo_boxes=yolo_boxes,
        retinaface_boxes=retinaface_boxes,
        predicted_name="Emma Watson",
        confidence=0.92
    )
    
    # Sauvegarder pour vérification
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    annotated.save(os.path.join(output_dir, "test_annotation.jpg"))
    print(f"Image annotée sauvegardée dans {output_dir}/test_annotation.jpg")
