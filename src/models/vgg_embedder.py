import torch
import torch.nn as nn
from torchvision import models


def charger_vgg_embedder(device: torch.device):
    """
    Charge VGG16 pré-entraîné et retourne un modèle qui sort des embeddings.
    Embedding par défaut: 4096 (sortie de fc2).
    """
    # Charge le modèle avec les poids pré-entraînés sur ImageNet
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # On garde les features + avgpool + classifier sans la dernière couche (1000 classes)
    # vgg.classifier est une Sequential. On prend tout sauf le dernier élément.
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])  # supprime la dernière Linear (fc3)

    vgg.eval()
    vgg.to(device)
    return vgg
