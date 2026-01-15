from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def creer_transformations(taille=224):
    return transforms.Compose([
        transforms.Resize((taille, taille)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def creer_dataloaders(dossier_splits="data_splits", batch_size=32, nb_workers=0):
    """
    Charge train/val/test depuis la racine du projet.
    """
    # racine du projet = .../AI_Models_Apllication_Project
    racine_projet = Path(__file__).resolve().parents[2]
    racine = (racine_projet / dossier_splits).resolve()

    train_dir = racine / "train"
    val_dir = racine / "val"
    test_dir = racine / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(
                f"Dossier introuvable : {d}\n"
                f"Vérifie que tu as bien : {racine_projet / dossier_splits}/train|val|test"
            )

    tfm = creer_transformations(taille=224)

    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)
    test_ds = datasets.ImageFolder(test_dir, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nb_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nb_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nb_workers)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx
