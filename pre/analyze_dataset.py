import os
from collections import Counter

dossier_courant = os.path.dirname(os.path.abspath(__file__))
dossier_data = os.path.join(
    dossier_courant,
    "..",
    "data",
    "105_classes_pins_dataset"
)

def analyze_dataset(dossier_data):
    nombre_de_classes = Counter()
    total_images = 0

    for class_name in os.listdir(dossier_data):
        chemin_de_la_classe = os.path.join(dossier_data, class_name)

        if not os.path.isdir(chemin_de_la_classe):
            continue

        images = [
            f for f in os.listdir(chemin_de_la_classe)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        nombre_de_classes[class_name] = len(images)
        total_images += len(images)

    return nombre_de_classes, total_images


if __name__ == "__main__":

    nombre_de_classes, total_images = analyze_dataset(dossier_data)

    print("\nAnalyse de données")
    print("-" * 40)
    print(f"Nombre de classes (célébrités): {len(nombre_de_classes)}")
    print(f"Nombre total d'images: {total_images}")
    print("-" * 40)

    print("Nombre d'images par classe (premiers 10):")
    for cls, count in nombre_de_classes.most_common(10):
        print(f"{cls:30s} -> {count}")

    print("-" * 40)
    print(f"Min images par classe: {min(nombre_de_classes.values())}")
    print(f"Max images par classe: {max(nombre_de_classes.values())}")
    print(
        f"Moyenne images par classe: "
        f"{total_images / len(nombre_de_classes):.2f}"
    )
