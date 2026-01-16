from .yolo import extract_person_bounding_boxes
from .extract_faces import extract_faces
from .identify_vgg import recognize_celebrities

dataset_path = "../dataset"
working_dir = "../working" 
faces_dataset_dir = "../faces_dataset" 
confidence_threshold = 0.9
db_path = "../celebrity_db"

if __name__ == "__main__":
    print("=" * 60)
    print("PIPELINE DE RECONNAISSANCE DE CÉLÉBRITÉS")    

    print("\n[ÉTAPE 1/3] Extraction des personnes avec YOLO...")
    extract_person_bounding_boxes(dataset_path, working_dir)
    
    print("[ÉTAPE 2/3] Extraction des visages avec RetinaFace...")
    extract_faces(working_dir, faces_dataset_dir, confidence_threshold=confidence_threshold)
    
    print("[ÉTAPE 3/3] Reconnaissance des célébrités avec VGG-Face...")
    recognize_celebrities(faces_dataset_dir, db_path)    
    print("=" * 60)
