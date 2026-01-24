from pre.yolo_detection import yolo_detector
from pre.retinaface_extraction import extract_faces
from pre.svm_prediction import identifier_visage


def main():
    try:
        #yolo_detector(input_dir="test_images/", output_dir="working/")
        #extract_faces(input_dir="working/", output_dir="faces_extraction/")
        identifier_visage(input_dir="faces_extraction/", model_path="models/svm_vgg_face.pkl", model_name="VGG-Face")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()