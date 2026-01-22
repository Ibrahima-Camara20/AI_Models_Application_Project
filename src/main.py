from pre.yolo_detection import yolo_detector
from pre.retinaface_extraction import extract_faces

if __name__ == "__main__":
    #yolo_detector("data_test", "working")
    extract_faces("working", "faces_dataset")

