from pre.yolo_detection import yolo_detector
from pre.retinaface_extraction import extract_faces
from extract_embedding import embedding_extractor
from pre.svm_prediction import svm_prediction

if __name__ == "__main__":
    #yolo_detector("data_test", "working")
    #extract_faces("working", "faces_dataset")
    embedding_extractor("VGGFace", "faces_dataset", "embeddings_dataset")
    svm_prediction()