import os
import cv2.face
from PIL import Image
import numpy as np
from face_detectors import FaceDetectors
import pickle


class Train_Faces:
    def __init__(self):
        super(Train_Faces, self).__init__()

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.BASE_DIR, "images")
        self.detector = FaceDetectors()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def train(self):
        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, dirs, files in os.walk(self.image_dir):
            # print(f"root = {root}, dirs = {dirs}, files = {files}")
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path)).replace(" ", "-")
                    # print(label, path)

                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id+=1

                    id_ = label_ids[label]

                    # image = cv2.imread(path)
                    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # image_array = np.array(gray, 'uint8')
                    pil_image_gray =Image.open(path).convert("L")  #grayscale
                    final_image = pil_image_gray.resize((416, 416), Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")

                    faces, _ = self.detector.face_detection_mp(image_array)
                    # print(path, faces)
                    # print(f"faces: {faces}")
                    if len(faces)>0:
                        for (x_min, y_min, x_max, y_max) in faces:
                            roi = image_array[y_min:y_max, x_min:x_max]
                            # print(roi.shape, roi)
                            if len(roi)>0:
                                x_train.append(roi)
                                y_labels.append(id_)

        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        # print(x_train, type(y_labels))
        self.recognizer.train(x_train, np.array(y_labels))
        self.recognizer.save("trainner.yml")




