import numpy as np
import cv2
from face_detection.face_detector_mp import FaceDetectorMediapipe

class FaceDetectors:
    def __init__(self):
        super(FaceDetectors, self).__init__()

        #Haarcascade
        self.faces_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        # faces_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")
        #Mediapipe
        self.face_detector = FaceDetectorMediapipe()

        #SSD
        self.caffe_model = "models/deploy.prototxt.txt"
        self.caffe_weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.caffe_model, self.caffe_weights)
        self.conf = 0.5



    def face_detection_haar(self, image):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faces_cascade.detectMultiScale(image, scaleFactor = 1.3, minNeighbors=5)
        # print(faces)
        for (x, y, w, h) in faces:
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(image, (x, y), (x+w, y+h), color, stroke)

        return faces, image

    def face_detection_mp(self, image):
        _, faces, l = self.face_detector.infer_image(image)
        for (x_min, y_min, x_max, y_max) in faces:
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, stroke)

        return faces, image

    def face_detection_ssd(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence

            if confidence > self.conf:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype("int"))
                # print(f"box: {faces}")
                (startX, startY, endX, endY) = box.astype("int")
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return faces, image