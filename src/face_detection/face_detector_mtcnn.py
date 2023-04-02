import os
import torch
from facenet_pytorch import MTCNN,InceptionResnetV1
import cv2 as cv
from PIL import Image
import pickle
import numpy as np

class FaceDetector:
    def __init__(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


    def infer_image(self, image):
        self.image=image
        bbox = []
        labels = []

        image = Image.fromarray(self.image[..., ::-1])
        boxes, _ = self.mtcnn.detect(image)
        re = isinstance(boxes, np.ndarray)

        if re==False:
            print("No face detected")
            return self.image, None, None

        for box in boxes:
            box = box.tolist()
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            image_to_write = cv.rectangle(self.image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            image_show = cv.resize(image_to_write, (self.image.shape[1], self.image.shape[0]))
            # cv.imshow('frame', image_to_write)

            # cv.imwrite(('output/img1.jpg'), image_show)
            # bbox = [x_min, y_min, x_max, y_max]
            bbox.append([x_min,y_min,x_max,y_max])

        return image_show, bbox, len(bbox)