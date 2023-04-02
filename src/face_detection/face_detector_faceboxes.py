from face_detection.detector_faceboxes.FaceDetector import FaceDetector
import cv2
import time

class FaceDetectorFaceboxes:
    def __init__(self):
        self.face_detector = FaceDetector()

    def infer_image(self, image):
        self.image = image

        prevTime = 0
        detected_faces = self.face_detector.detect_face(self.image)

        faces = []

        for face in detected_faces:
            # print(face)
            area = face["area"]
            (x1, y1, x2, y2) = face["coordinates"]
            faces.append(list(face["coordinates"]))
            # print((x, y, x1, y1))
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # img_show = cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 7)
            # cv2.putText(self.image, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # cv2.imshow("Face Detection", self.image)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # cv2.imshow("deteced", self.image)

        # return img_show, faces, len(faces)
        return self.image, faces, len(faces)



