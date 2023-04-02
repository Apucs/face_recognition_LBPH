import cv2
from face_detectors import FaceDetectors
import pickle
class Recognition:
    def __init__(self):
        super(Recognition, self).__init__()

        self.detector = FaceDetectors()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainner.yml")
        self.labels = {}
        with open("labels.pickle", "rb") as f:
            org_labels = pickle.load(f)
            self.labels = {v:k for  k,v in org_labels.items()}

    def recognition(self):
        cap = cv2.VideoCapture(0)
        while(True):
            #Capture frame-by-frame
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # faces, image = face_detection_mp(frame.copy())
            faces, _ = self.detector.face_detection_mp(gray.copy())
            for (x_min, y_min, x_max, y_max) in faces:
                roi_gray = gray[y_min:y_max, x_min:x_max]
                roi_color = frame[y_min:y_max, x_min:x_max]

                # print(f"roi gray: {roi_gray}")
                if len(roi_gray)!=0:
                    id_, conf = self.recognizer.predict(roi_gray)
                    print(f"Confidence: {conf}")
                    if conf>=45 and conf<=80:
                        print(id_, self.labels[id_])
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        name = self.labels[id_]
                        color = (0, 0, 255)
                        stroke = 2
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), stroke)
                        cv2.putText(frame, name, (x_min, y_min-5), font, 1, color, stroke, cv2.LINE_AA)

            #Display the resulting frame
            cv2.imshow("image", frame)
            if cv2.waitKey(20) & 0xFF ==  ord('q'):
                break

        #When everthing done, release the capture
        cap.release()
        cv2.destroyAllWindows()