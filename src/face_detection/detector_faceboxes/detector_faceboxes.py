from FaceDetector import FaceDetector
import cv2
import time

if __name__ == '__main__':
    face_detector = FaceDetector()

    cam = cv2.VideoCapture(0)
    ok, frame = cam.read()

    prevTime = 0

    while ok:
        ok, frame = cam.read()
        faces = face_detector.detect_face(frame)
        for face in faces:
            area = face["area"]
            (x, y, x1, y1) = face["coordinates"]
            # print((x, y, x1, y1))
            cv2.rectangle(frame, (x, x1), (y, y1), (0, 25, 255), 7)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Face Detection", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break

    cam.release()