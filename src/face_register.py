import os
import cv2
import numpy as np


class Register:
    def __init__(self):
        super(Register, self).__init__()

        # SSD
        self.caffe_model = "models/deploy.prototxt.txt"
        self.caffe_weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.caffe_model, self.caffe_weights)
        self.conf = 0.5

    def register(self):
        name = input("Input your name:")
        reg_path = os.path.join("images", name)
        print(reg_path)

        if not os.path.exists(reg_path):
            os.makedirs(reg_path)

        frame_count = 0
        cap = cv2.VideoCapture(0)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            image = frame.copy()

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
                    cv2.putText(image, str(frame_count+1), (startX, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

                    frame_count += 1

                    image_name = reg_path + "/" + name + "_" + str(frame_count+1)+".jpg"
                    cv2.imwrite(image_name, frame)

            # Display the resulting frame
            cv2.imshow("image", image)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            elif frame_count >=50:
                break

        # When everthing done, release the capture
        cap.release()
        cv2.destroyAllWindows()



#
# r = Register()
# r.register()