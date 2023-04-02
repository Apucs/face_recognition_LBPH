import cv2
import mediapipe as mp
import time
import math
from typing import Tuple, Union

class FaceDetectorMediapipe:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int,
                                     image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                            math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        # print(x_px, y_px)
        return x_px, y_px

    def infer_image(self, image):
        self.image = image
        faces=[]

        prevTime = 0

        with self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_detection.process(image)

            # print(results.detections[0].location_data.relative_bounding_box)
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    location = detection.location_data
                    image_rows, image_cols, _ = image.shape

                    relative_bounding_box = location.relative_bounding_box

                    rect_start_point = self._normalized_to_pixel_coordinates(relative_bounding_box.xmin,
                                                                        relative_bounding_box.ymin,
                                                                        image_cols, image_rows)
                    rect_end_point = self._normalized_to_pixel_coordinates(relative_bounding_box.xmin + relative_bounding_box.width,
                                                                    relative_bounding_box.ymin + relative_bounding_box.height,
                                                                    image_cols, image_rows)

                    # print(rect_start_point, rect_end_point)
                    x1, y1 = rect_start_point if rect_start_point else (0, 0)
                    x2, y2 = rect_end_point if rect_end_point else (0, 0)

                    faces.append([x1, y1, x2, y2])
                    # self.mp_drawing.draw_detection(self.image, detection)
                    cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img_to_show = self.image
            return img_to_show, faces, len(faces)


