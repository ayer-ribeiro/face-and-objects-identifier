"""Identify faces and objects in a video or camera capture"""

import cv2
import identifiers

def get_video_capture(file_name):
    """Returns an object of type VideoCapture"""

    if file_name is None:
        return cv2.VideoCapture(0)

    return cv2.VideoCapture(file_name)

face_identifier = identifiers.FaceIdentifier("haarcascade_frontalface_default.xml", (0,255,255))
face_identifier2 = identifiers.FaceIdentifier("haarcascade_frontalface_alt2.xml", (255,255,0))

blue_color_range = identifiers.BLUE_COLOR_RANGE
blue_object_identifier = identifiers.ColorObjectIdentifier(blue_color_range)

red_color_range = identifiers.RED_COLOR_RANGE
red_object_identifier = identifiers.ColorObjectIdentifier(red_color_range)

hard_coded_color_range = identifiers.ColorRange(color_light = [125,125,125], color_dark = [25,25,25])
custom_color_object_identifier = identifiers.ColorObjectIdentifier(hard_coded_color_range)

videoCapture = get_video_capture(None)

while True:
    (sucess, frame) = videoCapture.read()
    if not sucess: #reading failure or video ended
        break

    blue_object_identifier.identify_from_frame(frame)
    red_object_identifier.identify_from_frame(frame)
    custom_color_object_identifier.identify_from_frame(frame)

    face_identifier.identify_from_frame(frame)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

    face_identifier2.identify_from_frame(frame)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

videoCapture.release()
cv2.destroyAllWindows()
