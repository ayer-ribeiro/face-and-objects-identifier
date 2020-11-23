"""Face and objects identifiers"""

import cv2
import numpy as np

WINDOW_NAME = "Face and objects Identifier"

class FaceIdentifier:
    """
    Face Idenfier class

    init can receive 2 object types of classifier:
    - cv2.CascadeClassifier
    - str

    The str classfier should be used to initialize a cv2.CascadeClassifier object
    """

    __default_color = (0, 255, 255)

    def __init__(self, classifier, color = __default_color):
        if isinstance(classifier, str):
            self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + classifier)
        elif isinstance(classifier, cv2.CascadeClassifier):
            self.classifier = classifier
        else:
            raise Exception("Sorry, FaceIdentifier can not be initialied with these params")
        self.color = color

    def identify_from_frame(self, frame):
        """
        Identify faces from a frame using the self.classifier
        The identified faces will be highlitghed within a rectangule with the self.color
        """
        classifier = self.classifier
        frame_tons_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(frame_tons_cinza, scaleFactor = 1.3, minNeighbors=3)
        frame_temp = frame.copy()

        for(x, y, lar, alt) in faces:
            cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), self.color, 2)
        cv2.imshow(WINDOW_NAME, frame_temp)


class ColorRange:
    """ Defines an abstraction of a Color Range"""
    def __init__(self, color_light, color_dark):
        self.color_light = color_light
        self.color_dark = color_dark

    def get_np_lowerb(self):
        return np.array(self.color_dark, dtype="uint8")

    def get_np_upperb(self):
        return np.array(self.color_light, dtype="uint8")

    def get_tupple_lower(self):
        return (self.color_light[0], self.color_light[1], self.color_light[2])

    def get_tupple_upper(self):
        return (self.color_dark[0], self.color_dark[1], self.color_dark[2])

BLUE_COLOR_RANGE = ColorRange(color_light = [255,150,56], color_dark = [88,34,0])
RED_COLOR_RANGE = ColorRange(color_light = [97,105,255], color_dark = [6,0,171])
GRAY_COLOR_RANGE = ColorRange(color_light = [123,123,123], color_dark = [28,24,22])

class ColorObjectIdentifier:
    """
    Indentify the biggest object with self.color and highlight it.
    """

    def __init__(self, color_range = BLUE_COLOR_RANGE):
        if isinstance(color_range, ColorRange):
            self.color_range = color_range
            self.highlight_color = color_range.get_tupple_lower()
        else:
            raise Exception("Sorry, ColorObjectIdentifier can not be initialied with these params")

    def identify_from_frame(self, frame):
        """
        Identify the biggest object from a frame using the self.color
        The identified object will be highlitghed within a rectangule with the self.highlight_color
        """
        lowerb = self.color_range.get_np_lowerb()
        upperb = self.color_range.get_np_upperb()
        objeto = cv2.inRange(frame, lowerb, upperb)

        (contours, _) = cv2.findContours(objeto.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0 :
            contour = sorted(contours, key = cv2.contourArea, reverse=True)[0]
            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
            cv2.drawContours(frame, [rect], -1, self.highlight_color, 2)

        cv2.imshow(WINDOW_NAME, frame)
