#!/usr/bin/env python

import argparse
import os
import time

import cv2
import numpy as np


WINDOW_NAME = 'Image Getter'


def draw_rectangle(image, x, y, width, height, color_bgr):
    """
    Draw the rectangle at given location with the given width and height

    @param image - numpy array representing the image
    @param x - x coordinate of top left corner
    @param y - y coordinate of top left corner
    @param width - width of the rectangle along the x axis
    @param height - height of the rectangle along the y axis
    @returns top left corner and bottom right corner in a tuple
    """
    thickness = 1

    start_point = (x, y)
    end_point = (x + width, y + height)

    image = cv2.rectangle(image, start_point, end_point, color_bgr, thickness)
    return start_point, end_point


class BackgroundRemover:
    """
    Removes the background
    """


class SkinDetector:
    """
    Detects the range of colors representing the skin
    """
    def __init__(self):
        self._hue_low = None
        self._hue_high = None
        self._sat_low = None
        self._sat_high = None
        self._value_low = None  # brightness
        self._value_high = None
        self._top_sample = None
        self._bottom_sample = None

    def draw_skin_samples(self, image):
        """
        Draw the skin sample rectangles

        @param image - numpy array representing the image
        @returns two tuples of bottom left corners of the skin samples
        """
        color_bgr = (0, 0, 255)
        x, y = 150, 50

        draw_rectangle(image, x, y, 160, 380, color_bgr)

        x, y = 200, 100
        width = 60
        height = 80
        color_bgr = (0, 255, 0)

        self._top_sample = draw_rectangle(image, x, y, width, height, color_bgr)
        y = 300
        self._bottom_sample = draw_rectangle(image, x, y, width, height, color_bgr)

        return image, self._top_sample, self._bottom_sample

    def calibrate(self, image):
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        (x_left, y_left), (x_right, y_right) = self._top_sample
        top_sample = image[y_left:y_right, x_left:x_right]

        (x_left, y_left), (x_right, y_right) = self._bottom_sample
        bottom_sample = image[y_left:y_right, x_left:x_right]

        self._calculate_threshold(top_sample, bottom_sample)

    def _calculate_threshold(self, top, bottom):
        offset_low = 80
        offset_high = 30

        top_mean = np.mean(top, axis=(0, 1))
        bottom_mean = np.mean(bottom, axis=(0, 1))

        self._hue_low = np.min((top_mean[0], bottom_mean[0])) - offset_low
        self._hue_high = np.max((top_mean[0], bottom_mean[0])) + offset_high

        self._sat_low = np.min((top_mean[1], bottom_mean[1])) - offset_low
        self._sat_high = np.max((top_mean[1], bottom_mean[1])) + offset_high

        self._value_low = np.min((top_mean[2], bottom_mean[2])) - offset_low
        self._value_high = np.max((top_mean[2], bottom_mean[2])) + offset_high

    def calc_skin_mask(self, image):
        if self._hue_low is None:
            return np.zeros(image.shape, dtype=np.uint8)

        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low = (self._hue_low, self._sat_low, 0)
        high = (self._hue_high, self._sat_high, 255)

        output = cv2.inRange(image, low, high)

        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, struct_element)

        output = cv2.dilate(output, struct_element, iterations=1)

        return output


def start_capture(resolution):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    detector = SkinDetector()

    index = 0
    mask = None
    while cam.isOpened():
        try:
            k = cv2.waitKey(1) & 0xFF
            location = None
            ret, frame = cam.read()
            frame = np.array(np.fliplr(frame))

            if k == ord('o'):
                # Open pressed
                print("OPEN")
                location = "open"
            if k == ord('q'):
                print("Exiting by user request...")
                break
            if k == ord('c'):
                print("Performing calibration...")
                detector.calibrate(frame.copy())
            if k == ord('m'):
                print("Calculating the skin mask")
                mask = detector.calc_skin_mask(frame.copy())
            if k == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            if not ret:
                print("failed to grab frame")
                break

            skin_samples, top, bottom = detector.draw_skin_samples(frame.copy())

            (x_left, y_left), (x_right, y_right) = top
            top_sample = frame[y_left:y_right, x_left:x_right]

            (x_left, y_left), (x_right, y_right) = bottom
            bottom_sample = frame[y_left:y_right, x_left:x_right]

            cv2.imshow(WINDOW_NAME, skin_samples)
            cv2.imshow("top_sample", top_sample)
            cv2.imshow("bottom_sample", bottom_sample)

            if mask is not None:
                cv2.imshow("mask", mask)

            if location is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_to_size = cv2.resize(frame, resolution)

                prefix = time.time_ns()
                file_name = f"{prefix}{str(index)}.png"
                full_path = os.path.join(location, file_name)

                if not os.path.exists(location):
                    try:
                        os.makedirs(location)
                    except FileExistsError:
                        pass
                cv2.imwrite(full_path, frame_to_size)
                print(full_path)
                index = index + 1

        except KeyboardInterrupt:
            # When everything done, release the capture
            cam.release()
            cv2.destroyAllWindows()
            print("Exiting due to interrupt...")


def to_resolution(resolution_s):
    '''
    Convert resolution string to a tuple of width and height

    @param resolution_s - string representing required resolution, ie: 640x480
    @returns tuple of width and height
    '''
    try:
        width, height = (int(v) for v in resolution_s.split("x"))
        return width, height
    except Exception as e:
        raise argparse.ArgumentError(str(e))


def main():
    parser = argparse.ArgumentParser(description="Capture Images to specified folder at maximum rate")
    parser.add_argument("-r",
                        "--resolution",
                        default="50x50",
                        type=to_resolution)

    args = parser.parse_args()

    start_capture(args.resolution)



if __name__ == "__main__":
    main()
