#!/usr/bin/env python

import argparse
import os
import time

from torchvision import models, transforms
import cv2
import numpy as np
import torch
import torch.nn as nn


WINDOW_NAME = 'Hand Recognizer'


eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(240),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_model(device, model_path):
    """
    Load the model weights to produce a working network

    @param device - string representing either GPU or CPU
    @param model_path - path to the model weights that has already been pre-trained
    @returns an initialized neural network
    """
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)

    model_conv.load_state_dict(torch.load(model_path, map_location=device))
    model_conv.eval()

    return model_conv


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
    def __init__(self):
        self._background = None

    def calibrate(self, image):
        self._background = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @property
    def calibrated(self):
        return self._background is not None

    def get_foreground(self, image):
        mask = self._calc_foreground_mask(image)
        foreground = image.copy()
        foreground[mask] = (0, 0, 0)
#        cv2.imshow("foreground", foreground)
        return foreground

    def _calc_foreground_mask(self, image):
        if self._background is None:
            return np.zeros(image.shape)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold_offset = 10

        bg_diff_low = np.clip(self._background - threshold_offset, 0, 255)
        bg_diff_high = np.clip(self._background + threshold_offset, 0, 255)

        low_match = bg_diff_low <= gray
        high_match = gray <= bg_diff_high

        all_matches = np.logical_and(low_match, high_match)

        return all_matches


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

    @property
    def calibrated(self):
        return self._hue_low is not None

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        (x_left, y_left), (x_right, y_right) = self._top_sample
        top_sample = image[y_left:y_right, x_left:x_right]

        (x_left, y_left), (x_right, y_right) = self._bottom_sample
        bottom_sample = image[y_left:y_right, x_left:x_right]

#        cv2.imshow("top_sample", top_sample)
#        cv2.imshow("bottom_sample", bottom_sample)

        self._calculate_threshold(top_sample, bottom_sample)

    def _calculate_threshold(self, top, bottom):
        offset_low = 80
        offset_high = 30

        top_mean = np.mean(top, axis=(0, 1))
        bottom_mean = np.mean(bottom, axis=(0, 1))

        self._hue_low = max(0.0, np.min((top_mean[0], bottom_mean[0])) - offset_low)
        self._hue_high = np.max((top_mean[0], bottom_mean[0])) + offset_high

        self._sat_low = max(0.0, np.min((top_mean[1], bottom_mean[1])) - offset_low)
        self._sat_high = np.max((top_mean[1], bottom_mean[1])) + offset_high

        self._value_low = max(0.0, np.min((top_mean[2], bottom_mean[2])) - offset_low)
        self._value_high = np.max((top_mean[2], bottom_mean[2])) + offset_high

    def calc_skin_mask(self, image):
        if self._hue_low is None:
            return np.zeros(image.shape, dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low = (self._hue_low, self._sat_low, self._value_low)
        high = (self._hue_high, self._sat_high, self._value_high)

        output = cv2.inRange(image, low, high)

        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, struct_element)

        output = cv2.dilate(output, struct_element, iterations=1)

        return output


def draw_skin_status(image, skin_detector):
    status = "READY" if skin_detector.calibrated else "NEEDS A SAMPLE"
    text = f"Skin detector: {status}"
    location = (20, 30)
    font_scale = 1
    color = (255, 100, 255)
    thickness = 2

    cv2.putText(image,
                text,
                location,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA)


def draw_bg_status(image, bg_remover):
    status = "READY" if bg_remover.calibrated else "NEEDS A SAMPLE"
    text = f"BG remover: {status}"
    location = (600, 30)
    font_scale = 1
    color = (55, 100, 255)
    thickness = 2

    cv2.putText(image,
                text,
                location,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA)


def draw_process_status(image, mode):
    text = f"Mode: {mode}"
    location = (600, 600)
    font_scale = 1
    color = (55, 100, 255)
    thickness = 2

    cv2.putText(image,
                text,
                location,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA)


def draw_hand_square(image):
    """
    Draw the hand square on the image

    @param image - the image to draw on. This function will modify the passed
                   in image object.
    @returns coordinate of the square for easy extraction
    """
    color_bgr = (100, 50, 255)
    x, y = 20, 30

    return draw_rectangle(image, x, y, 260, 260, color_bgr)


def start_capture(resolution):
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    detector = SkinDetector()
    bg_remover = BackgroundRemover()

    skin_status_location = (20, 20)

    mask = None
    mode = "setup"

    while cam.isOpened():
        try:
            k = cv2.waitKey(1) & 0xFF
            ret, frame = cam.read()
            frame = np.array(np.fliplr(frame))

            if k == ord('o'):
                # Open pressed
                print("OPEN")
            if k == ord('q'):
                print("Exiting by user request...")
                break
            if k == ord('c'):
                print("Performing calibration...")
                detector.calibrate(frame)
            if k == ord('m'):
                print("Calculating the skin mask")
                foreground = bg_remover.get_foreground(frame)
                mask = detector.calc_skin_mask(foreground)
            if k == ord('b'):
                print("Capturing background")
                bg_remover.calibrate(frame)
            if k == ord('f'):
                bg_remover.get_foreground(frame)
            if k == ord('p'):
                mode = "process" if mode == "setup" else "setup"
                print("Switching to processing mode...")
            if k == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            if not ret:
                print("failed to grab frame")
                break

            if mode == "setup":
                skin_samples, top, bottom = detector.draw_skin_samples(frame.copy())
                draw_process_status(skin_samples, mode)

                (x_left, y_left), (x_right, y_right) = top
                top_sample = frame[y_left:y_right, x_left:x_right]

                (x_left, y_left), (x_right, y_right) = bottom
                bottom_sample = frame[y_left:y_right, x_left:x_right]

                draw_skin_status(skin_samples, detector)
                draw_bg_status(skin_samples, bg_remover)

                cv2.imshow(WINDOW_NAME, skin_samples)
    #            cv2.imshow("top_sample", top_sample)
    #            cv2.imshow("bottom_sample", bottom_sample)

            elif mode == "process":
                draw_on_me = frame.copy()
                hand_square = draw_hand_square(draw_on_me)
                draw_process_status(draw_on_me, mode)
                cv2.imshow(WINDOW_NAME, draw_on_me)
#                if mask is not None:
#                    cv2.imshow(WINDOW_NAME, mask)

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
    parser.add_argument("-m",
                        "--model",
                        type=argparse.FileType("r"),
                        required=False)

    args = parser.parse_args()

    start_capture(args.resolution)



if __name__ == "__main__":
    main()
