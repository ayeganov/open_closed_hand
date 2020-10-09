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

        self.selection = None
        self.drag_start = None
        self.track_window = None
        self.hist = None

    @property
    def calibrated(self):
        return self.hist is not None and self.track_window is not None

    def sample_skin_color(self, vis, hsv, mask):
        """
        Samples the color of the skin, and returns the image with the
        highlighted area of selection

        @param image - numpy array representing the image
        @param hsv - image converted to hsv color space
        @param mask - image with background removed
        @returns 
        """
        if self.selection is None:
            return vis

        if self.track_window is not None:
            return vis

        x0, y0, x1, y1 = self.selection
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self.hist = hist.reshape(-1)
        self.show_hist()

        vis_roi = vis[y0:y1, x0:x1]
        cv2.bitwise_not(vis_roi, vis_roi)
        vis[mask == 0] = 0

        return vis

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

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

        print(f"lh: {self._hue_low}, hh: {self._hue_high}, sl: {self._sat_low}, sh: {self._sat_high}, vl: {self._value_low}, vh: {self._value_high}")

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

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def extract_hand_image(self, hsv, mask):
        """
        Calculates the location of the hand in the image

        @param hsv - frame in HSV color space
        @param mask - best guess at background removal
        """
        prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

        change_me = prob.copy()
        change_me[change_me > 0] = 255

        kernel = np.ones((5, 5), np.uint8)
        change_me = cv2.morphologyEx(change_me, cv2.MORPH_CLOSE, kernel)

        image, contours, hierarchy = cv2.findContours(change_me, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None] * len(contours)
        bound_rect = [None] * len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            bound_rect[i] = cv2.boundingRect(contours_poly[i])

        try:
            hand_box = None
            max_area = 0
            for i in range(len(contours)):
                x, y, w, h = np.array(bound_rect[i]).astype(np.uint16)
    #                        print(f"Drawing bounding rect {int_rect}")
                pt1, pt2 = (x, y), (x+w, y+h)
                area = w * h
                if area > max_area:
                    max_area = area
                    hand_box = pt1, pt2

            if max_area < 1000:
                return None, change_me, None

            pt1, pt2 = hand_box
            hand = change_me[pt1[1]:pt2[1], pt1[0]:pt2[0]]

            return hand, change_me, hand_box

        except Exception as error:
            traceback.print_exc()
            print(error)
            print(track_box)
            raise RuntimeError("Failed to extract image")


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


def draw_message(image, msg):
    location = (100, 700)
    font_scale = 1
    color = (250, 0, 5)
    thickness = 2

    cv2.putText(image,
                msg,
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
    x, y = 200, 50

    return draw_rectangle(image, x, y, 260, 260, color_bgr)


def mouse_callback(event, x, y, event_flag, skin_detector):
    """
    Handle the mouse callback
    """
    print(f"event: {event}, x: {x}, y: {y}, flag: {event_flag}, detector: {skin_detector}")


def start_capture(resolution, output_res, model_path):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Failed to open the camera, bailing...")
        return

    print(f"Starting the camera in resolution: {resolution}")
    width, height = resolution
    cam.set(3, width)
    cam.set(4, height)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using '{device}' for running the model")
    network = get_model(device, model_path)
    detector = SkinDetector()
    bg_remover = BackgroundRemover()

    skin_status_location = (20, 20)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, detector.on_mouse)
    cv2.createTrackbar("Vmin", WINDOW_NAME, 30, 255, lambda _: None)
    cv2.createTrackbar("Vmax", WINDOW_NAME, 255, 255, lambda _: None)
    cv2.createTrackbar("Smin", WINDOW_NAME, 30, 255, lambda _: None)

    mask = None
    mode = "setup"
    class_names = ["closed", "open"]
    show_backproj = False

    message = "Use mouse for skin calibration"
    location = None
    index = 0

    while cam.isOpened():
        try:
            k = cv2.waitKey(5) & 0xFF
            ret, frame = cam.read()
            frame = np.array(np.fliplr(frame))

            if not ret:
                print("failed to grab frame")
                break

            if k == ord('r'):
                show_backproj = not show_backproj
            if k == ord('o'):
                # Open pressed
                print("OPEN")
                location = "open"
                message = f"Collecting data for {location} hand"
            elif k == ord('c'):
                print("CLOSED")
                location = "closed"
                message = f"Collecting data for {location} hand"
            elif k == ord('q'):
                print("Exiting by user request...")
                break
            elif k == ord('m'):
                print("Calculating the skin mask")
                foreground = bg_remover.get_foreground(frame)
                mask = detector.calc_skin_mask(foreground)
            elif k == ord('b'):
                print("Capturing background")
                bg_remover.calibrate(frame)
            elif k == ord('f'):
                bg_remover.get_foreground(frame)
            elif k == ord('p'):
                if not detector.calibrated:
                    message = "Calibrate skin detector"
                    continue
                mode = "process"
                message = "Move your hand"
            elif k == ord('s'):
                mode = "setup"
                message = "Use mouse for skin calibration"
            elif k == ord('d'):
                location = None
                message = "Move your hand"
            elif k == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            draw_on_me = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            draw_message(draw_on_me, message)
            if mode == "setup":
                draw_on_me = detector.sample_skin_color(draw_on_me, hsv, mask)
                draw_process_status(draw_on_me, mode)

                draw_skin_status(draw_on_me, detector)

                cv2.imshow(WINDOW_NAME, draw_on_me)

            elif mode == "process":
                hand, back_proj, coordinates = detector.extract_hand_image(hsv, mask)

                if hand is None:
                    print("failed to find hand")
                    continue

                pt1, pt2 = coordinates
                center = int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)
#                print(f"Center: {center}")

                if show_backproj:
                    draw_on_me[:] = back_proj[...,np.newaxis]

                draw_on_me = cv2.rectangle(draw_on_me, pt1, pt2, (0, 255, 0), 2)
                draw_on_me = cv2.circle(draw_on_me, center, 5, (0, 255, 200), 2, cv2.LINE_AA)

                cv2.imshow(WINDOW_NAME, draw_on_me)

                if location is not None:
                    frame_to_size = cv2.resize(hand, output_res)

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
            cv2.destroyAllWindows()
            cam.release()
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
                        default="1024x768",
                        type=to_resolution)
    parser.add_argument("-o",
                        "--output_res",
                        default="120x120",
                        type=to_resolution)
    parser.add_argument("-m",
                        "--model",
                        type=argparse.FileType("r"),
                        required=True)

    args = parser.parse_args()

    start_capture(args.resolution, args.output_res, args.model.name)


if __name__ == "__main__":
    main()
