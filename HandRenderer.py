import cv2
import numpy as np

# https://google.github.io/mediapipe/solutions/hands.html
HAND_LINES = [[0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8],
              [5, 9], [9, 10], [10, 11], [11, 12],
              [9, 13], [13, 14], [14, 15], [15, 16],
              [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]]


class HandRenderer:
    """
    Class used to draw landmarks on hands and display image

    """
    def __init__(self, gestures):
        self.gestures = gestures

        self.show_lm_rot_rect = False
        self.show_landmarks = True

        self.gesture_text = []

    def draw_hand(self, frame, hand):
        """
        Draw hand on frame

        :param frame: cv2 image frame
        :param hand: HandRegion object
        :return: frame with hand landmarks drawn on
        """

        if hand is not None:
            # used for thickness of lines with respect to hand size
            thickness_coef = hand.rect_w_a / 420

            if hand.lm_score > self.gestures.lm_score_thresh:
                if self.show_lm_rot_rect:
                    cv2.polylines(frame, [hand.rect_points], True, (0, 255, 255), 2, cv2.LINE_AA)

                if self.show_landmarks:
                    lines = [np.array([hand.landmarks[point] for point in line], int) for line in HAND_LINES]
                    # color in BGR
                    cv2.polylines(frame, lines, False, (255, 0, 0), int(1 + thickness_coef * 3), cv2.LINE_AA)

                    radius = int(1 + thickness_coef * 5)

                    for x, y in hand.landmarks[:, :2]:
                        cv2.circle(frame, (x, y), radius, (0, 255, 00), -1)

        return frame

    def draw_gesture(self, frame, gesture):
        """
        Draw gesture text in upper left corner of frame

        :param frame: cv video frame
        :param gesture: detected gesture string
        :return: frame with gesture text drawn on
        """
        if gesture is not None:
            if len(self.gesture_text) > 0:
                if gesture != self.gesture_text[-1]:
                    self.gesture_text.append(gesture)
            else:
                self.gesture_text.append(gesture)

            self.gesture_text = self.gesture_text[-5:]

        cv2.rectangle(frame, (0, 0), (1152, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(self.gesture_text), (3, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (240, 240, 255), 2, cv2.LINE_AA)

        return frame

    def wait_key(self, delay=1):
        """
        Wait for key input
        Key 1: shows landmarks on hand
        Key 2: shows bounding rectangle around hand

        :param delay: delay when waiting each cycle
        :return: key value
        """
        key = cv2.waitKey(delay)

        if key == ord('1'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('2'):
            self.show_lm_rot_rect = not self.show_lm_rot_rect
        elif key == ord('3'):
            self.gesture_text = []

        return key

    @staticmethod
    def exit():
        """
        Free resources

        :return: None
        """
        cv2.destroyAllWindows()
