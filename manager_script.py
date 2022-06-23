import marshal
from math import sin, cos, atan2, pi, degrees, floor
from enum import Enum


# BufferMgr is used to statically allocate buffers once
# (replace dynamic allocation).
# These buffers are used for sending result to host
class BufferMgr:
    """
    BufferMgr is used to statically allocate buffers once
    (replace dynamic allocation).
    These buffers are used for sending result to host

    """
    def __init__(self):
        self._bufs = {}
    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
        return buf


class State(Enum):
    """
    State class represent current state of hand detection and landmark regression system

    TRACKING state means that detection is not running, but we get next landmarks
    from box obtained by tracking current landmarks

    DETECTION state means we lost hand, and it needs to be detected again, this means SSD detector model runs first

    """
    TRACKING = 0
    DETECTION = 1


buffer_mgr = BufferMgr()
def send_result(result):
    """
    Send dictionary containing result to host

    :param result: dictionary containing results
    :return: None
    """
    result_serial = marshal.dumps(result)
    buffer = buffer_mgr(len(result_serial))
    buffer.getData()[:] = result_serial
    node.io['host'].send(buffer)


def send_result_hand(lm_score, handedness,
                     rect_center_x, rect_center_y, rect_size,
                     rotation, rrn_lms, sqn_lms, world_lms):
    """
    Make dictionary and send to host

    :param lm_score: landmark detection score
    :param handedness: left / right
    :param rect_center_x: rectangle center x coord
    :param rect_center_y: rectangle center y coord
    :param rect_size: rectangle size (w == h)
    :param rotation: rectangle rotation
    :param rrn_lms: landmark coords in normalised rotated rect system
    :param sqn_lms: landmark coords in normalised square image system
    :param world_lms: landmarks in world coord system
    :return: None
    """
    result = {"detection": 1, "lm_score": lm_score, "handedness": handedness,
              "rotation": rotation, "rect_center_x": rect_center_x, "rect_center_y": rect_center_y,
              "rect_size": rect_size, "rrn_lms": rrn_lms, "sqn_lms": sqn_lms,
              "world_lms": world_lms}
    send_result(result)


def send_result_no_hand():
    """
    Send result when no hand is detected

    :return: None
    """
    result = dict([("detection", 0)])
    send_result(result)


def rr2img(rrn_x, rrn_y, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, cos_rot, sin_rot):
    """
    https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmark_projection_calculator.cc
    Convert a point (rrn_x, rrn_y) expressed in normalized rotated rectangle (rrn)
    into (X, Y) expressed in normalized image (sqn)

    :param rrn_x: x coordinate in rrn
    :param rrn_y: y coordinate in rrn
    :param sqn_rr_center_x: x coord of rotated rectangle center in image
    :param sqn_rr_center_y: y coord of rotated rectangle center in image
    :param sqn_rr_size: size of rotated rectangle
    :param cos_rot: cosine rotation of rectangle
    :param sin_rot: sine rotation of rectangle
    :return: x and y in normalized image
    """
    X = sqn_rr_center_x + sqn_rr_size * ((rrn_x - 0.5) * cos_rot + (0.5 - rrn_y) * sin_rot)
    Y = sqn_rr_center_y + sqn_rr_size * ((rrn_y - 0.5) * cos_rot + (rrn_x - 0.5) * sin_rot)
    return X, Y


def normalize_radians(angle):
    """
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc

    :param angle: angle of rotation
    :return: angle normalized between -pi to pi
    """
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


def make_lm_pre_config(sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation):
    rotated_rect = RotatedRect()
    rotated_rect.center.x = sqn_rr_center_x
    rotated_rect.center.y = (sqn_rr_center_y * frame_size - pad_h) / img_h
    rotated_rect.size.width = sqn_rr_size
    rotated_rect.size.height = sqn_rr_size * frame_size / img_h

    rotated_rect.angle = degrees(rotation)

    config = ImageManipConfig()
    config.setCropRotatedRect(rotated_rect, True)
    config.setResize(lm_input_size, lm_input_size)

    return config


def transform_to_sqn_landmarks(rrn_lms, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation):
    """
    https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmark_projection_calculator.cc

    :param rrn_lms: landmarks in rotated rectangle coordinate system
    :param sqn_rr_center_x: x coord of rotated rectangle center in image
    :param sqn_rr_center_y: y coord of rotated rectangle center in image
    :param sqn_rr_size: size of rotated rectangle
    :param rotation:
    :return:
    """
    sqn_lms = []
    cos_rot = cos(rotation)
    sin_rot = sin(rotation)
    for i in range(21):
        # normalize
        rrn_lms[3 * i] /= lm_input_size
        rrn_lms[3 * i + 1] /= lm_input_size
        rrn_lms[3 * i + 2] /= lm_input_size
        # transform to original image
        sqn_x, sqn_y = rr2img(rrn_lms[3 * i], rrn_lms[3 * i + 1],
                              sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size,
                              cos_rot, sin_rot)
        sqn_lms += [sqn_x, sqn_y]

    return sqn_lms


def hand_landmarks_to_ROI(sqn_lms):
    """
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc

    Calcualted ROI for next frame, this way we perform tracking of hand, and don't need to detect hand again

    :param sqn_lms: landmarks in original (square) image coordinates
    :return: (sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size) -> new ROI rectangle params
    """
    # Compute rotation
    x0 = sqn_lms[0]
    y0 = sqn_lms[1]
    x1 = 0.25 * (sqn_lms[2 * index_mcp_id] + sqn_lms[2 * ring_mcp_id])
    x1 += 0.5 * sqn_lms[2 * middle_mcp_id]
    y1 = 0.25 * (sqn_lms[2 * index_mcp_id + 1] + sqn_lms[2 * ring_mcp_id + 1])
    y1 += 0.5 * sqn_lms[2 * middle_mcp_id + 1]

    rotation = 0.5 * pi - atan2(y0 - y1, x1 - x0)
    rotation = normalize_radians(rotation)

    # Find boundaries of landmarks
    min_x = min_y = 1
    max_x = max_y = 0
    for id in bounding_box_ids:
        min_x = min(min_x, sqn_lms[2 * id])
        max_x = max(max_x, sqn_lms[2 * id])
        min_y = min(min_y, sqn_lms[2 * id + 1])
        max_y = max(max_y, sqn_lms[2 * id + 1])

    axis_aligned_center_x = 0.5 * (max_x + min_x)
    axis_aligned_center_y = 0.5 * (max_y + min_y)
    cos_rot = cos(rotation)
    sin_rot = sin(rotation)

    # Find boundaries of rotated landmarks
    min_x = min_y = 1
    max_x = max_y = -1

    for id in bounding_box_ids:
        original_x = sqn_lms[2 * id] - axis_aligned_center_x
        original_y = sqn_lms[2 * id + 1] - axis_aligned_center_y

        projected_x = original_x * cos_rot - original_y * sin_rot
        projected_y = original_x * sin_rot + original_y * cos_rot

        min_x = min(min_x, projected_x)
        max_x = max(max_x, projected_x)
        min_y = min(min_y, projected_y)
        max_y = max(max_y, projected_y)

    projected_center_x = 0.5 * (max_x + min_x)
    projected_center_y = 0.5 * (max_y + min_y)
    center_x = (projected_center_x * cos_rot - projected_center_y * sin_rot + axis_aligned_center_x)
    center_y = (projected_center_x * sin_rot + projected_center_y * cos_rot + axis_aligned_center_y)
    width = (max_x - min_x)
    height = (max_y - min_y)

    sqn_rr_size = 2 * max(width, height)
    sqn_rr_center_x = (center_x + 0.1 * height * sin_rot)
    sqn_rr_center_y = (center_y - 0.1 * height * cos_rot)

    return sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size

# manager params

pd_score_thresh = 0.5   # palm detection score threshold
lm_score_thresh = 0.5   # landmarks regression score threshold

# ids of landmarks used for tracking
wrist_id = 0
index_mcp_id = 5
middle_mcp_id = 9
ring_mcp_id = 13
bounding_box_ids = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]

pd_input_size = 192     # palm detection model input size 192 x 192
lm_input_size = 224     # landmark regression model input size 224 x 224

frame_size = 1152.0     # size of output frame
pad_h = 252.0           # height padding added to make image square
img_h = 648.0           # height of image

preprocess_pd_config = ImageManipConfig()
preprocess_pd_config.setResizeThumbnail(pd_input_size, pd_input_size, 0, 0, 0)


def main():
    # we start in detection sate since we don't know where the hand is
    current_state = State.DETECTION

    while True:
        if current_state is State.DETECTION:
            # send config to postprocessing for palm detection
            node.io["preprocess_pd_manip_config"].send(preprocess_pd_config)

            # get post processed score and bounding box from palm detector
            detection = node.io["postprocess_pd_result"].get().getLayerFp16("result")
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp1_x, kp1_y = detection

            node.warn(f"Manager received pd result (len={len(detection)}) : "+str(detection))

            # detection[0] is pd_score
            if detection[0] < pd_score_thresh:
                send_result_no_hand()
                continue

            # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
            rotation = pi / 2.0 - atan2(-(kp1_y - kp0_y), kp1_x - kp0_x)
            rotation = normalize_radians(rotation)

            sqn_rr_size = 2.9 * box_size
            sqn_rr_center_x = box_x + box_size * sin(rotation) / 2.0
            sqn_rr_center_y = box_y - box_size * cos(rotation) / 2.0

        # make and send config to lm preprocessing module, to crop the region of hand
        lm_config = make_lm_pre_config(sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation)
        node.io["preprocess_lm_manip_config"].send(lm_config)

        # get results from landmark model
        lm_result = node.io["lm_result"].get()
        lm_score = lm_result.getLayerFp16("Identity_1")[0]

        if lm_score > lm_score_thresh:
            # landmarks successfully obtained -> transition to tracking
            current_state = State.TRACKING

            handedness = lm_result.getLayerFp16("Identity_2")[0]
            rrn_lms = lm_result.getLayerFp16("Identity_dense/BiasAdd/Add")
            world_lms = lm_result.getLayerFp16("Identity_3_dense/BiasAdd/Add")

            sqn_lms = transform_to_sqn_landmarks(rrn_lms, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation)

            # Send result to host
            send_result_hand(lm_score, handedness, sqn_rr_center_x,
                             sqn_rr_center_y, sqn_rr_size, rotation,
                             rrn_lms, sqn_lms, world_lms)


            # Calculate the ROI for next frame
            sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size = hand_landmarks_to_ROI(sqn_lms)
        else:
            current_state = State.DETECTION
            send_result_no_hand()


main()
