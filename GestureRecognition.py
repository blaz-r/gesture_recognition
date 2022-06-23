import marshal

import cv2
import numpy as np

import mediapipe_utils as mpu

import depthai as dai


class GestureRecognition:
    def __init__(self):
        # path to models
        self.palm_detection_path = "models/n_palm_detection_lite_sh4.blob"
        self.hand_landmark_path = "models/n_hand_landmark_lite_sh4.blob"
        self.pd_postprocessing_path = "models/palm_detection_post/palm_detection_post_sh2.blob"

        self.manager_script_path = "manager_script.py"

        self.resolution = (1920, 1080)

        # scale image, this better in my case than native 1920 x 1080
        self.internal_frame_height = 640
        width, self.scale_nd = mpu.find_isp_scale_params(self.internal_frame_height * self.resolution[0] / self.resolution[1],
                                                         self.resolution,
                                                         is_height=False)

        self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.pad_h = (self.img_w - self.img_h) // 2
        self.frame_size = self.img_w

        self.pd_input_length = 192
        self.lm_input_length = 224

        self.lm_score_thresh = 0.5

        # depthAI oak-d
        self.device = dai.Device()
        print(f"Device started")

        # pipeline
        self.device.startPipeline(self.create_pipeline())

        # video queue
        self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

        # hand data queue
        self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=1, blocking=False)

        # lm debug q
        self.q_pre_lm_manip_out = self.device.getOutputQueue(name="pre_lm_manip_out", maxSize=1, blocking=False)

    def create_pipeline(self):
        print("Starting pipeline creation")

        pipeline = dai.Pipeline()

        print("Adding RGB camera")
        rgb_cam = pipeline.createColorCamera()
        rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_cam.setInterleaved(False)
        rgb_cam.setFps(36)
        rgb_cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        rgb_cam.setVideoSize(self.img_w, self.img_h)
        rgb_cam.setPreviewSize(self.img_w, self.img_h)

        print("Adding video out to queue")
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        rgb_cam.video.link(cam_out.input)

        print("Adding manager script")
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        print("Adding manager out link")
        manager_out = pipeline.createXLinkOut()
        manager_out.setStreamName("manager_out")
        manager_script.outputs["host"].link(manager_out.input)

        print("Adding palm detection preprocessing image manipulator")
        # palm detection preprocessing, crop to square width
        preprocess_pd_manip = pipeline.create(dai.node.ImageManip)
        preprocess_pd_manip.setMaxOutputFrameSize(self.pd_input_length**2 * 3)
        # wait for manager script config, this way we can skip if tracking was successful
        preprocess_pd_manip.setWaitForConfigInput(True)
        preprocess_pd_manip.inputImage.setQueueSize(1)
        preprocess_pd_manip.inputImage.setBlocking(False)
        rgb_cam.preview.link(preprocess_pd_manip.inputImage)
        manager_script.outputs["preprocess_pd_manip_config"].link(preprocess_pd_manip.inputConfig)

        print("Adding palm detection mediapipe model")
        # palm detection mediapipe model
        pd_mediapipe_model = pipeline.create(dai.node.NeuralNetwork)
        pd_mediapipe_model.setBlobPath(self.palm_detection_path)
        preprocess_pd_manip.out.link(pd_mediapipe_model.input)

        print("Adding palm detection postprocessing model")
        # palm detection postprocessing model
        # torch(decode_bboxes + nms)
        postprocess_pd = pipeline.create(dai.node.NeuralNetwork)
        postprocess_pd.setBlobPath(self.pd_postprocessing_path)
        pd_mediapipe_model.out.link(postprocess_pd.input)
        postprocess_pd.out.link(manager_script.inputs["postprocess_pd_result"])

        print("Adding landmark preprocessing image manipulator")
        preprocess_lm_manip = pipeline.create(dai.node.ImageManip)
        preprocess_lm_manip.setMaxOutputFrameSize(self.lm_input_length**2 * 3)
        preprocess_lm_manip.setWaitForConfigInput(True)
        preprocess_lm_manip.inputImage.setQueueSize(1)
        preprocess_lm_manip.inputImage.setBlocking(False)
        # image comes directly from camera then get cropped according to config obtained from manager script
        rgb_cam.preview.link(preprocess_lm_manip.inputImage)
        manager_script.outputs["preprocess_lm_manip_config"].link(preprocess_lm_manip.inputConfig)

        # pre lm manipulation debug output queue
        pre_lm_manip_out = pipeline.createXLinkOut()
        pre_lm_manip_out.setStreamName("pre_lm_manip_out")
        preprocess_lm_manip.out.link(pre_lm_manip_out.input)

        print("Adding landmark mediapipe model")
        lm_mediapipe_model = pipeline.create(dai.node.NeuralNetwork)
        lm_mediapipe_model.setBlobPath(self.hand_landmark_path)
        preprocess_lm_manip.out.link(lm_mediapipe_model.input)
        lm_mediapipe_model.out.link(manager_script.inputs["lm_result"])

        print("Pipeline successfully created")
        return pipeline

    def extract_hand_data(self, res):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"] * self.frame_size
        hand.rotation = res["rotation"]
        hand.rect_points = mpu.rotated_rect_to_points(hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a,
                                                      hand.rect_h_a, hand.rotation)
        hand.lm_score = res["lm_score"]
        hand.handedness = res["handedness"]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms']).reshape(-1, 3)
        hand.landmarks = (np.array(res["sqn_lms"]) * self.frame_size).reshape(-1, 2).astype(int)

        # since we added padding to make the image square,
        # we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            hand.landmarks[:, 1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h

        return hand

    def build_manager_script(self):
        with open(self.manager_script_path) as m_script:
            return m_script.read()

    def next_frame(self):

        in_video = self.q_video.get()
        video_frame = in_video.getCvFrame()

        pre_lm_manip = self.q_pre_lm_manip_out.tryGet()
        if pre_lm_manip:
            pre_lm_manip = pre_lm_manip.getCvFrame()
            cv2.imshow("pre_lm_manip", pre_lm_manip)

        res = marshal.loads(self.q_manager_out.get().getData())

        hand = None
        if res["detection"]:
            hand = self.extract_hand_data(res)

        return video_frame, hand, None

    def exit(self):
        self.device.close()