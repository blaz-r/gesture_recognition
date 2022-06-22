import depthai as dai


class GestureRecognition:
    def __init__(self):
        # path to models
        self.palm_detection_path = "/models/n_palm_detection_lite_sh4.blob"
        self.hand_landmar_path = "/models/n_hand_landmark_lite_sh4.blob"
        self.pd_postprocessing_path = "/models/palm_detection_post/palm_detection_post_sh2.blob"

        self.manager_script_path = "manager_script.py"

        self.resolution = (1920, 1080)
        self.pd_input_length = 192
        self.lm_input_length = 224

        # depthAI oak-d
        self.device = dai.Device()
        print(f"Device started")

        # pipeline
        self.device.startPipeline(self.create_pipeline())

        # data queue
        self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=1, blocking=False)

    def create_pipeline(self):
        print("Starting pipeline creation")

        pipeline = dai.Pipeline()

        print("Adding RGB camera")
        rgb_cam = pipeline.createColorCamera()
        rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb_cam.setInterleaved(False)
        rgb_cam.setFps(35.0)

        print("Adding manager script")
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

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
        preprocess_pd_manip.inputImage.setBlocking(False)
        # image comes directly from camera then get cropped according to config obtained from manager script
        rgb_cam.preview.link(preprocess_lm_manip.inputImage)
        manager_script.outputs["preprocess_lm_manip_config"].link(preprocess_lm_manip.inputConfig)

        print("Adding landmark mediapipe model")
        lm_mediapipe_model = pipeline.create(dai.node.NeuralNetwork)
        lm_mediapipe_model.setBlobPath(self.hand_landmar_path)
        preprocess_lm_manip.out.link(lm_mediapipe_model.input)
        lm_mediapipe_model.out.link(manager_script.inputs["lm_result"])

        return pipeline

    def build_manager_script(self):
        with open(self.manager_script_path) as m_script:
            return m_script.read()

    def next_frame(self):
        pass

    def close(self):
        pass