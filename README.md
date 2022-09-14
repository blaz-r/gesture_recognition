# gesture_recognition
Repository is part of my BSc thesis with title "Gesture recognition in video streams on an embedded device"

Hand gesture recognition is implemented on a wearable prototype, which works on the OAK-D embedded device, part of DepthAI platform. Device is mounted on user’s head as shown in the following image:

<img src="https://github.com/blaz-r/gesture_recognition/blob/main/images/system.png" width=500>

System offers music/video playback control using the following gestures:
<img src="https://github.com/blaz-r/gesture_recognition/blob/main/images/gestures.jpg" width=500>

You can see detected keypoints and detected gestures on host system:
<img src="https://github.com/blaz-r/gesture_recognition/blob/main/images/host_view.png" width=1000>

Using pyautogui the command is then forwarded to OS.

All the code used to train and test LSTM model is located in ![gesture_lstm repository](https://github.com/blaz-r/gesture_lstm).

### Credits
[mediapipe](https://github.com/google/mediapipe)

[tflite2tensorflow by PINTO0309](https://github.com/PINTO0309/tflite2tensorflow)

[depthai_hand_tracker by geaxgx](https://github.com/geaxgx/depthai_hand_tracker)
