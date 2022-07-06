import torch
from torchvision.ops import nms
import numpy as np
import onnx
import onnxsim
from mediapipe_utils import generate_handtracker_anchors

PD_MODEL_INPUT_LENGTH = 192


class TorchPDPostprocess(torch.nn.Module):
    """
    Torch neural network module, used to perform math on depthai device

    N in new palm detection models is 2016

    """
    def __init__(self, anchors, top_k=1):
        super(TorchPDPostprocess, self).__init__()
        self.top_k = top_k
        self.anchors = torch.from_numpy(anchors[:, :2]).float() # [N, 2]
        self.plus_anchor_center = np.array([[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                            [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
        self.plus_anchor_center = torch.from_numpy(self.plus_anchor_center).float()
        self.iou_threshold = 0.3

    def forward(self, x, y):
        """
        Perform froward pass, where x and y in this case are value from palm detection model

        :param x: classificators (Identity_1), shape: [1, N, 1]
        :param y: regressors (Identity), shape: [1, N, 18]
        :return: (scores, cx, cy, w, kp0, kp2), w = h
        """
        # N
        scores = torch.sigmoid(x[0, :, 0])

        dets = torch.squeeze(y, 0)  # [N, 18]
        # [N, 18] = [N, 18] + [N, 2] * [2, 18]
        dets = dets / PD_MODEL_INPUT_LENGTH + torch.mm(self.anchors, self.plus_anchor_center)

        # dets : the 4 first elements describe the square bounding box around the hand (cx, cy, w, h)
        # cx,cy,w,h -> x1,y1,x2,y2 with:
        # x1 = cx - w/2
        # y1 = cy - h/2
        # x2 = cx + w/2
        # y2 = cy + h/2

        bb_cxcy = dets[:, :2]
        bb_wh_half = dets[:, 2:4] * 0.5
        bb_x1_y1 = bb_cxcy - bb_wh_half
        bb_x2_y2 = bb_cxcy + bb_wh_half
        bb_x1y1x2y2 = torch.cat((bb_x1_y1, bb_x2_y2), dim=1).float()

        # NMS
        # Parameters:
        # boxes: [N, 4] in (x1, y1, x2, y2) format
        # scores: [N]
        # iou_threshold: float
        # Returns: int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        # print(bb_x1y1x2y2.dtype, scores.dtype)
        keep_idx = nms(bb_x1y1x2y2, scores, self.iou_threshold)[:self.top_k]

        # The 14 elements of dets from 4 to 18 corresponds to 7 (x,y) normalized keypoints coordinates (useful for determining rotated rectangle)
        # Among the the 7 keypoints kps, we are interested only in kps[0] (wrist center) and kps[2] (middle finger)
        kp0 = dets[:, 4:6][keep_idx]
        kp2 = dets[:, 8:10][keep_idx]

        # We return (scores, cx, cy, w, kp0, kp2) (shape: [top_k, 8]) (no need of h since w=h)
        scores = torch.unsqueeze(scores[keep_idx], 1)
        cxcyw = dets[:, :3][keep_idx]

        dets = torch.cat((scores, cxcyw, kp0, kp2), dim=1)
        return dets


def test(anchors, top_k=1):
    """
    Check dimensions

    :param anchors: anchors for SSD
    :param top_k: number of best results to keep after nms
    :return: None
    """
    print("Testing postprocessing model")
    model = TorchPDPostprocess(anchors, top_k)
    N = anchors.shape[0]
    X = torch.randn(1, N, 1, dtype=torch.float)
    Y = torch.randn(1, N, 18, dtype=torch.float)
    result = model(X, Y)
    print("Result shape:", result.shape)


def export_onnx(anchors, top_k, onnx_name):
    """
    Export model as onnx

    :param anchors: anchors used for ssd
    :param top_k:  number of best results to keep after nms
    :param onnx_name: name of output onnx file
    :return:
    """
    model = TorchPDPostprocess(anchors, top_k)
    N = anchors.shape[0]
    X = torch.randn(1, N, 1, dtype=torch.float)
    Y = torch.randn(1, N, 18, dtype=torch.float)

    print(f"Generating {onnx_name}")
    # names of output from palm detection model
    # classificators (Identity_1), shape: [1, N, 1]
    # regressors (Identity), shape: [1, N, 18]
    names = ['Identity_1', 'Identity']
    torch.onnx.export(
        model,
        (X, Y),
        onnx_name,
        opset_version=11,
        do_constant_folding=True,
        # verbose=True,
        input_names=names,
        output_names=['result']
    )


def simplify(model):
    """
    Simplify onnx model

    :param model: model to be simplified
    :return: simplified model
    """
    model_simp, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    print("Model has been simplified.")
    return model_simp


def main():
    anchors = generate_handtracker_anchors(PD_MODEL_INPUT_LENGTH, PD_MODEL_INPUT_LENGTH).astype(float)

    test(anchors)

    file_name = "palm_detection_post"
    non_simp_onnx = f"{file_name}_nosimp.onnx"
    export_onnx(anchors, 1, non_simp_onnx)

    model = onnx.load(non_simp_onnx)
    model = simplify(model)

    simp_onnx = f"{file_name}.onnx"
    onnx.save(model, simp_onnx)
    print("Model has been successfully created and saved as onnx")


if __name__ == "__main__":
    main()
