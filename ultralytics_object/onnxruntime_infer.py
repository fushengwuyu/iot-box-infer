import argparse
import cv2
import numpy as np
from numpy import array
import onnxruntime as rt
import time


class LetterBox:
    """
    调整图像大小和填充
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better val mAP)
        if not self.scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
                 new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(
                dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                    shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'],
                                   (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels"""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


class NonMaxSuppression:
    def __init__(self,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 classes=None,
                 agnostic=False,
                 multi_label=False,
                 labels=(),
                 max_det=300,
                 nc=0,  # number of classes (optional)
                 max_time_img=0.05,
                 max_nms=30000,
                 max_wh=7680, ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.labels = labels
        self.max_det = max_det
        self.nc = nc
        self.max_time_img = max_time_img
        self.max_nms = max_nms
        self.max_wh = max_wh

    def __call__(self, prediction):
        # Checks
        assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction,
                      (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        nc = self.nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].max(1) > self.conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + self.max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        self.multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height

            x = x.transpose(1, 0)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if self.labels and len(self.labels[xi]):
                lb = self.labels[xi]
                v = np.zeros((len(lb), nc + nm + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = np.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x[:, :4], x[:, 4:nc + 4], x[:, nc + 4:]
            box = self.xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            if self.multi_label:
                i, j = (cls > self.conf_thres).nonzero(as_tuple=False).T
                x = np.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf = np.max(cls, axis=1, keepdims=True)
                j = np.argmax(cls, axis=1)
                x = np.concatenate((box, conf, j[:, np.newaxis].astype(np.float64), mask), 1)[
                    conf.reshape(-1) > self.conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            # x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
            x = x[np.argsort(-x[:, 4])[:self.max_nms]]
            # Batched NMS
            c = x[:, 5:6] * (0 if self.agnostic else self.max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.numpy_nms(boxes, scores, self.iou_thres)  # NMS

            i = i[:self.max_det]  # limit detections

            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > self.iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = np.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output

    def box_area(self, boxes: array):
        """
        :param boxes: [N, 4]
        :return: [N]
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
            y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def box_iou(self, box1: array, box2: array):
        """
        :param box1: [N, 4]
        :param box2: [M, 4]
        :return: [N, M]
        """
        area1 = self.box_area(box1)  # N
        area2 = self.box_area(box2)  # M
        # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou  # NxM

    def numpy_nms(self, boxes: array, scores: array, iou_threshold: float):
        idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
        keep = []
        while idxs.size > 0:  # 统计数组中元素的个数
            max_score_index = idxs[-1]
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)

            if idxs.size == 1:
                break
            idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
            other_boxes = boxes[idxs]  # [?, 4]
            ious = self.box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
            idxs = idxs[ious[0] <= iou_threshold]

        keep = np.array(keep)
        return keep


class YoloInference:
    def __init__(self, model_path, confidence_thre=0.25, nms_iou=0.7, task='detect') -> None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.task = task
        self.sess = rt.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.out_name = [output.name for output in self.sess.get_outputs()]
        self.metadata = self.sess.get_modelmeta().custom_metadata_map

        self.img_size = eval(self.metadata['imgsz']) if self.metadata else [640, 640]

        self.non_max_suppression = NonMaxSuppression(
            conf_thres=confidence_thre,
            iou_thres=nms_iou
        )
        self.letter_box = LetterBox(self.img_size, auto=False, scaleFill=False, scaleup=True, stride=32)

    def get_names_and_colors(self):
        # metadata

        names = eval(self.metadata['names'])
        COLORS = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')
        return names, COLORS

    def __call__(self, img):
        self.im = img
        # 前处理
        self.im0 = self.preprocess(im=self.im)
        # 推理
        time3 = time.time()
        self.preds = self.sess.run(
            self.out_name, {self.input_name: [self.im0]})
        # 后处理
        if self.task == 'detect':
            results = self.postprocess(preds=self.preds)
        else:
            results = self.postprocess_cls(self.preds)
        return results

    def preprocess(self, im):
        """
        前处理
        """
        im1 = self.letter_box(image=im)
        im2 = im1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im3 = np.ascontiguousarray(im2)  # contiguous
        im4 = im3.astype(np.float32) / 255.0
        return im4

    def postprocess(self, preds):
        """
        后处理
        """
        p = self.non_max_suppression(preds)

        results = []

        if len(p[0]) == 0:
            return []

        for i, pred in enumerate(p):
            pred[:, :4] = self.scale_boxes(
                self.im0.shape[1:], pred[:, :4], self.im.shape)

            results.append(pred.tolist())
        return results

    def postprocess_cls(self, preds):
        return [np.argmax(preds[0]), np.max(preds[0])]

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """
        缩放矩形框
        """
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                  2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(
            0, img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(
            0, img0_shape[0])  # y1, y2

        return boxes


class OnnxRuntimeInference:
    def __init__(self, model_path, **kwargs) -> None:
        if model_path.__contains__("damoyolo"):
            # damo-yolo
            self.model = DAMO_YOLO(model_path, **kwargs)
        else:
            self.model = YoloInference(model_path, **kwargs)

    def __call__(self, img):
        return self.model(img)

    def get_names_and_colors(self):
        return self.model.get_names_and_colors()


class DAMO_YOLO():
    """
    from modelscope.exporters import Exporter
    model_id = 'damo/cv_tinynas_object-detection_damoyolo_facemask'
    model = Model.from_pretrained(model_id)
    Exporter.from_model(model).export_onnx(
    input_shape=(1, 3, 640, 640), output_dir="/sdk/pre_models/cv/yolov8/onnx/damo-yolo")
    """

    def __init__(self, model_path, confidence_thre=0.25, nms_iou=0.7, task='detect'):
        label_path = model_path.replace('onnx', 'txt')
        self.classes = [line.strip() for line in open(label_path, 'r', encoding='utf-8').readlines()]
        self.num_0class = len(self.classes)
        so = rt.SessionOptions()
        so.log_severity_level = 3
        self.session = rt.InferenceSession(model_path, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        self.confThreshold = confidence_thre
        self.nmsThreshold = nms_iou

    def __call__(self, frame):
        temp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded_image = np.ones((self.input_height, self.input_width, 3), dtype=np.uint8)
        ratio = min(self.input_height / temp_image.shape[0], self.input_width / temp_image.shape[1])
        neww, newh = int(temp_image.shape[1] * ratio), int(temp_image.shape[0] * ratio)
        temp_image = cv2.resize(temp_image, (neww, newh), interpolation=cv2.INTER_LINEAR)
        padded_image[:newh, :neww, :] = temp_image

        padded_image = padded_image.transpose(2, 0, 1)
        padded_image = np.expand_dims(padded_image, axis=0).astype(np.float32)

        # Inference
        results = self.session.run(None, {self.input_name: padded_image})
        # self.postprocess(results)
        bboxes = results[0][..., :4].squeeze(axis=0)
        scores = results[0][..., 4:].squeeze(axis=0)
        bboxes /= ratio
        boxes, confidences, classIds = [], [], []
        for i in range(bboxes.shape[0]):
            score = np.max(scores[i, :])
            if score < self.confThreshold:
                continue

            class_id = np.argmax(scores[i, :])
            x, y, xmax, ymax = bboxes[i, :].astype(np.int32)
            width, height = xmax - x, ymax - y
            boxes.append([x, y, width, height])
            classIds.append(class_id)
            confidences.append(score)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        res = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            res.append([left, top, left + width, top + height, confidences[i], classIds[i]])
        return [res]

    def get_names_and_colors(self):
        # metadata

        names = self.classes
        COLORS = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')
        return names, COLORS


def draw_box(img, boxs):
    for box in boxs:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
        color = [int(c) for c in (255, 255, 0)]
        text = '{}'.format(cls_id)

        img = cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


if '__main__' == __name__:
    # opt = parse_opt()
    # main(opt)
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # sess = rt.InferenceSession("/media/sunshine/浮生物语的外置空间/trained_model/ultralytics/person.onnx", providers=providers)
    #
    # input_name = sess.get_inputs()[0].name
    # out_name = [output.name for output in sess.get_outputs()]

    # onnx_client = OnnxRuntimeInference(model_path='/opt/sunshine/trained_model/damo-yolo/human_damoyolo.onnx')
    onnx_client = OnnxRuntimeInference(model_path='/opt/sunshine/trained_model/ultralytics/onnx/person.onnx')
    img = cv2.imread('../t/微信图片_20241219155220.jpg')
    # for _ in range(5):
    #     onnx_client(img)
    s = time.time()
    result = onnx_client(img)
    img = draw_box(img, result[0])
    cv2.imwrite('../t/result.jpg', img)
    print(time.time() - s)
    print(result)
