# author: sunshine
# datetime:2023/5/17 上午10:41
from ultralytics import YOLO
import numpy as np


class UltralyticsInference:
    def __init__(self, engine_path, confidence_thre=0.25, nms_iou=0.7):
        self.model = YOLO(engine_path, task='detect')
        self.confidence_thre = confidence_thre
        self.nms_iou = nms_iou
        self.names = None
        self.COLORS = None

    def __call__(self, img):
        results = self.model(img, conf=self.confidence_thre, iou=self.nms_iou)  # 对图像进行预测
        output = []
        for result in results:
            # print(result)
            boxes = result.boxes
            if boxes:
                boxes = boxes.data.cpu().numpy().tolist()
                output.append(boxes)
            else:
                output.append([])
        return output

    def get_names_and_colors(self):
        if self.names is None:
            self.names = list(self.model.predictor.model.names.values())
            self.COLORS = np.random.randint(0, 255, size=(len(self.names), 3), dtype='uint8')
        return self.names, self.COLORS


if __name__ == '__main__':
    engine_path = "/opt/sunshine/trained_model/ultralytics/onnx/hat.onnx"
    model = YOLO(engine_path, task='detect')
    result = model.predict("../t/1.png")
    for r in result:
        r.save('../t/1_result.jpg')
