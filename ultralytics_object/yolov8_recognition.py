# author: sunshine
# datetime:2022/8/11 下午2:23
from threading import Thread

import cv2
import numpy as np
import time
from minio import Minio
from datetime import datetime
from config import logger
from queue import Queue
import requests
import io
from config import cfg, alert_label, backend, file_server, task

if backend == 'onnxruntime':
    from onnxruntime_infer import OnnxRuntimeInference as TaskPredictor
elif backend == 'tensorrt':
    from tensorrt_infer import TensorRTInference as TaskPredictor
else:
    from ultralytics_Infer import UltralyticsInference as TaskPredictor

client = Minio(f'{cfg.minio_ip}:{cfg.minio_port}',
               access_key=cfg.minio_access,
               secret_key=str(cfg.minio_secret),
               secure=False)


class FileUtils:
    def __init__(self, model_name, q_size=20):
        self.model_name = model_name
        self.q = Queue(q_size)

    def put_file(self, img):
        # 生成文件路径
        file_name = "recognized_{}.jpg".format(str(time.time()).replace('.', '_'))
        object_name = "/".join([self.model_name, datetime.now().strftime("%Y%m"), file_name])
        self.q.put((img, object_name))
        return object_name

    def upload_image(self, img, object_name):
        try:
            field = object_name.split('/')
            file_name, img_dir = field[-1], '/'.join(field[:-1])
            data = cv2.imencode(".jpg", img)[1].tobytes()
            files = [
                ('file', (file_name, data, 'image/png'))
            ]

            result = requests.post(cfg.upload_url + img_dir, files=files)
            if result.status_code == 200:
                path = img_dir + "/" + file_name
            else:
                path = ""
        except Exception as err:
            path = ""
            logger.error(str(err))
        return path

    def upload_image_minio(self, img, bucket_name, object_name):
        try:
            success, encoded_image = cv2.imencode(".jpg", img)
            # 将数组转为bytes
            img_bytes = encoded_image.tobytes()
            value_as_a_stream = io.BytesIO(img_bytes)

            # 传入队列中
            client.put_object(bucket_name, object_name, value_as_a_stream, length=len(encoded_image))
            return object_name
        except Exception as err:
            logger.error(err)
            return ''

    def run(self, bucket_name, file_server):
        while True:
            data = self.q.get()
            img, object_name = data
            if file_server == 'minio':
                self.upload_image_minio(img, bucket_name, object_name)
            else:
                self.upload_image(img, object_name)


file_client = FileUtils(model_name=cfg.model_name)

# 启动线程
Thread(target=file_client.run, args=(cfg.bucket_name, file_server)).start()


class ObjectInference:
    def __init__(self, model_name, engine_path, bucket_name, alert_label, confidence_thre=0.25, nms_iou=0.7,
                 upload=True, task='detect', distinct=0):
        self.alert_label = alert_label
        self.upload = upload
        self.task = task
        self.predictor = TaskPredictor(engine_path, confidence_thre=confidence_thre, nms_iou=nms_iou, task=task)
        self.confidence_thre = confidence_thre
        self.model_name = model_name
        self.bucket_name = bucket_name
        self.names, self.colors = self.predictor.get_names_and_colors()
        self.distinct = distinct
        self.box_count = 0
        self.infer_time = time.time()

    def __call__(self, img, points=None):
        if self.model_name.startswith("interaction"):
            return self.interaction_predict(img, points=points)
        else:
            return self.predict(img, points=points)

    def distinct_alarm(self, count):
        """
        过滤重复预警，在self.distinct时间内，若图片中检测的目标个数没变化，则不告警。
        :param count:
        :return:
        """

        current_time = time.time()
        if count == self.box_count and current_time - self.infer_time < self.distinct:
            flag = False
        else:
            self.infer_time = current_time
            flag = True

        self.box_count = count

        return flag

    def predict(self, img, points=None):
        start = time.time()
        results = self.predictor(img)  # 对图像进行预测
        infer_time = time.time()

        count = len(results[0])
        filter_boxes = [box for box in results[0] if
                        int(box[-1]) in self.alert_label and self.box_intersection(box[:4], points)]
        flag = len(filter_boxes) > 0

        if flag and self.upload and self.distinct_alarm(count):
            img = self.draw_box(img, boxs=filter_boxes)
            img_path = file_client.put_file(img)
        else:
            img_path = ""
        end_time = time.time()

        logger.debug(
            f'infer cost time: {infer_time - start}, upload time: {end_time - infer_time}, total: {end_time - start}')
        return img_path

    def interaction_predict(self, img, points=None):
        """
        多目标检测，检测两个目标是否有交集
        :param img:
        :param points:
        :param task:
        :return:
        """
        start = time.time()
        results = self.predictor(img)  # 对图像进行预测
        count = len(results[0])
        infer_time = time.time()
        flag = False

        filter_boxes = [(int(box[-1]), box) for box in results[0] if
                        int(box[-1]) in self.alert_label and self.box_intersection(box[:4], points)]

        distinct_box = list(set([fb[0] for fb in filter_boxes]))

        if len(distinct_box) > 1:
            # 有效
            part1 = [fb for fb in filter_boxes if fb[0] == distinct_box[0]]
            part2 = [fb for fb in filter_boxes if fb[0] == distinct_box[1]]
            for c1, box1 in part1:
                for c2, box2 in part2:

                    # 直接判断是否存在交集
                    x1, y1, x2, y2, conf1, cls1 = box1
                    x3, y3, x4, y4, conf2, cls2 = box2
                    if x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4:
                        ...
                    else:
                        flag = True
        else:
            flag = False
        if flag and self.upload and self.distinct_alarm(count):
            # 画出矩形框
            img = self.draw_box(img, boxs=[fb[1] for fb in filter_boxes])
            img_path = file_client.put_file(img)
        else:
            img_path = ""
        end_time = time.time()

        logger.debug(
            f'infer cost time: {infer_time - start}, upload time: {end_time - infer_time}, total: {end_time - start}')
        return img_path

    def box_intersection(self, rectangle, points):
        """
        判断矩形框是否在多边形内
        :param box:
        :param points:
        :return:
        """
        if points:
            x1, y1, x2, y2 = rectangle
            testContour = np.array([points], dtype=np.float32)
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            exist = cv2.pointPolygonTest(testContour, center_point, False)
            return exist >= 0
        else:
            return True

    def draw_box(self, img, boxs):
        for box in boxs:
            x1, y1, x2, y2, conf, cls_id = box
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            color = [int(c) for c in self.colors[cls_id]]
            text = '{}: {:.3f}'.format(self.names[cls_id], conf)

            img = cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 1)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        return img

    def ele_fence(self, img, rectangle, points, fence_flag, text):
        x1, y1, x2, y2 = rectangle
        testContour = np.array([points], dtype=np.int)
        cv2.polylines(img, testContour, True, (0, 255, 255), 2)

        center_point = (int((x1 + x2) / 2), int(y2))
        exist = cv2.pointPolygonTest(testContour, center_point, False)
        logger.debug(f'rectangle: {(x1, y1, x2, y2)}, center_point: {center_point}, exist: {exist}')

        if exist >= 0:
            img = cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            fence_flag = True

        return fence_flag, img


if __name__ == '__main__':
    engine = ObjectInference(
        model_name=cfg.model_name,
        engine_path=cfg.engine_path,
        bucket_name=cfg.bucket_name,
        alert_label=[0, 7],
        confidence_thre=cfg.confidence_thre,
        nms_iou=cfg.nms_iou,
        upload=True
    )
    img = cv2.imread('../t/微信截图_20241216175744.png')
    start = time.time()

    # results = engine.predictor(img)  # 对图像进行预测
    # img = engine.draw_box(img, results[0])
    # cv2.imwrite('../t/result.jpg', img)
    x1, y1, x2, y2 = [518.2283592224121, 234.650306224823, 652.4664239883423, 334.2034478187561]

    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    result = engine.predict(img)
    print(result)
    # # result = engine.predict(img)
    # print(time.time() - start)
    # print(result)
else:
    object_recognition = ObjectInference(
        model_name=cfg.model_name,
        engine_path=cfg.engine_path,
        bucket_name=cfg.bucket_name,
        alert_label=alert_label,
        confidence_thre=cfg.confidence_thre,
        nms_iou=cfg.nms_iou,
        distinct=cfg.__dict__.get('distinct', 0)
    )
