# author: sunshine
# datetime:2021/11/18 下午2:24
import time
import traceback
from queue import Queue
from mqtt_plugins import MqttPlugin
from threading import Thread
import json
from yolov8_recognition import object_recognition
import urllib.request
import numpy as np
import cv2
from config import cfg, logger

# 静态参数
mqtt_queue = cfg.mqtt_queue
mqtt_topic_pub_name = cfg.__dict__.get('topic_pub_name', '')

mqtt_qos = 1
client_id = str(int(time.time())) + '233333333'

infer_time = {}  # 记录推理时间


def fetch_image(image_url, timeout=10):
    """读取图片
    """
    if image_url.startswith('http'):
        # 远程图片地址
        try:
            resp = urllib.request.urlopen(image_url, timeout=timeout)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        except Exception as err:
            return None
    else:
        # 本地图片地址
        image = cv2.imread(image_url)
        return image


# 消息链接
def connect_task(mtqq_plugin):
    mtqq_plugin.connect()


def heartbeat(mqtt_object):
    while True:
        mqtt_object.send(cfg.topic_heart, json.dumps({"msg": "heart"}), 1)
        time.sleep(cfg.heart_interval)


def init_mqtt():
    """初始化mtqq
    """

    # 创建队列
    q = Queue(mqtt_queue)
    # 订阅端topic
    mtqq_object = MqttPlugin(
        clientId=client_id + cfg.model_name,
        host=cfg.mqtt_host,
        port=cfg.mqtt_port,
        topic=cfg.topic_sub_name,
        qos=mqtt_qos,
        q=q,
        username=cfg.mqtt_username,
        password=cfg.mqtt_password
    )

    # 订阅通知topic
    Thread(target=connect_task, args=(mtqq_object,)).start()

    if cfg.topic_heart:
        # 心跳线程
        Thread(target=heartbeat, args=(mtqq_object,)).start()
    return mtqq_object, q


def predict_task(mqtt_object, q):
    while True:
        try:
            data = json.loads(q.get())
            start_time = time.time()

            # 模型预测
            need_alert = False
            img_url = data.pop('img_url')
            points = data.get('points', None)
            try:
                img = fetch_image(img_url)
                try:
                    img_path = object_recognition.predict(img, points=points)
                    need_alert = True if img_path else False

                    # 返回预测结果
                    _data = {"code": 200, "msg": "predict success", "detail": img_path}
                except Exception as err:
                    traceback.print_exc()
                    _data = {"code": 400, "msg": "predict failed", "detail": str(err)}
            except Exception as err:
                _data = {"code": 400, "msg": "Picture path is incorrect!", "detail": str(err), }
            _data.update(
                # {"mode": cfg.model_name, "image_url": img_url, "cost time": time.time() - start_time})
                {"mode": cfg.model_name, "image_url": img_url})

            data.update(_data)
            logger.info(_data)
            # logger.info({"code": data['code'], "cost": time.time()-start})
            if need_alert:
                # 判断人员逗留
                pub_topic_name = data['pub_name'] if data.get("pub_name", "") else mqtt_topic_pub_name
                customParams = data.get('customParams', None)
                deviceld = str(data.get('deviceld', "default_id"))
                if deviceld not in infer_time:
                    infer_time[deviceld] = start_time
                if customParams is not None:
                    duration = data['customParams'].get("duration", 0)
                    if start_time - infer_time[deviceld] > duration:
                        logger.debug(f"send to topic: {pub_topic_name}")
                        mqtt_object.send(pub_topic_name, json.dumps(data), 1)
                else:
                    logger.debug(f"send to topic: {pub_topic_name}")
                    mqtt_object.send(pub_topic_name, json.dumps(data), 1)
            else:
                infer_time[str(data['deviceld'])] = start_time

        except Exception as err:
            logger.error(str(err))


def main():
    mqtt_object, q = init_mqtt()
    predict_task(mqtt_object, q)


if __name__ == '__main__':
    main()
