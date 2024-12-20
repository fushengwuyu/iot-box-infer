# author: sunshine
# datetime:2024/3/7 下午3:10
import time

from pytrt.yolo_infer import YoloModel
import cv2
labels = open('/sdk/sunshine/coco.names', 'r', encoding='utf-8').readlines()
labels = [l.strip('\n') for l in labels]

client = YoloModel('/sdk/sunshine/pytensorrt/demo/yolov8n.transd.engine', "V8", labels)
for _ in range(5):
    client.single_inference('3.jpg')

s = time.time()
img, objs = client.single_inference('3.jpg')
print(time.time()-s)
# img = client.single_inference('3.jpg')
# cv2.imwrite('result.jpg', img)
print(objs)
