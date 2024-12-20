#FROM silverlogic/python3.8
FROM registry.cn-chengdu.aliyuncs.com/fushengwuyu/onnxruntime-gpu:latest
WORKDIR /root/ultralytics_object/
COPY . .

ENV LANG C.UTF-8
RUN mkdir ~/.pip
RUN pip3 install -r /root/ultralytics_object/requirements.txt -i https://mirrors.aliyun.com/pypi/simple

CMD python3.8 run.py


#iot-mqtt-box:v1