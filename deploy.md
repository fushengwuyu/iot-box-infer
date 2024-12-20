模型地址: /home/nvidia/software/models

注意： 使用镜像前，需将docker运行时默认设置为nvidia。

#### 1. 启动服务

服务通过docker-compose方式启动，样例配置如下，其它服务如fire等，可参考如下配置新增：

```yaml
version: "3"
services:
  person:
    image: qy_object:latest
    deploy:
      replicas: 2 # 实例启动的个数
    environment:
      - experiment=qy  # 实验名称，随意写
      - model_name=person  # 模型名称，["fire", "workclothes", "hat", "person", "phone"]
      - engine_path=/root/models/person.onnx  # 模型地址
      - confidence_thre=0.7  # 阈值
      - nms_iou=0.6  # 并交比
      - mqtt_host=192.168.0.15  
      - mqtt_port=1883
      - mqtt_queue=3  # mqtt队列长度
      - topic_sub_name=qy/box/penson  # 订阅的topic名称， 共享订阅采用 $$SHARE/qy/test 形式，推送用 test
      - minio_ip=192.168.0.79
      - minio_port=9000
      - minio_access=minio
      - minio_secret=12345678
      - bucket_name=img  # minio bucket地址
      - logging_level=10  # 日志等级，{10： debug， 20： info， 30： WARNING， 40： ERROR，50：CRITICAL }
      - face_url=http://192.168.0.15:5003/face/face_retrieve  # 人脸识别服务地址，电子围栏启动人脸识别才需要使用，其他任务用不到
    volumes:
      - /sdk/sunshine/trained_model/ultralytics/:/root/models/  # 磁盘映射
    restart: always
```



启动命令：

```shell
docker-compose up -d person
```



最终消息推送至topic :  `{experiment}/{model_name}/{data["key"]}`

以样例yaml为例，如接收到的消息如下：

```json
{ "img_url": "http://192.168.0.174:8080/box/original/1683168519975_87176819769473.jpg",
  "key": "device_001"
}
```

则推送topic为： `qy/person/device_001`

推送的消息体为：

```json
{
    "key": "device_001", 
 	"code": 200, 
 	"msg": "predict success", 
 	"detail": "person/202305/recognized_1684805037.jpg",   # 识别成功的图片地址，在minio中的路径需加上bucket
 	"mode": "person", 
	"image_url": "http://192.168.0.174:8080/box/original/1683168519975_87176819769473.jpg"  # 原始图片路径
}
```



#### 2.服务列表

* fire  火灾

* workclothes  工服

* hat  安全帽

* person  人员检测（区域入侵）

* fence  电子围栏

* phone  打电话

* fall  跌倒

  

  备注： 单人作业和人数异常，可通过person服务实现，返回的消息中带有人员数量。消息如下：

  ```json
  {"key": "device_001", "code": 200, "msg": "predict success", "detail": "person/202306/recognized_1687167250_8299918.jpg", "mode": "person", "image_url": "https://www.healthnews.com.tw/imageFile/202209/b51e951f6f86bbc78f4ac66e305ce787_l.jpg", "count": 2}
  ```

  

  

