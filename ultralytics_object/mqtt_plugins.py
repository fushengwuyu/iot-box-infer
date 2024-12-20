import paho.mqtt.client as mqtt
from config import logger


class MqttPlugin(object):
    def __init__(self, clientId, host, port, topic, qos, q, username="", password=""):
        logger.info(f"{clientId}, {host}:{port}, {topic}")
        self.connected = False
        self.host = host
        self.port = port
        # self.username = username
        # self.password = password
        self.mqttc = mqtt.Client(clientId, clean_session=True)
        if username and password:
            self.mqttc.username_pw_set(username, password)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_disconnect = self.on_disconnect
        self.mqttc.on_message = self.on_message
        self.topic = topic
        self.qos = qos
        self.q = q

    def on_connect(self, mqttc, obj, flags, rc):
        self.mqttc.subscribe(self.topic, self.qos)
        self.connected = True

    def on_disconnect(self, client, userdata, rc):
        self.connected = False

    def on_message(self, mqttc, obj, msg):
        try:
            self.q.put_nowait(msg.payload)
        except:
            logger.info(msg.payload)
        # self.q.put(msg.payload)

    def connect(self):
        self.mqttc.connect(self.host, self.port, 60)
        self.mqttc.loop_forever()

    def send(self, topic, data, qos):
        self.mqttc.publish(topic, data, qos)
