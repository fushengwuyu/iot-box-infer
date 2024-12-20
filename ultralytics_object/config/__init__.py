# author: sunshine
# datetime:2024/3/12 上午11:21
from omegaconf import OmegaConf
import os
import logging
from argparse import Namespace

cur_dir = os.path.dirname(os.path.abspath(__file__))
config = OmegaConf.load(os.path.join(cur_dir, 'config.yaml'))

cfg = Namespace(**config.basic, **config.mqtt, **config.minio, **config.face, **config.web, **config)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=cfg.logging_level, format=LOG_FORMAT)
logger = logging.getLogger("app")

alert_label = cfg.alarm_map[cfg.model_name]

if cfg.engine_path.endswith('onnx'):
    backend = 'onnxruntime'  # ['onnxruntime', 'ultralytics', 'tensorrt]
elif cfg.engine_path.endswith('trt'):
    backend = 'tensorrt'
else:
    backend = 'ultralytics'
file_server = cfg.file_server_type  # ['minio', 'web']

task = 'detect'
