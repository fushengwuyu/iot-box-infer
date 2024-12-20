# author: sunshine
# datetime:2023/6/5 上午11:04
import argparse

import onnxruntime as rt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument("--onnx", default="/sdk/sunshine/trained_model/ultralytics/onnx/fire_and_smoke.onnx", type=str,
                    help='onnx path')
parser.add_argument("--saveEngine", default='/sdk/sunshine/trained_model/ultralytics/tensorrt/fire_and_smoke.trt',
                    type=str, help='save engine path')
parser.add_argument("--half", default=False, type=bool, help='是否半精度')

args = parser.parse_args()
LOGGER = trt.Logger()


def export_engine(onnx_path, trt_path, workspace=4, verbose=False, is_half=True, custom_map=None):
    """YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt."""

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx_sess = rt.InferenceSession(onnx_path, providers=providers)

    custom_metadata_map = onnx_sess.get_modelmeta().custom_metadata_map
    if custom_map is not None:
        custom_metadata_map.update(custom_map)
    """
    {'names': "{0: 'normal', 1: 'phone', 2: 'smoke'}", 'imgsz': '[224, 224]', 'batch': '1'
    """
    metadata = {"names": custom_metadata_map['names'], "imgsz": custom_metadata_map['imgsz'],
                'batch': custom_metadata_map['batch']}

    # trt_path = onnx_path.split('.')[0] + '.trt'

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

    # inputs = [network.get_input(i) for i in range(network.num_inputs)]
    # outputs = [network.get_output(i) for i in range(network.num_outputs)]
    # for inp in inputs:
    #     LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    # for out in outputs:
    #     LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
    #
    # LOGGER.info(
    #     f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and is_half else 32} engine as {trt_path}')
    if builder.platform_has_fast_fp16 and is_half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Write file
    with builder.build_engine(network, config) as engine, open(trt_path, 'wb') as t:
        # Metadata
        meta = json.dumps(metadata)
        t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
        t.write(meta.encode())
        # Model
        t.write(engine.serialize())


if __name__ == '__main__':
    export_engine(args.onnx,
                  trt_path=args.saveEngine,
                  is_half=args.half,
                  custom_map={})
