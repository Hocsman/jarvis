"""TensorRT engines for Ditto's warping network and decoder (`face.spec.md`,
"Model files and engines").

The two networks do most of the work of drawing a frame. They are exported to
ONNX in half precision and compiled on the machine that runs them, once per
GPU, TensorRT version and weights; an engine built for anything else is built
again, never loaded.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

from ...debug import debug_log

EXPORT_VERSION = 1   # what the export below produces; a change to it builds new engines
_WORKSPACE = 4 << 30


def engine_key(gpu: str, capability: Tuple[int, int], trt_version: str, warp_size: int, decoder_size: int) -> str:
    text = f"{EXPORT_VERSION}|{gpu}|{capability[0]}.{capability[1]}|{trt_version}|{warp_size}|{decoder_size}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def ensure_engines(root: Path, warp_pth: str, decoder_pth: str) -> Tuple[Path, Path]:
    """The paths of the two engines for this GPU, building whichever is missing."""
    import tensorrt as trt
    import torch

    gpu = torch.cuda.get_device_name(0)
    key = engine_key(gpu, torch.cuda.get_device_capability(0), trt.__version__,
                     os.path.getsize(warp_pth), os.path.getsize(decoder_pth))
    folder = root / "engines" / key
    warp_engine, decoder_engine = folder / "warp_fp16.engine", folder / "decoder_fp16.engine"
    if warp_engine.is_file() and decoder_engine.is_file():
        return warp_engine, decoder_engine

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "key.json").write_text(json.dumps({
        "export": EXPORT_VERSION, "gpu": gpu, "tensorrt": trt.__version__,
        "warp_bytes": os.path.getsize(warp_pth), "decoder_bytes": os.path.getsize(decoder_pth),
    }, indent=1), encoding="utf-8")
    print(f"  🔧 Face: building the TensorRT engines for {gpu} (about two minutes, once)", flush=True)
    feature_shape = _export_warp(warp_pth, folder / "warp_fp16.onnx")
    _export_decoder(decoder_pth, folder / "decoder_fp16.onnx", feature_shape)
    for onnx, engine in ((folder / "warp_fp16.onnx", warp_engine), (folder / "decoder_fp16.onnx", decoder_engine)):
        if not engine.is_file():
            _build(onnx, engine)
        onnx.unlink(missing_ok=True)
    return warp_engine, decoder_engine


def _export_warp(warp_pth: str, onnx_path: Path) -> Tuple[int, ...]:
    import torch

    from .ditto.models.warp_network import WarpNetwork

    model = WarpNetwork(warp_pth).model.half().eval()
    f_s = torch.randn(1, 32, 16, 64, 64, device="cuda", dtype=torch.float16)
    kp = 0.1 * torch.randn(1, 21, 3, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        shape = tuple(model(f_s, kp, kp).shape)
        torch.onnx.export(model, (f_s, kp, kp), str(onnx_path), input_names=["feature_3d", "kp_source", "kp_driving"],
                          output_names=["out"], opset_version=20, dynamo=False)
    del model
    torch.cuda.empty_cache()
    return shape


def _export_decoder(decoder_pth: str, onnx_path: Path, feature_shape: Tuple[int, ...]) -> None:
    import torch

    from .ditto.models.decoder import Decoder

    model = Decoder(decoder_pth).model.half().eval()
    feature = torch.randn(*feature_shape, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(model, (feature,), str(onnx_path), input_names=["feature"], output_names=["image"],
                          opset_version=17, dynamo=False)
    del model
    torch.cuda.empty_cache()


def _build(onnx_path: Path, engine_path: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # The ONNX graph is already in half precision; a strongly typed network keeps it so.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not read {onnx_path.name}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE)
    started = time.monotonic()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT could not build {engine_path.name}")
    partial = engine_path.with_suffix(".partial")
    partial.write_bytes(bytes(plan))
    partial.replace(engine_path)
    debug_log(f"face: built {engine_path.name} in {time.monotonic() - started:.0f}s", "face")


class Engine:
    """One engine, its inputs and outputs held as static CUDA tensors.

    It runs on its own stream: on the default one TensorRT adds
    synchronisations of its own. The returned tensors are overwritten by the
    next call.
    """

    def __init__(self, path: Path) -> None:
        import tensorrt as trt
        import torch

        types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not load {Path(path).name}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.tensors: Dict[str, "torch.Tensor"] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.tensors[name] = torch.empty(shape, dtype=types[self.engine.get_tensor_dtype(name)], device="cuda")
            self.context.set_tensor_address(name, self.tensors[name].data_ptr())

    def __call__(self, **inputs):
        import torch

        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            for name, value in inputs.items():
                self.tensors[name].copy_(value)
            self.context.execute_async_v3(self.stream.cuda_stream)
        caller.wait_stream(self.stream)
        return self.tensors


def warp_and_decode(warp_engine: Path, decoder_engine: Path) -> Tuple[Callable, Callable]:
    """The two calls Ditto's stream makes, backed by the engines."""
    warp_e, decode_e = Engine(warp_engine), Engine(decoder_engine)

    def warp(f_s, x_s, x_d):
        # A copy: the decoder, on another thread, may still be reading the
        # previous output when the next frame overwrites the engine's buffer.
        return warp_e(feature_3d=f_s, kp_source=x_s, kp_driving=x_d)["out"].clone()

    def decode(feature):
        return decode_e(feature=feature)["image"]

    return warp, decode
