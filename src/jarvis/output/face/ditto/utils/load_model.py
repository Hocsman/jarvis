# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# loads PyTorch weights only; the ONNX models open their own sessions
# (aux_models/), and Jarvis builds its own TensorRT engines (engines.py).


def load_model(model_path: str, device: str = "cuda", **kwargs):
    if model_path.endswith(".pt") or model_path.endswith(".pth"):
        return create_model(model_path, device, **kwargs)
    raise ValueError(f"Unsupported model file type: {model_path}")


def create_model(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..models.modules",
    **kwargs,
):
    import importlib

    module = getattr(importlib.import_module(package_name, __package__), module_name)
    model = module(**kwargs)
    model.load_model(model_path).to(device)
    return model
