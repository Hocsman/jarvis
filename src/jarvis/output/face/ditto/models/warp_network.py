# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# loads the PyTorch network only; Jarvis exports it and runs it through TensorRT
# (engines.py), so it is never called here.
from ..utils.load_model import load_model


class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        self.model = load_model(model_path, device=device, module_name="WarpingNetwork")
        self.device = device
