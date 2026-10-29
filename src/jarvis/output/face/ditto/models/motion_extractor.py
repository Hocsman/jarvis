# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# PyTorch only.
import torch
from ..utils.load_model import load_model


class MotionExtractor:
    def __init__(self, model_path, device="cuda"):
        self.model = load_model(model_path, device=device, module_name="MotionExtractor")
        self.device = device

        self.output_names = [
            "pitch",
            "yaw",
            "roll",
            "t",
            "exp",
            "scale",
            "kp",
        ]

    def __call__(self, image):
        """
        image: np.ndarray, shape (1, 3, 256, 256), RGB, 0-1
        """
        with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
            pred = self.model(torch.from_numpy(image).to(self.device))
        outputs = {name: pred[i].float().cpu().numpy() for i, name in enumerate(self.output_names)}
        outputs["exp"] = outputs["exp"].reshape(1, -1)
        outputs["kp"] = outputs["kp"].reshape(1, -1)
        return outputs
