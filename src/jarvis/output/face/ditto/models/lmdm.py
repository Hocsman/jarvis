# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# PyTorch only, and the sampling can be captured once as a CUDA graph in half
# precision and replayed for every call (capture_graph).
import torch
from ..utils.load_model import load_model


class LMDM:
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["module_name"] = "LMDM"

        self.model, _ = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.motion_feat_dim = kwargs.get("motion_feat_dim", 265)
        self.audio_feat_dim = kwargs.get("audio_feat_dim", 1024+35)
        self.seq_frames = kwargs.get("seq_frames", 80)
        self._graph = None

    def setup(self, sampling_timesteps):
        self.model.setup(sampling_timesteps)

    def capture_graph(self, sampling_timesteps):
        """Record one sampling as a CUDA graph; later calls replay it.

        Captured per thread (``thread_local``): other threads of the process
        keep using the GPU while it is recorded.
        """
        model = self.model
        kp = torch.zeros(1, model.motion_feat_dim, device=self.device)
        aud = torch.zeros(1, model.seq_frames, model.audio_feat_dim, device=self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.float16):
                    model.ddim_sample(kp, aud, sampling_timesteps)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, capture_error_mode="thread_local"):
            with torch.autocast("cuda", dtype=torch.float16):
                out = model.ddim_sample(kp, aud, sampling_timesteps)
        self._graph = (graph, kp, aud, out, sampling_timesteps)

    def __call__(self, kp_cond, aud_cond, sampling_timesteps):
        if self._graph is not None and self._graph[4] == sampling_timesteps:
            graph, kp, aud, out, _ = self._graph
            kp.copy_(torch.from_numpy(kp_cond))
            aud.copy_(torch.from_numpy(aud_cond))
            graph.replay()
            return out.float().cpu().numpy()
        with torch.no_grad():
            return self.model.ddim_sample(
                torch.from_numpy(kp_cond).to(self.device),
                torch.from_numpy(aud_cond).to(self.device),
                sampling_timesteps,
            ).cpu().numpy()
