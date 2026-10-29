# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# the pickle is read with an unpickler that admits numpy arrays only, and the
# MediaPipe landmark model is not part of the avatar registrar; every model path
# is resolved inside the model folder, never against the working directory.
import os

from ...safe_pickle import load as load_pkl


def parse_cfg(cfg_pkl, data_root):

    def _check_path(p):
        if os.path.isabs(p) or ".." in p.replace("\\", "/").split("/"):
            raise ValueError(f"model path outside the model folder: {p}")
        return os.path.join(data_root, p)

    cfg = load_pkl(cfg_pkl)

    base_cfg = cfg["base_cfg"]
    audio2motion_cfg = cfg["audio2motion_cfg"]
    default_kwargs = cfg["default_kwargs"]

    for k in base_cfg:
        if k != "landmark478_cfg":
            base_cfg[k]["model_path"] = _check_path(base_cfg[k]["model_path"])

    audio2motion_cfg["model_path"] = _check_path(audio2motion_cfg["model_path"])

    avatar_registrar_cfg = {
        k: base_cfg[k]
        for k in [
            "insightface_det_cfg",
            "landmark106_cfg",
            "landmark203_cfg",
            "appearance_extractor_cfg",
            "motion_extractor_cfg",
        ]
    }

    stitch_network_cfg = base_cfg["stitch_network_cfg"]
    warp_network_cfg = base_cfg["warp_network_cfg"]
    decoder_cfg = base_cfg["decoder_cfg"]

    condition_handler_cfg = {
        k: audio2motion_cfg[k]
        for k in [
            "use_emo",
            "use_sc",
            "use_eye_open",
            "use_eye_ball",
            "seq_frames",
        ]
    }

    lmdm_cfg = {
        k: audio2motion_cfg[k]
        for k in [
            "model_path",
            "device",
            "motion_feat_dim",
            "audio_feat_dim",
            "seq_frames",
        ]
    }

    wav2feat_cfg = {
        "w2f_cfg": base_cfg["hubert_cfg"],
        "w2f_type": audio2motion_cfg["w2f_type"],
    }

    return [
        avatar_registrar_cfg,
        condition_handler_cfg,
        lmdm_cfg,
        stitch_network_cfg,
        warp_network_cfg,
        decoder_cfg,
        wav2feat_cfg,
        default_kwargs,
    ]
