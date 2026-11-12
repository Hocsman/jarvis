# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# registers one still image handed over as an RGB array; videos, the file
# loader and the smoothing of a sequence of frames are gone.
from .source2info import Source2Info


class AvatarRegistrar:
    """
    rgb image -> source_info
    """
    def __init__(
        self,
        insightface_det_cfg,
        landmark106_cfg,
        landmark203_cfg,
        appearance_extractor_cfg,
        motion_extractor_cfg,
    ):
        self.source2info = Source2Info(
            insightface_det_cfg,
            landmark106_cfg,
            landmark203_cfg,
            appearance_extractor_cfg,
            motion_extractor_cfg,
        )

    def register(self, rgb, **kwargs):
        """
        kwargs:
            crop_scale: 2.3
            crop_vx_ratio: 0
            crop_vy_ratio: -0.125
            crop_flag_do_rot: True
        """
        info = self.source2info(rgb, **kwargs)
        return {
            "x_s_info_lst": [info["x_s_info"]],
            "f_s_lst": [info["f_s"]],
            "eye_open_lst": [info["eye_open"]],
            "eye_ball_lst": [info["eye_ball"]],
            "sc": info["x_s_info"]["kp"].flatten(),
        }

    def __call__(self, *args, **kwargs):
        return self.register(*args, **kwargs)
