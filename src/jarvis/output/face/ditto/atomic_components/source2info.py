# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# a single still image only, a portrait with no face raises NoFaceFound, and the
# eyes are measured from the 203-point landmarks (utils/eye_info.py) instead of
# MediaPipe.
import cv2
import numpy as np

from ..aux_models.insightface_det import InsightFaceDet
from ..aux_models.insightface_landmark106 import Landmark106
from ..aux_models.landmark203 import Landmark203
from ..models.appearance_extractor import AppearanceExtractor
from ..models.motion_extractor import MotionExtractor
from ..utils.crop import crop_image
from ..utils.eye_info import eye_info_from_lmk203


class NoFaceFound(ValueError):
    """The image holds no face the detector can find."""


class Source2Info:
    def __init__(
        self,
        insightface_det_cfg,
        landmark106_cfg,
        landmark203_cfg,
        appearance_extractor_cfg,
        motion_extractor_cfg,
    ):
        self.insightface_det = InsightFaceDet(**insightface_det_cfg)
        self.landmark106 = Landmark106(**landmark106_cfg)
        self.landmark203 = Landmark203(**landmark203_cfg)

        self.appearance_extractor = AppearanceExtractor(**appearance_extractor_cfg)
        self.motion_extractor = MotionExtractor(**motion_extractor_cfg)

    def _crop(self, img, **kwargs):
        # img_rgb -> det -> landmark106 -> landmark203 -> crop
        det, _ = self.insightface_det(img)
        boxes = det[np.argsort(-(det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))]
        if len(boxes) == 0:
            raise NoFaceFound("no face in the portrait")
        lmk_for_track = self.landmark106(img, boxes[0])  # 106

        crop_dct = crop_image(
            img,
            lmk_for_track,
            dsize=self.landmark203.dsize,
            scale=1.5,
            vy_ratio=-0.1,
            pt_crop_flag=False,
        )
        lmk203 = self.landmark203(crop_dct["img_crop"], crop_dct["M_c2o"])

        ret_dct = crop_image(
            img,
            lmk203,
            dsize=512,
            scale=kwargs.get("crop_scale", 2.3),
            vx_ratio=kwargs.get("crop_vx_ratio", 0),
            vy_ratio=kwargs.get("crop_vy_ratio", -0.125),
            flag_do_rot=kwargs.get("crop_flag_do_rot", True),
            pt_crop_flag=False,
        )

        return ret_dct["img_crop"], ret_dct["M_c2o"], lmk203

    @staticmethod
    def _img_crop_to_bchw256(img_crop):
        rgb_256 = cv2.resize(img_crop, (256, 256), interpolation=cv2.INTER_AREA)
        rgb_256_bchw = (rgb_256.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)
        return rgb_256_bchw

    def __call__(self, img, **kwargs):
        """
        img: rgb, uint8
        kwargs: optional crop cfg
            crop_scale: 2.3
            crop_vx_ratio: 0
            crop_vy_ratio: -0.125
            crop_flag_do_rot: True
        """
        img_crop, M_c2o, lmk203 = self._crop(img, **kwargs)

        # The landmarks are in the image's pixels; the eyes are measured in the crop's.
        M_o2c = cv2.invertAffineTransform(M_c2o[:2])
        lmk_crop = lmk203 @ M_o2c[:, :2].T + M_o2c[:, 2]
        eye_open, eye_ball = eye_info_from_lmk203(lmk_crop, img_crop.shape[0])

        rgb_256_bchw = self._img_crop_to_bchw256(img_crop)
        kp_info = self.motion_extractor(rgb_256_bchw)
        fs = self.appearance_extractor(rgb_256_bchw)

        return {
            "x_s_info": kp_info,
            "f_s": fs,
            "eye_open": eye_open,   # [1, 2]
            "eye_ball": eye_ball,    # [1, 6]
        }
