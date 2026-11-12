# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee),
# stream_pipeline_online.py: streaming only, from a still portrait registered once;
# each frame is Ditto's own 512x512 square handed to a callback (no paste-back
# into the portrait, no video file); the warping network and the decoder are
# TensorRT engines handed in by the caller; the motion sampling is replayed from
# a CUDA graph; the models that only register the portrait are released after
# it; the workers stop on request; a frame's controls are taken as it is drawn.
import queue
import threading
import traceback

import numpy as np
import torch

from .atomic_components.audio2motion import Audio2Motion
from .atomic_components.avatar_registrar import AvatarRegistrar
from .atomic_components.cfg import parse_cfg
from .atomic_components.condition_handler import ConditionHandler
from .atomic_components.motion_stitch import MotionStitch
from .atomic_components.wav2feat import Wav2Feat
from ..timeline import CHUNK_FRAMES, CHUNK_SAMPLES, CHUNKSIZE


class Stream:
    def __init__(self, cfg_pkl, data_root):
        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            _warp_network_cfg,   # run through TensorRT, see engines.py
            _decoder_cfg,
            wav2feat_cfg,
            self.default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root)

        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.wav2feat = Wav2Feat(**wav2feat_cfg)
        self.thread_list = []

    def setup(self, rgb, warp, decode, on_frame, crop_scale=2.3, sampling_timesteps=10):
        """Register ``rgb`` (a still portrait) and start the workers.

        ``warp(f_s, x_s, x_d)`` and ``decode(feature)`` return CUDA tensors;
        ``on_frame(rgb_uint8)`` receives every frame, in order, from a worker
        thread. Each sampling of the motion model yields one chunk's frames.
        """
        overlap_v2 = self.audio2motion.lmdm.seq_frames - CHUNK_FRAMES
        kwargs = dict(self.default_kwargs)
        self.emo = kwargs.get("emo", 4)
        self.on_frame = on_frame
        self._warp = warp
        self._decode = decode

        self.source_info = self.avatar_registrar(
            rgb,
            crop_scale=crop_scale,
            crop_vx_ratio=kwargs.get("crop_vx_ratio", 0),
            crop_vy_ratio=kwargs.get("crop_vy_ratio", -0.125),
            crop_flag_do_rot=kwargs.get("crop_flag_do_rot", True),
        )
        self.condition_handler.setup(self.source_info, self.emo)

        self.audio2motion.setup(
            self.condition_handler.x_s_info_0,
            overlap_v2=overlap_v2,
            fix_kp_cond=kwargs.get("fix_kp_cond", 0),
            fix_kp_cond_dim=kwargs.get("fix_kp_cond_dim", None),
            sampling_timesteps=sampling_timesteps,
            online_mode=True,
            v_min_max_for_clip=kwargs.get("v_min_max_for_clip", None),
            smo_k_d=kwargs.get("smo_k_d", 3),
        )
        self.audio2motion.lmdm.capture_graph(sampling_timesteps)

        self.motion_stitch.setup(
            N_d=-1,
            use_d_keys=kwargs.get("use_d_keys", None),
            relative_d=kwargs.get("relative_d", True),
            drive_eye=kwargs.get("drive_eye", None),
            delta_eye_arr=kwargs.get("delta_eye_arr", None),
            delta_eye_open_n=kwargs.get("delta_eye_open_n", 0),
            fade_out_keys=kwargs.get("fade_out_keys", ("exp",)),
            fade_type=kwargs.get("fade_type", ""),
            flag_stitching=kwargs.get("flag_stitching", True),
            is_image_flag=True,
            x_s_info=self.source_info["x_s_info_lst"][0],
            d0=None,
            ch_info=None,
            overall_ctrl_info=kwargs.get("overall_ctrl_info", dict()),
        )
        self.ctrl_info = kwargs.get("ctrl_info", dict())

        # The motion model reads the last seq_frames frames of audio; silence fills
        # the ones before the first chunk.
        self.audio_feat = self.wav2feat.wav2feat(np.zeros((overlap_v2 * 640,), dtype=np.float32))
        assert len(self.audio_feat) == overlap_v2, f"{len(self.audio_feat)}"
        self.cond_idx_start = 0 - len(self.audio_feat)

        self._f_s = torch.from_numpy(self.source_info["f_s_lst"][0]).cuda().half()
        # The registrar's models measured the portrait and are not needed again.
        del self.avatar_registrar
        torch.cuda.empty_cache()

        self.worker_exception = None
        self.stop_event = threading.Event()
        self.audio2motion_queue = queue.Queue(maxsize=100)
        self.motion_stitch_queue = queue.Queue(maxsize=100)
        self.warp_f3d_queue = queue.Queue(maxsize=100)
        self.decode_f3d_queue = queue.Queue(maxsize=100)

        self.thread_list = [
            threading.Thread(target=self._guard, args=(self._audio2motion_worker,), daemon=True, name="ditto-motion"),
            threading.Thread(target=self._guard, args=(self._motion_stitch_worker,), daemon=True, name="ditto-stitch"),
            threading.Thread(target=self._guard, args=(self._warp_f3d_worker,), daemon=True, name="ditto-warp"),
            threading.Thread(target=self._guard, args=(self._decode_f3d_worker,), daemon=True, name="ditto-decode"),
        ]
        for thread in self.thread_list:
            thread.start()

    def _guard(self, worker):
        try:
            worker()
        except Exception as e:
            traceback.print_exc()
            self.worker_exception = e
            self.stop_event.set()

    def _put(self, q, item):
        while not self.stop_event.is_set():
            try:
                q.put(item, timeout=1)
                return
            except queue.Full:
                continue

    def _get(self, q):
        while not self.stop_event.is_set():
            try:
                return q.get(timeout=1)
            except queue.Empty:
                continue
        return None

    def _get_ctrl_info(self, fid):
        # Taken, not read: the renderer adds one entry per silent frame for as
        # long as it runs.
        if isinstance(self.ctrl_info, dict):
            return self.ctrl_info.pop(fid, {})
        return {}

    def _decode_f3d_worker(self):
        while not self.stop_event.is_set():
            f_3d = self._get(self.decode_f3d_queue)
            if f_3d is None:
                break
            img = self._decode(f_3d)
            rgb = (img[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            self.on_frame(rgb)

    def _warp_f3d_worker(self):
        while not self.stop_event.is_set():
            item = self._get(self.warp_f3d_queue)
            if item is None:
                break
            x_s, x_d = item
            f_3d = self._warp(
                self._f_s,
                torch.from_numpy(x_s).cuda().half(),
                torch.from_numpy(x_d).cuda().half(),
            )
            self._put(self.decode_f3d_queue, f_3d)

    def _motion_stitch_worker(self):
        x_s_info = self.source_info["x_s_info_lst"][0]
        while not self.stop_event.is_set():
            item = self._get(self.motion_stitch_queue)
            if item is None:
                break
            x_d_info, ctrl_kwargs = item
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
            self._put(self.warp_f3d_queue, [x_s, x_d])

    def _audio2motion_worker(self):
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        aud_feat_dim = self.wav2feat.feat_dim
        item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

        res_kp_seq = None
        res_kp_seq_valid_start = None

        global_idx = 0   # frame idx, for template
        local_idx = 0    # for cur audio_feat
        gen_frame_idx = 0
        while not self.stop_event.is_set():
            item = self._get(self.audio2motion_queue)
            if item is None:
                break
            item_buffer = np.concatenate([item_buffer, item], 0)
            if item_buffer.shape[0] < valid_clip_len:
                # wait at least valid_clip_len new item
                continue
            self.audio_feat = np.concatenate([self.audio_feat, item_buffer], 0)
            item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

            while True:
                aud_feat = self.audio_feat[local_idx: local_idx + seq_frames]
                if len(aud_feat) < seq_frames:
                    # wait next chunk
                    break

                aud_cond = self.condition_handler(aud_feat, global_idx + self.cond_idx_start)[None]
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                if res_kp_seq_valid_start is None:
                    # first chunk
                    res_kp_seq_valid_start = res_kp_seq.shape[1] - self.audio2motion.fuse_length
                    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
                    self.motion_stitch.d0 = d0

                    local_idx += valid_clip_len
                    global_idx += valid_clip_len
                    continue

                valid_res_kp_seq = res_kp_seq[:, res_kp_seq_valid_start: res_kp_seq_valid_start + valid_clip_len]
                for x_d_info in self.audio2motion.cvt_fmt(valid_res_kp_seq):
                    self._put(self.motion_stitch_queue, [x_d_info, self._get_ctrl_info(gen_frame_idx)])
                    gen_frame_idx += 1

                res_kp_seq_valid_start += valid_clip_len
                local_idx += valid_clip_len
                global_idx += valid_clip_len

                L = res_kp_seq.shape[1]
                if L > seq_frames * 2:
                    cut_L = L - seq_frames * 2
                    res_kp_seq = res_kp_seq[:, cut_L:]
                    res_kp_seq_valid_start -= cut_L

                if local_idx >= len(self.audio_feat):
                    break

            L = len(self.audio_feat)
            if L > seq_frames * 2:
                cut_L = L - seq_frames * 2
                self.audio_feat = self.audio_feat[cut_L:]
                local_idx -= cut_L

    def run_chunk(self, audio_chunk):
        """Feed one chunk of CHUNK_SAMPLES samples at 16 kHz."""
        assert len(audio_chunk) == CHUNK_SAMPLES, len(audio_chunk)
        aud_feat = self.wav2feat(audio_chunk, chunksize=CHUNKSIZE)
        self._put(self.audio2motion_queue, aud_feat)

    def stop(self):
        self.stop_event.set()
        for thread in self.thread_list:
            thread.join(timeout=5)
