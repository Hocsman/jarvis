# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee),
# core/aux_models/modules/hubert_stream.py: the ONNX model on the CPU, on four
# threads that do not spin, under the name the audio encoder uses.
import onnxruntime


class HubertStreaming:
    def __init__(self, model_path, device="cuda", **kwargs):
        # On the CPU, which spares the GPU for the voice and Whisper. It runs every
        # 200 ms next to them, so it takes a few cores and does not spin on them.
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = onnxruntime.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    def forward_chunk(self, input_values):
        encoding_out = self.session.run(
            None,
            {"input_values": input_values.reshape(1, -1)}
        )[0]
        return encoding_out

    def __call__(self, audio_chunk):
        return self.forward_chunk(audio_chunk)
