import torch
import torch.nn.functional as F

import utils
from xtract.nn import t2n
import tools.audio.extend

from . import base


class T2NAutoSigner(utils.args.AutoSigner):
    @staticmethod
    def format_args(arg_name: str, arg_value):
        if isinstance(arg_value, torch.Tensor):
            arg_value = t2n(arg_value)
        return arg_value


class PESQ(base.ScoreMetric):
    desired_sample_rate = 0

    def __init__(self, input_sample_rate, cpu_num=0):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.cpu_num = cpu_num

    @T2NAutoSigner()
    def update(self, generated_audio, audio):
        new_scores = tools.audio.extend.pesq_batch(generated_audio, sample_rate=self.input_sample_rate,
                                                   ref_audio=audio, cpu_num=self.cpu_num)
        self.scores += [s for s in new_scores if isinstance(s, float)]


class STOI(base.ScoreMetric):
    desired_sample_rate = 0

    def __init__(self, input_sample_rate):
        super().__init__()
        self.input_sample_rate = input_sample_rate

    @T2NAutoSigner()
    def update(self, generated_audio, audio):
        new_scores = tools.audio.extend.stoi_batch(generated_audio, sample_rate=self.input_sample_rate,
                                                   ref_audio=audio, )
        self.scores += new_scores


class MERT(base.ScoreMetric):
    desired_sample_rate = 0

    def __init__(
        self,
        input_sample_rate: int,
        model_id_or_path: str | None = "m-a-p/MERT-v1-95M",
        use_layer_weights: bool = True,
        layer_subset: tuple[int, ...] | None = (-6, -5, -4, -3, -2, -1),
        device: str | torch.device | None = "auto",
        amp_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        from tools.loss.mert import MertPerceptualLoss

        self.mert = MertPerceptualLoss(
            model_id_or_path=model_id_or_path or "m-a-p/MERT-v1-95M",
            input_sample_rate=input_sample_rate,
            use_layer_weights=use_layer_weights,
            layer_subset=layer_subset,
            device=device,
            amp_dtype=amp_dtype,
        )

    @utils.args.AutoSigner()
    def update(self, generated_audio, audio):
        pred_feat = self.mert.get_feature(generated_audio)
        tgt_feat = self.mert.get_feature(audio)
        sim = F.cosine_similarity(pred_feat, tgt_feat, dim=-1).mean()
        self.scores.append(float(sim.detach().cpu()))
