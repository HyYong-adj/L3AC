import torch
import torch.nn.functional as F
import xtract

import utils
from xtract.nn import t2n
import tools.audio.extend

from . import base
from ..loss.audio import MultiStft


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


class MultiResSTFT(base.ScoreMetric):
    desired_sample_rate = 0

    def __init__(self):
        super().__init__()
        self.loss_fn = MultiStft()

    @utils.args.AutoSigner()
    def update(self, generated_audio, audio):
        loss = self.loss_fn(generated_audio, audio)
        self.scores.append(float(loss.detach().cpu()))


class LogMelL1(base.ScoreMetric):
    desired_sample_rate = 0

    def __init__(self, input_sample_rate: int, n_fft: int = 1024, n_mels: int = 128, hop_length: int | None = None,
                 eps: float = xtract.nn.EPS):
        super().__init__()
        from torchaudio.transforms import MelSpectrogram

        self.eps = eps
        self.spec_func = MelSpectrogram(
            sample_rate=input_sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length if hop_length else n_fft // 4,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    @utils.args.AutoSigner()
    def update(self, generated_audio, audio):
        self.spec_func = self.spec_func.to(generated_audio.device)

        gen = self.spec_func(generated_audio).clamp_min(self.eps)
        ref = self.spec_func(audio).clamp_min(self.eps)

        gen_db = 10.0 * gen.log10()
        ref_db = 10.0 * ref.log10()

        loss = F.l1_loss(gen_db, ref_db)
        self.scores.append(float(loss.detach().cpu()))
