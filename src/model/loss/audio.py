import pathlib

from torch import nn
import torch.nn.functional as F

import utils.args
import xtract


class ElementWise(nn.Module):
    ignore_factor = 0.01

    def __init__(self, ignore_small_values=False, mse_or_l1='mse'):
        super().__init__()
        if ignore_small_values:
            self.mean_func = lambda d: d[d > (d.mean(dim=-1, keepdim=True) * self.ignore_factor)].mean(dim=-1)  # ! 0.01
        else:
            self.mean_func = lambda d: d.mean(dim=-1)
        if mse_or_l1 == 'mse':
            self.before_mean = lambda x: x ** 2
        elif mse_or_l1 == 'l1':
            self.before_mean = lambda x: x.abs()
        else:
            raise ValueError(f"mse_or_l1 must be 'mse' or 'l1', but got {mse_or_l1}")
        self.after_mean = lambda y: y

    @utils.args.AutoSigner()
    def forward(self, generated_audio, audio):
        all_loss = self.before_mean(generated_audio - audio)
        mean_loss = self.mean_func(all_loss)
        return self.after_mean(mean_loss).mean()


class StftLoss(nn.Module):
    def __init__(self, n_fft=400, hop_length=None, eps=xtract.nn.EPS):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = (self.n_fft / 2) ** 0.5
        self.hop_length = hop_length if hop_length else n_fft // 4
        self.eps = eps
        from torchaudio.transforms import Spectrogram
        self.spec_func = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=1)

    @utils.args.AutoSigner()
    def forward(self, generated_audio, audio):
        generated_audio_s = self.spec_func(generated_audio)
        audio_s = self.spec_func(audio)
        dd = F.l1_loss(generated_audio_s, audio_s)
        # log_dd = self.alpha * F.mse_loss(generated_audio_s.clamp(self.eps).log10(), audio_s.clamp(self.eps).log10())
        log_dd = F.l1_loss(generated_audio_s.clamp(self.eps).pow(2).log10(), audio_s.clamp(self.eps).pow(2).log10())
        return dd + log_dd


class MelLoss(StftLoss):
    def __init__(self, sample_rate, n_fft, n_mels=8, hop_length=None, eps=xtract.nn.EPS):
        super().__init__(n_fft=n_fft, hop_length=hop_length, eps=eps)
        self.n_mels = n_mels
        from torchaudio.transforms import MelSpectrogram
        self.spec_func = MelSpectrogram(sample_rate=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                                        n_mels=self.n_mels, power=1)


class MultiStft(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList([StftLoss(2 ** i) for i in range(5, 11)])  #music -> remove n_fft=32,64,128 

    @utils.args.AutoSigner()
    def forward(self, generated_audio, audio):
        loss = [loss_func(generated_audio=generated_audio, audio=audio) for loss_func in self.losses]
        return sum(loss) / len(loss)


class MultiMel(nn.Module):

    def __init__(self, sample_rate):
        super().__init__()
        self.losses = nn.ModuleList([MelLoss(sample_rate=sample_rate, n_fft=2 ** i) for i in range(5, 11)])

    @utils.args.AutoSigner()
    def forward(self, generated_audio, audio):
        loss = [loss_func(generated_audio=generated_audio, audio=audio) for loss_func in self.losses]
        return sum(loss) / len(loss) * 1.0  # ! *1.0


class DenseMel(MelLoss):
    def __init__(self, sample_rate):
        super().__init__(sample_rate, n_fft=512, n_mels=128, hop_length=10)


class PerceptualLoss(nn.Module):
    def __init__(self, sample_rate, weight_path='s2t/whisper/tiny.en.pt', backend='mert', mert_model_id: str | None = None, mert_use_layer_weights: bool = True):
        super().__init__()
        """Wrapper that selects which perceptual loss backend to use.

        backend: 'mert' or 'whisper'
        - If 'mert' the `mert_model_id` (HF model id or local path) is passed to the
          MERT loader; if not provided it falls back to `weight_path` for a local
          checkpoint when available. `mert_use_layer_weights` controls whether a
          learnable weighted-average across MERT layers is used.
        - If 'whisper' the `weight_path` is used as before.

        Default: 'mert' to use the new MERT-based perceptual loss while keeping
        Whisper-based loss available for fallback and compatibility.
        """
        if not pathlib.Path(weight_path).is_absolute():
            weight_path = utils.file.DATA_PATH / 'model' / weight_path

        if backend.lower() == 'whisper':
            from tools.loss.asr import PerceptualLoss as WhisperLoss
            self.loss_model = WhisperLoss(model_path=weight_path, sample_rate=sample_rate)
        else:
            try:
                from tools.loss.mert import MertPerceptualLoss as MertLoss
                # Treat empty string or 'None' from configs as disabling HF MERT
                if isinstance(mert_model_id, str) and mert_model_id.strip().lower() in ("", "none"):
                    model_id_or_path = "m-a-p/MERT-v1-95M"
                elif mert_model_id is None:
                    model_id_or_path = "m-a-p/MERT-v1-95M"
                else:
                    model_id_or_path = mert_model_id
                log = utils.log.get_logger()
                log.info(f"PerceptualLoss backend=mert, model_path={model_id_or_path}")
                self.loss_model = MertLoss(
                    model_id_or_path=model_id_or_path,
                    input_sample_rate=sample_rate,
                    use_layer_weights=mert_use_layer_weights,
                    layer_subset=(-6, -5, -4, -3, -2, -1),
                )
            except Exception as e:
                from tools.loss.asr import PerceptualLoss as WhisperLoss
                self.loss_model = WhisperLoss(model_path=weight_path, sample_rate=sample_rate)
                log = utils.log.get_logger()
                log.warning(f"MertPerceptualLoss import failed ({e}), falling back to Whisper-based perceptual loss")

    @utils.args.AutoSigner()
    def forward(self, generated_audio, audio):
        return self.loss_model(generated_audio, audio)


class ContrastiveLoss(nn.Module):
    @utils.args.AutoSigner()
    def forward(self, hidden_feature):
        return F.mse_loss(hidden_feature['quantized_feature'], hidden_feature['encoded_feature'].detach())
