import torch
import xtract
from .base import BaseMetric
from .classify import Accuracy
from .audio import STOI, PESQ, MERT, MultiResSTFT, LogMelL1
from .vq import CodebookUsage
from ..network import Network


def metric_builder(
    name: str,
    network: Network,
    sample_rate: int,
    cpu_num: int,
    cuda_device: int,
    mert_model_id: str | None = None,
    mert_use_layer_weights: bool = True,
    mert_layer_subset: tuple[int, ...] | None = (-6, -5, -4, -3, -2, -1),
    mert_device: str | None = "auto",
    mert_amp_dtype: str | None = None,
    log_mel_n_fft: int = 1024,
    log_mel_n_mels: int = 128,
    log_mel_hop_length: int | None = None,
    log_mel_eps: float = xtract.nn.EPS,
) -> BaseMetric:
    match name.lower():
        case 'accuracy':
            metric = Accuracy()
        case 'stoi':
            metric = STOI(input_sample_rate=sample_rate)
        case 'pesq':
            metric = PESQ(input_sample_rate=sample_rate, cpu_num=cpu_num)
        case 'mert':
            amp_dtype = None
            if isinstance(mert_amp_dtype, str) and mert_amp_dtype.strip():
                amp_dtype = getattr(torch, mert_amp_dtype, None)
            metric = MERT(
                input_sample_rate=sample_rate,
                model_id_or_path=mert_model_id or "m-a-p/MERT-v1-95M",
                use_layer_weights=mert_use_layer_weights,
                layer_subset=mert_layer_subset,
                device=mert_device,
                amp_dtype=amp_dtype,
            )
        case 'multi_res_stft':
            metric = MultiResSTFT()
        case 'log_mel_l1':
            metric = LogMelL1(
                input_sample_rate=sample_rate,
                n_fft=log_mel_n_fft,
                n_mels=log_mel_n_mels,
                hop_length=log_mel_hop_length,
                eps=log_mel_eps,
            )
        case 'codebook_usage':
            metric = CodebookUsage(network.quantizer.vq.codebook_size, cuda_device=cuda_device)
        case _:
            raise NotImplementedError(f"{name} not implemented")

    return metric


class Metrics:
    def __init__(self, network: Network, metric_names: list[str], **metric_config):
        self.metrics = [metric_builder(metric_name, network, **metric_config) for metric_name in metric_names]

    def add_metric(self, metric: BaseMetric):
        self.metrics.append(metric)

    def __getitem__(self, metric_name) -> BaseMetric:
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric
        raise KeyError(f"{metric_name} not found")

    def reset(self):
        for scorer in self.metrics:
            scorer.reset()

    def update(self, nn_output, ref_input, ):
        for metric in self.metrics:
            metric.update(nn_output, ref_input)

    def compute_internal_results(self) -> dict:
        return {metric.name: metric.compute_internal_results() for metric in self.metrics}

    def get_results(self, internal_results: dict) -> dict:
        metrics_results = {metric_name: self[metric_name].get_results(res)
                           for metric_name, res in internal_results.items()}
        return metrics_results

    def log_results(self, tlog: xtract.tensor_log.Writer | None, namespace: str) -> dict:
        all_results = {}
        for metric in self.metrics:
            internal_results = metric.compute_internal_results()
            if tlog is not None:
                results = metric.get_results(internal_results)
                all_results[metric.name] = metric.log_results(
                    results, tlog=tlog, namespace=f"{namespace}/{metric.name}")
        return all_results
