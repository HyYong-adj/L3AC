import os
import utils
import wandb  # ✅ 추가
from model.exp.acc_runtime import CONFIG_DIR, RS, ACC
from model.exp import Config, Model
from model.exp.mlogging import progress_bar
import torch
import glob
import pickle
from safetensors.torch import load_file

log = utils.log.get_logger()


def _wandb_safe_init(config_dict: dict) -> None:
    """
    Main process에서만 wandb를 init.
    RS.version이 resume 접두어를 가질 수도 있으니, run id/name을 안정적으로 만든다.
    """
    if not ACC.is_main_process:
        return

    # (선택) W&B가 필요 없을 때 끄고 싶으면 환경변수로 제어 가능
    # export WANDB_DISABLED=true
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        log.warning("WANDB_DISABLED is set. Skip wandb.init()")
        return

    # run name / id 정리: resume 접두어는 제거해서 같은 run으로 이어갈 수 있게
    # 예) RS.version == "resume_MYRUN"  -> run_id == "MYRUN"
    if RS.version.startswith("resume"):
        run_id = utils.remove_special_char(RS.version.removeprefix("resume"), mode="abc+n")
        resume_opt = "allow"
    else:
        run_id = utils.remove_special_char(RS.version, mode="abc+n")
        resume_opt = False

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "l3ac"),
        entity=os.environ.get("WANDB_ENTITY", "lets-jam"),  # team name
        name=run_id,                 # 사람이 보기 좋은 이름
        id=run_id,                   # resume을 위해 고정 id
        resume=resume_opt,           # resume_* 일 때만 allow
        config=config_dict,          # config 기록
        dir=str(RS.output_dir),      # 로그 저장 위치
        tags=[RS.config_path] if hasattr(RS, "config_path") else None,
    )

    # (선택) 실험 메타데이터를 summary에 넣고 싶다면
    wandb.summary["rs_version"] = RS.version
    wandb.summary["config_path"] = str(getattr(RS, "config_path", ""))


def init_model() -> Model:
    log.warning(">>> init_model: start")
    config = Config(config_file=CONFIG_DIR / f"{RS.config_path}.toml")
    log.warning(">>> init_model: config loaded")
    if ACC.is_main_process:
        utils.output.dictionary(config.model_dump(), out_fun=log.info)
        RS.tlog.hyper_parameters(config.model_dump())

    log.warning(">>> init_model: before Model(config)")
    # ✅ wandb init (main process only)
    _wandb_safe_init(config.model_dump())

    model = Model(config)
    log.warning(">>> init_model: after Model(config)")

    if RS.version.startswith("resume"):
        log.warning(">>> init_model: resume branch entered")
        resumed_version = RS.version.removeprefix("resume_")  # 단순히 prefix 제거만
        acc_cache_dir = [
            log_dir for log_dir in (RS.output_dir / "log").iterdir()
            if (resumed_version.upper() in log_dir.name.upper()) and ('resume' not in log_dir.name)
        ]
        log.warning(f">>> init_model: acc_cache_dir candidates = {acc_cache_dir}")
        assert len(acc_cache_dir) == 1, f"{acc_cache_dir} should have one directory"
        log.warning(">>> init_model: before ACC.load_state")
        ACC.load_state(acc_cache_dir[0] / 'state_cache')
        log.warning(">>> init_model: after ACC.load_state")

        if ACC.is_main_process:
            log.warning(f"Resumed from {acc_cache_dir[0] / 'state_cache'}")

    if ACC.is_main_process:
        log.warning(">>> init_model: before get_model_info")
        from l3ac import get_model_info
        codec_info = get_model_info(
            ACC.unwrap_model(model.network),
            eval_flops_seconds=10,
            sample_rate=model.mc.sample_rate
        )
        log.warning(">>> init_model: after get_model_info")
        utils.output.dictionary(codec_info, out_fun=log.info)

        # ✅ codec info도 wandb에 기록 (config or summary로)
        if wandb.run is not None:
            wandb.summary.update({f"codec/{k}": v for k, v in codec_info.items()})
    log.warning(">>> init_model: end")
    return model

def load_hf_checkpoint(model, ckpt_dir: str):
    """HF 체크포인트에서 모델/옵티마이저/스케줄러 로드 (생성기 + 판별기)"""
    net = ACC.unwrap_model(model.network)
    opt = ACC.unwrap_optimizer(model.optimizer)
    sched = model.scheduler.scheduler  # 이미 unwrap 되어 있음
    
    dis_net = ACC.unwrap_model(model.dis_nn)
    dis_opt = ACC.unwrap_optimizer(model.dis_optimizer)
    dis_sched = model.dis_scheduler.scheduler

    # 1) 모델 가중치 (여러 shard 병합)
    state = {}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "model*.safetensors"))):
        state.update(load_file(path, device="cpu"))
    missing, unexpected = net.load_state_dict(state, strict=False)
    log.warning(f"[Generator] loaded weights; missing={len(missing)}, unexpected={len(unexpected)}")

    # 2) 생성기 옵티마이저/스케줄러
    opt_path = os.path.join(ckpt_dir, "optimizer.bin")
    if os.path.exists(opt_path):
        opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
        log.warning(f"[Generator] loaded optimizer from {opt_path}")
    sched_path = os.path.join(ckpt_dir, "scheduler.bin")
    if os.path.exists(sched_path):
        sched.load_state_dict(torch.load(sched_path, map_location="cpu"))
        log.warning(f"[Generator] loaded scheduler from {sched_path}")

    # 3) 판별기 옵티마이저/스케줄러
    dis_opt_path = os.path.join(ckpt_dir, "optimizer_1.bin")
    if os.path.exists(dis_opt_path):
        dis_opt.load_state_dict(torch.load(dis_opt_path, map_location="cpu"))
        log.warning(f"[Discriminator] loaded optimizer from {dis_opt_path}")
    dis_sched_path = os.path.join(ckpt_dir, "scheduler_1.bin")
    if os.path.exists(dis_sched_path):
        dis_sched.load_state_dict(torch.load(dis_sched_path, map_location="cpu"))
        log.warning(f"[Discriminator] loaded scheduler from {dis_sched_path}")

    # 4) RNG 상태 (optional)
    rng_path = os.path.join(ckpt_dir, "random_states_0.pkl")
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng = pickle.load(f)
        if "cpu_rng_state" in rng:
            torch.set_rng_state(rng["cpu_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in rng:
            torch.cuda.set_rng_state_all(rng["cuda_rng_state"])
        log.warning(f"[RNG] loaded random states from {rng_path}")

def train():
    model = init_model()
    
    # HF 체크포인트 로드 (환경변수나 파일 존재로 조건부 로드)
    # export HF_CKPT_DIR=path/to/hf_checkpoint
    #hf_ckpt_dir = os.environ.get("HF_CKPT_DIR", "hf_ckpt")
    #if os.path.exists(hf_ckpt_dir):
    #    log.warning(f">>> Loading HF checkpoint from {hf_ckpt_dir}")
    #    load_hf_checkpoint(model, hf_ckpt_dir)
    #else:
    #    log.warning(f">>> HF checkpoint dir not found at {hf_ckpt_dir}. Starting from scratch.")
    # ✅ ONLY_EVAL=1 이면 eval만 1번 돌리고 종료
    if os.environ.get("ONLY_EVAL", "").lower() in {"1", "true", "yes"}:
        metric_results = model.evaluate(model.eval_loader, "evaluating")
        extra_eval_results = {}
        for name, loader in model.eval_loaders.items():
            if name == "evaluating":
                continue
            extra_eval_results[name] = model.evaluate(loader, name)
        if ACC.is_main_process:
            log.info(f"[ONLY_EVAL] score: {metric_results}")
            if extra_eval_results:
                log.info(f"[ONLY_EVAL] extra eval scores: {extra_eval_results}")
            if wandb.run is not None:
                if isinstance(metric_results, dict):
                    log_dict = {f"eval/{k}": v for k, v in metric_results.items()}
                else:
                    log_dict = {"eval/score": metric_results}
                for name, results in extra_eval_results.items():
                    if isinstance(results, dict):
                        log_dict.update({f"{name}/{k}": v for k, v in results.items()})
                log.info(f"Logging ONLY_EVAL metrics to wandb: {log_dict}")
                wandb.log(log_dict)
                wandb.finish()
        ACC.wait_for_everyone()
        ACC.end_training()
        return
    start_epoch, total_epoch = model.estimate_progress()
    train_with_discriminator = 'network_gen_loss' in model.mc.loss_config['loss_weights']

    # best model tracking (maximize preferred eval metrics)
    best_score = None
    best_epoch = -1
    best_metric_label = None
    best_model_path = RS.output_dir / "best_model"

    def _select_eval_score(metric_results):
        """Pick a scalar score from nested metric_results.

        Priority (higher is better):
        1) MultiResSTFT.MultiResSTFT (reconstruction quality, lower is better)
        2) LogMelL1.LogMelL1 (reconstruction quality, lower is better)
        3) MERT.MERT (perceptual quality)
        4) CodebookUsage.usage_probs (token modeling friendliness)
        Fallback: first numeric value found.
        """
        priority = [
            ("MultiResSTFT", "MultiResSTFT", "min"),
            ("LogMelL1", "LogMelL1", "min"),
            ("MERT", "MERT", "max"),
            ("CodebookUsage", "usage_probs", "max"),
        ]

        for m_name, key, mode in priority:
            if isinstance(metric_results, dict) and m_name in metric_results:
                m_val = metric_results[m_name]
                if isinstance(m_val, dict) and key in m_val:
                    try:
                        return float(m_val[key]), mode, f"{m_name}.{key}"
                    except (TypeError, ValueError):
                        pass
                elif not isinstance(m_val, dict):
                    try:
                        return float(m_val), mode, m_name
                    except (TypeError, ValueError):
                        pass

        if isinstance(metric_results, dict) and len(metric_results) > 0:
            first_key = next(iter(metric_results))
            first_val = metric_results[first_key]
            if isinstance(first_val, dict) and len(first_val) > 0:
                sub_key = next(iter(first_val))
                try:
                    return float(first_val[sub_key]), "max", f"{first_key}.{sub_key}"
                except (TypeError, ValueError):
                    return None, "max", None
            try:
                return float(first_val), "max", str(first_key)
            except (TypeError, ValueError):
                return None, "max", None

        if isinstance(metric_results, (int, float)):
            return float(metric_results), "max", "metric"

        return None, "max", None

    for epoch in progress_bar(range(start_epoch, total_epoch), desc="Epoch"):
        log.info(f"Starting epoch {epoch}/{total_epoch}")

        if train_with_discriminator:
            model.train_epoch()
        else:
            model.train_epoch_without_discriminator()

        metric_results = model.evaluate(model.eval_loader, "evaluating")
        extra_eval_results = {}
        for name, loader in model.eval_loaders.items():
            if name == "evaluating":
                continue
            extra_eval_results[name] = model.evaluate(loader, name)
        ACC.save_state(RS.log_path / 'state_cache')

        if ACC.is_main_process:
            log.info(f"Eval epoch({epoch}) score: {metric_results}")
            if extra_eval_results:
                log.info(f"Eval epoch({epoch}) extra scores: {extra_eval_results}")

            current_score, mode, score_label = _select_eval_score(metric_results)

            if current_score is not None:
                # First valid eval score should always initialize the best model.
                if best_score is None:
                    improved = True
                else:
                    improved = (current_score > best_score) if mode == "max" else (current_score < best_score)
                if improved:
                    best_score = current_score
                    best_epoch = epoch
                    best_metric_label = score_label
                    ACC.unwrap_model(model.network).save_model(best_model_path)
                    log.info(f"Best model updated at epoch {epoch} with {score_label} = {best_score}")
                    if wandb.run is not None:
                        wandb.summary["best_score"] = best_score
                        wandb.summary["best_epoch"] = best_epoch
                        wandb.summary["best_metric"] = best_metric_label
                        try:
                            artifact = wandb.Artifact(
                                name="best-model",
                                type="model",
                                metadata={
                                    "epoch": epoch,
                                    "score": best_score,
                                    "metric": best_metric_label,
                                },
                            )
                            artifact.add_dir(str(best_model_path))
                            wandb.log_artifact(artifact)
                        except Exception as e:  # artifact 업로드 실패는 학습을 막지 않게 함
                            log.warning(f"wandb artifact log failed: {e}")
            else:
                log.warning("Unable to extract a numeric eval score; skipping best model update.")

            # ✅ wandb log (step 자동 관리, main process에서만)
            if wandb.run is not None:
                log.info(f">>> Wandb logging at epoch {epoch}")
                if isinstance(metric_results, dict):
                    log_dict = {f"eval/{k}": v for k, v in metric_results.items()}
                else:
                    log_dict = {"eval/score": metric_results}
                for name, results in extra_eval_results.items():
                    if isinstance(results, dict):
                        log_dict.update({f"{name}/{k}": v for k, v in results.items()})
                log_dict["epoch"] = epoch
                log.info(f">>> Logging metrics: {log_dict}")
                wandb.log(log_dict)

        ACC.wait_for_everyone()

    if ACC.is_main_process:
        if best_epoch >= 0 and best_model_path.exists():
            import shutil
            log.info(f"Copying best model (epoch {best_epoch}, score {best_score}) to {RS.output_path}")
            if best_model_path.is_dir():
                shutil.copytree(best_model_path, RS.output_path, dirs_exist_ok=True)
            else:
                shutil.copy2(best_model_path, RS.output_path)
        else:
            ACC.unwrap_model(model.network).save_model(RS.output_path)
        log.info("Finished training.")

        # ✅ 마무리
        if wandb.run is not None:
            wandb.finish()

    ACC.end_training()


if __name__ == '__main__':
    train()
