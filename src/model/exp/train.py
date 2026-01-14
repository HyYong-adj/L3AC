import os
import utils
import wandb  # ✅ 추가
from model.exp.acc_runtime import CONFIG_DIR, RS, ACC
from model.exp import Config, Model
from model.exp.mlogging import progress_bar
import torch
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
        resumed_version = utils.remove_special_char(RS.version.removeprefix("resume"), mode='abc+n')
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


def train():
    model = init_model()
        # ✅ ONLY_EVAL=1 이면 eval만 1번 돌리고 종료
    if os.environ.get("ONLY_EVAL", "").lower() in {"1", "true", "yes"}:
        metric_results = model.evaluate(model.eval_loader, "evaluating")
        if ACC.is_main_process:
            log.info(f"[ONLY_EVAL] score: {metric_results}")
            if wandb.run is not None and isinstance(metric_results, dict):
                wandb.log({f"eval/{k}": v for k, v in metric_results.items()}, step=0)
                wandb.finish()
        ACC.wait_for_everyone()
        ACC.end_training()
        return
    start_epoch, total_epoch = model.estimate_progress()
    train_with_discriminator = 'network_gen_loss' in model.mc.loss_config['loss_weights']

    for epoch in progress_bar(range(start_epoch, total_epoch), desc="Epoch"):
        if train_with_discriminator:
            model.train_epoch()
        else:
            model.train_epoch_without_discriminator()

        metric_results = model.evaluate(model.eval_loader, "evaluating")
        ACC.save_state(RS.log_path / 'state_cache')

        if ACC.is_main_process:
            log.info(f"Eval epoch({epoch}) score: {metric_results}")

            # ✅ wandb log (epoch 단위)
            if wandb.run is not None:
                # metric_results가 dict 가정. 아니면 dict로 바꿔서 넣기
                log_dict = {f"eval/{k}": v for k, v in metric_results.items()} if isinstance(metric_results, dict) else {
                    "eval/score": metric_results
                }
                log_dict["epoch"] = epoch
                wandb.log(log_dict, step=epoch)

        ACC.wait_for_everyone()

    if ACC.is_main_process:
        ACC.unwrap_model(model.network).save_model(RS.output_path)
        log.info("Finished training.")

        # ✅ 마무리
        if wandb.run is not None:
            wandb.finish()

    ACC.end_training()


if __name__ == '__main__':
    train()
