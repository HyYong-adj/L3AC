import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 🔽 여기 경로는 main.py에서 Config import하는 경로랑 동일해야 함
    from src.model.exp.configs import Config   # ❗ 이 줄은 프로젝트에 맞게 바꿔야 할 수도 있음

    config = Config.load(args.config)

    dl = config.train_data.get_dataloader(prefetch_size=0)
    t0 = time.time()
    batch = next(iter(dl))
    audio = batch["audio"]

    print(
        "loaded one batch in",
        round(time.time() - t0, 3),
        "sec",
        tuple(audio.shape),
        audio.dtype,
    )

if __name__ == "__main__":
    main()
