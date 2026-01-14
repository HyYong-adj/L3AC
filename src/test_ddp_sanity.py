import os
import torch
import torch.distributed as dist

from xtract.ddp import tensor_reducer  # 경로 맞춰

def maybe_init_ddp():
    # torchrun이면 보통 이 env들이 존재
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

@tensor_reducer("sum")
def make_tensor():
    # 이제 init 되어있으면 rank 반영됨
    r = dist.get_rank() if dist.is_initialized() else 0
    return torch.tensor([1.0 + r], device="cuda" if torch.cuda.is_available() else "cpu")

def main():
    maybe_init_ddp()
    t = make_tensor()

    if dist.is_initialized():
        print("rank", dist.get_rank(), "world", dist.get_world_size(), "t", t.item())
        dist.barrier()
        dist.destroy_process_group()
    else:
        print("single", "t", t.item())

if __name__ == "__main__":
    main()
