"""显卡基准测试：确认 PyTorch + CUDA 工作正常，并给出本机性能基线。

运行：  python scripts/gpu_bench.py
"""
import time

import _console  # noqa: F401  修复 Windows 控制台编码

import torch


def human(n):
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:,.1f}{unit}"
        n /= 1000
    return f"{n:,.1f}P"


def main():
    print("=" * 58)
    print(" 显卡基准测试")
    print("=" * 58)
    if not torch.cuda.is_available():
        print("CUDA 不可用！检查是否装了 CPU 版 torch。")
        return

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU        : {name}")
    print(f"算力       : sm_{props.major}{props.minor}")
    print(f"显存       : {total / 1024**3:.2f} GB（空闲 {free / 1024**3:.2f} GB）")
    print(f"PyTorch    : {torch.__version__}  CUDA {torch.version.cuda}")

    # 矩阵乘法：测 fp16 算力
    print("\n[1] fp16 矩阵乘法吞吐")
    for n in (1024, 2048, 4096):
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        tflops = 2 * n**3 / dt / 1e12
        print(f"    {n:>5}²  {dt*1000:7.2f} ms/次  {tflops:7.2f} TFLOPS")

    # 显存带宽
    print("\n[2] 显存带宽（大张量拷贝）")
    x = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(f"    拷贝 256 MB 用时 {dt*1000:.2f} ms -> 约 {2*256/1024/dt:.0f} GB/s（读写合计）")

    # 反向传播
    print("\n[3] 小规模训练一步（含反向）")
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 3072), torch.nn.GELU(), torch.nn.Linear(3072, 768)
    ).cuda().half()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    xb = torch.randn(8, 128, 768, device="cuda", dtype=torch.float16)
    for _ in range(2):
        loss = model(xb).float().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = model(xb).float().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"    batch=8, seq=128, hidden=768  {dt*1000:.1f} ms/step")

    print("\n" + "=" * 58)
    print(" 结论：环境正常。把上面的数字记下来，以后换环境可以对比。")
    print("=" * 58)


if __name__ == "__main__":
    main()

