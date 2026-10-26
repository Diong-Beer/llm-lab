"""第一个大模型 Demo：下载 Qwen3-0.6B 并生成文字。

运行：  python scripts/hello_llm.py
首次运行会从 hf-mirror 下载约 1.2 GB，请保证网络畅通。
"""
import time

import _console  # noqa: F401  修复 Windows 控制台编码

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"  # 6G 显存轻松跑；想换大一点的可以用 Qwen/Qwen3-1.7B


def main():
    print(f"加载 {MODEL_ID} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print(f"加载完成，用时 {time.time() - t0:.1f}s，"
          f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    prompt = "用三句话解释什么是卡尔曼滤波："
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    dt = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("\n" + "=" * 58)
    print(f"提问：{prompt}")
    print("-" * 58)
    print(answer)
    print("=" * 58)
    print(f"生成 {n_new} tokens，用时 {dt:.2f}s，速度 {n_new / dt:.1f} tokens/s")
    print(f"峰值显存 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()

