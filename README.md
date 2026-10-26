# llm-lab

东南大学 2029 届 · 大模型方向自学基地
从零到能部署：**手写 → 微调 → 部署 → Agent**。

---

## 目录结构

```
llm-lab/
├── scripts/
│   ├── gpu_bench.py      显卡基准测试（先跑这个）
│   └── hello_llm.py      加载 Qwen3-0.6B 生成第一段文字
├── notebooks/            实验与笔记（Jupyter）
├── notes/                每日复盘 md
├── data/                 数据集（小文件，大文件放 D:\dev\data）
└── output/               模型权重、日志、图表
```

---

## 环境（已配好）

```powershell
conda activate llm
python scripts\gpu_bench.py
```

详细说明见 `D:\dev\ENVIRONMENT.md`。

---

## 12 周路线（当前进度）

- [ ] **W1** 环境 + PyTorch 基础：跑通 `gpu_bench.py`、`hello_llm.py`，写出第一个训练循环
- [ ] **W2** 手写 attention（single-head），理解 QKV 维度
- [ ] **W3** 手写 multi-head + 因果掩码 + RoPE；与 PyTorch 官方实现数值对齐
- [ ] **W4** KV cache 计算 + 生成循环（greedy / top-k / temperature）
- [ ] **W5** 手写 BPE tokenizer
- [ ] **W6** 读 nanoGPT，跑通小模型训练
- [ ] **W7** 训练循环进阶：混合精度 + 梯度累积 + 显存计算
- [ ] **W8** LoRA / QLoRA 微调 1.7B 模型
- [ ] **W9** 构建评测集（50–100 条）+ 微调前后对比报告
- [ ] **W10** vLLM 部署（云端 GPU）+ 压测 QPS / P95 / 显存
- [ ] **W11** 量化（AWQ/GPTQ）+ batch 调优，出对比表
- [ ] **W12** RAG + 工具调用最小闭环

---

## 每周固定动作

1. 写代码 ≥ 5 小时
2. 输出一篇 300 字笔记（`notes/week-XX.md`）
3. 读一段源码（W6 后开始读 vLLM / llama.cpp）
4. 周日复盘 30 分钟，把坑记进 `notes/pitfalls.md`

---

## 三条纪律

- **不看完整课程不动手**：每看 20 分钟必须写代码。
- **没有评测的优化不算优化**：改 prompt / 改参数都要有对照。
- **每个项目都要有数字**：显存、延迟、吞吐、准确率，至少一个。
