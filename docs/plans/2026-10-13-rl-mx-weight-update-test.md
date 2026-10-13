# RL MX Weight Update Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 增加一个单 NPU 测试，验证 RL 场景下 MXFP8/MXFP4 权重经过两轮热更新后，与相同 payload 的全新加载结果一致。

**Architecture:** 测试使用真实 `torch_npu.npu_dynamic_mx_quant`、vllm-ascend MX scheme 的 restore/process 方法和真实 NPU kernel。测试不启动完整 RL trainer；以相同低精度 payload 的 fresh-load layer 作为 oracle，并分别覆盖 Dense、MoE 和混合 BF16 层。

**Tech Stack:** pytest、PyTorch、torch_npu、vLLM、vllm-ascend。

---

### Task 1: 建立可复用的 MX payload 和 reload 测试 helper

**Files:**
- Create: `tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`

**Step 1: 写出 payload 生成、layer 构造和 reload helper**

- 使用 `npu_dynamic_mx_quant(axis=-1)` 生成权重与 scale。
- MXFP4 权重在 loader 边界转换成 `uint8`。
- reload helper 严格执行 restore、分 bucket copy、process。

**Step 2: 静态验证文件可解析**

Run: `python -m py_compile tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`

Expected: PASS。

### Task 2: 实现 Dense MXFP8/MXFP4 两轮热更新测试

**Files:**
- Modify: `tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`

**Step 1: 构造 W0、W1、W2 和固定输入**

- 维度使用 group size 32 的整数倍。
- W1/W2 使用确定性偏移，确保 weight 和 scale 均发生变化。

**Step 2: 比较 hot reload 与 fresh load**

- 比较推理布局权重和 scale。
- 比较实际 MX matmul 输出。
- 验证 MXFP8 持久 buffer 的地址不变。
- 验证初始输出与更新后输出不同。

**Step 3: 在 NPU 环境执行 Dense 用例**

Run: `pytest -sv tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py -k dense`

Expected: 两种格式均 PASS。

### Task 3: 实现 MoE MXFP8/MXFP4 reload 测试

**Files:**
- Modify: `tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`

**Step 1: 构造小型双 expert 权重**

- 分别生成 `w13_weight`、`w2_weight` 及其 scale。
- 每轮 restore 后分开加载 weight bucket 和 scale bucket。

**Step 2: 比较 hot reload 与 fresh load**

- 逐张量比较推理布局。
- 执行真实 grouped matmul MLP 路径；若当前硬件不支持最小 shape，则明确跳过输出部分并保留张量 oracle。
- 验证第二轮 reload。

**Step 3: 在 NPU 环境执行 MoE 用例**

Run: `pytest -sv tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py -k moe`

Expected: 两种格式均 PASS，或由明确的硬件 capability 条件跳过不支持的 kernel。

### Task 4: 实现混合 MX/BF16 更新检查

**Files:**
- Modify: `tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`

**Step 1: 在 MX Dense 测试模型中加入 BF16 residual projection**

- BF16 层不经过量化。
- 两轮更新时与 MX payload 一起更新。

**Step 2: 验证 BF16 精确更新和组合输出**

Run: `pytest -sv tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py -k mixed`

Expected: PASS。

### Task 5: 完成代码质量检查

**Files:**
- Test: `tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`
- Test: `docs/source/developer_guide/Design_Documents/rl_mx_weight_update_test.md`

**Step 1: 运行格式和静态检查**

Run: `ruff check tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`

Run: `ruff format --check tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py`

Run: `git diff --check`

Expected: 全部 PASS。

**Step 2: 汇报硬件验证边界**

如果当前主机没有 NPU，明确记录未运行的 NPU 命令，不把静态检查表述成运行通过。
