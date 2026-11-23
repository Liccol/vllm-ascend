# 参考 MindSpeed-Agent 的 LoRA 用例改造与实测

2026-11-23：新增独立数值参考、隔离与加载生命周期用例，完成真实 NPU 测试；**结果为 PARTIAL / failed，不能宣布标准低秩适配验收通过**。本轮只改测试代码，生产实现仍为 `a2f6c92082d951f10d3b94eb253e4437d4d9dbeb`。

## 参考与改造范围

提供的 `MindSpeed-Agent_9400` 检出在 master `3c6e565`，新增 LoRA 内容实际位于 `origin/feat/lora-adaptation`。使用独立只读 worktree 阅读固定提交 `a86e85307e87888ad6ab0ac41e4b5925ca45b65c`，没有切换或修改原工作目录。

主要参考该提交的 `skills/inference-model-skills/model-adaptation/vllm-lora-adaptation/`，以及 `plugins/inference-model-optimize/engine/lora/`。其默认设备采集器面向普通 causal LM；M3 使用多模态配置、五段融合 QKV 和 indexer 尾部，因而复用协议、另写 M3 观测器。参考仓的 CPU 测试实测为 **74 passed、15 subtests passed**，这不代表其默认设备用例已经在 M3 NPU 上通过。

| 参考要求 | 本次实现 |
| --- | --- |
| 原始磁盘因子独立参考 | CPU FP64 `scale*(X@A.T)@B.T`；不从加载后的 buffer 构造参考 |
| 公式交叉检查 | 每个 slice 另算 `X@(scale*B@A).T`；取前4行输入、前64列输出，避免构造完整大矩阵 |
| 信号与误差合同 | 非零 A/B、zero-B；每个活动 slice 信号大于冻结误差和重复噪声裕量的10倍 |
| 加载布局 | 实际 A/B、scale、rank padding、token slot、基座 checkpoint 与存储稳定性 |
| 融合隔离 | Q-only/K-only/V-only；其他 attention slice 和 indexer 尾部保持不变 |
| 层级观测 | 每次请求、每个物理目标采集一次 prefill 和两步 decode 的真实输入、基座结果、完整结果、原始 delta |
| 独立模型 | 已安装 Transformers 的 `MiniMaxM3SparseForConditionalGeneration`，CPU FP32、eager、原始因子未合并 |
| 生命周期 | 真正执行 load → remove → load；非法 rank 被实际 loader 拒绝，随后检查 base/zero 恢复 |
| 灵敏度 | 对已采集增量进行漏增量、双倍缩放和错列移位，验证检查器拒绝；这些是测量负例，不是设备故障注入 |
| 判定 | 证据缺失不能 VERIFIED；汇总 CLI 对通过/失败/待补证据分别返回0/1/2 |

独立协议采用 **TP1/DP1、单 adapter、eager**。保留原模型正确性、TP/DP/EP 混批与跨 TP 用例，结果不互相替代。

## 固定资产与合同

- vLLM：`c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，没有更新到浮动 main。
- Ascend 生产代码：`a2f6c92082d951f10d3b94eb253e4437d4d9dbeb`；测试脚本额外上传，执行时源码摘要保存于证据。
- Python 3.12.16、Torch `2.10.0+cpu`、torch-npu `2.10.0.post7`、Transformers `5.19.0`、safetensors `0.8.0`。Torch 版本后缀如实记录；设备执行通过 NPU 插件完成。未升级远端依赖。
- A3，真实 BF16 checkpoint 的前4层，保持 hidden=6144、Q heads=64、KV heads=4、head dim=128、128 routed experts、top-k=4、词表200064；包含3个 dense MLP 层和1个 routed MoE 层。
- 仅 attention Q/K/V/O adapter；r=8、alpha=16；原 A/B 为两套非零合成权重，seed=20261121/20261122、std=0.1。不是训练 adapter。原始资产没有替换或缩小幅度。
- TP1 融合 QKV 物理输出9856列：8192 Q +512 K +512 V +512 index-Q +128 index-K。后三种隔离样本只启用相应 Q/K/V，O 的 B 为零。
- 数值容差在首次实测前冻结为 `atol=rtol=1/128`。参考仓微型 FP16 Llama 示例的0.003未直接套用到 BF16 M3；本合同也未根据失败事后放宽。
- 4种 prompt、每次32个生成 token；temperature=0、seed=0、ignore_eos=True，记录原生 token ID、finish_reason、top-5及所选 token 的 logprob。

## 实测结果

成功执行 `run-004` 共24次请求：base/zero/A/B各4次、Q/K/V隔离各1次、load/remove/load共3次、非法 rank 后 base/zero 恢复各1次。每次均覆盖8个物理目标 ×3个阶段，共576条记录。

| 检查 | 结果 |
| --- | --- |
| 请求、目标和 prefill/两步 decode 覆盖 | 24/24请求、576/576记录完整 |
| A/B buffer、缩放/padding、token slot | 576/576通过 |
| 基座等于原 checkpoint | 8/8物理目标通过 |
| 基座内容与存储不变 | 8/8通过 |
| indexer 尾部无增量且输出不变 | 288/288通过 |
| 非活动 slice 精确为零且输出不变 | 636/636通过，包含 base/zero/隔离记录 |
| FP64 两公式抽样交叉检查 | 1152/1152通过 |
| A/B 有效信号 | 240条物理记录、480个逻辑 slice 通过 |
| 测量灵敏度 | 240条非零记录 ×3种变异均被拒绝 |
| zero 与 base | 4/4 prompt 的 token、finish、logprob精确相同 |
| base/A/B 可区分 | 三种两两组合均存在不同生成 token 的 prompt |
| load/remove/load、非法 rank 拒绝与恢复 | 全部通过；拒绝原因明确为rank9超过max rank8 |
| 原始 delta 对 FP64 参考 | QKV 288/288；O 179/288 |
| 完整结果对“实测基座结果 + 理想 delta”的 BF16 值 | QKV 288/288；O 175/288 |
| 完整结果对 FP64 `XW + delta` | QKV 252/288；O 172/288 |
| 所有层级门槛同时通过 | **424/576** |
| HF teacher-forcing greedy 与设备序列逐 token 相同 | base/zero 8/8；非零 A/B 3/8；合计11/16 |
| HF sampled logprob 数值门槛 | 2/16通过，仅 base-p1、zero-p1 |
| 汇总状态 | **PARTIAL / failed**，逐层数值与独立模型门槛未通过 |

HF 模型独立读取原 checkpoint，确认无缺失或多余权重，并逐个确认16个 Q/K/V/O 权重与 checkpoint 相同。它在设备生成的同一历史上做 teacher forcing，比较每步 greedy 与 logprob；不是另一次自由生成。基座自身有3/4 prompt 的 sampled logprob 超限，不能把全部模型级差异都归因于 LoRA。

## 分歧定位与边界

原始 delta 超限的109条全部来自非零 A/B 的 O 分支。当前 `max_loras=1` 在该硬件上选择单槽快速路径；O 的 Triton expand 将 FP32 shrink 转为 B 的 BF16 类型后再做第二次乘法。M3 的未打包三段 QKV 使用另一条算子路径。本轮没有修改这些生产算子。

对120条非零 O 记录使用同一真实输入和原始磁盘因子，额外在 CPU 模拟 `X·Aᵀ` 中间 BF16 舍入：相对 FP64 理想增量为11/120通过，相对 BF16 中间值参考为119/120通过。此证据支持中间精度是主要差异来源，仍有1条未完全解释；它是诊断对照，**不替代或放宽原门槛**。全输出还包含基座 GEMM 的数值误差及加法舍入，不能只看 `Y-Y_base` 判断低秩分支。

首次执行还保留了三次工具兼容性失败：默认 fork 无法重新初始化 NPU、callable worker RPC 被安全序列化限制、generic RPC 未保留 LoRARequest 类型。分别改用现有 spawn 配置、具名 worker 扩展、原生 typed engine LoRA API 后完成24次请求；没有启用不安全 RPC 序列化。

本轮没有重跑旧多卡测试，也没有将其失败改成通过。上次标准低秩结果仍为：同拓扑 merged 对照3/8、TP2/DP2/EP实际混批隔离36/36、跨拓扑对比3/8（TP1/DP1对TP2/DP2/EP，也包含DP/EP变化）。详见 [原报告](STANDARD_LOW_RANK.md)。

没有覆盖全60层、真实训练 adapter、专家/视觉/embedding/LM-head LoRA、fully-sharded、图模式、FlashComm、MTP、长上下文容量及性能。新协议限制在4层、max-model-len1024、max-num-seqs1，不构成128k容量或吞吐结论。

## 复现

在已安装固定版本的环境执行，使用原 [资产准备](README.md#1-资产准备) 产物。以下路径为示例；原始 A/B 不会被覆盖，每次输出目录必须不存在。

```bash
cd /workspace
export VLLM_VERSION=0.33.0.dev0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=1
export ASCEND_RT_VISIBLE_DEVICES=0
scripts=/vllm-workspace/vllm-ascend/tests/e2e/models/minimax_m31_lora
weights=/workspace/lora-validation/weights-strong
result=/workspace/lora-validation/reference-new
python "$scripts/reference_acceptance.py" --weights "$weights" --output "$result"
```

设备阶段结束后，即使数值失败，继续独立运行 CPU 参考和最终汇总；不要用 `&&` 因上一阶段返回1而跳过后续证据。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python "$scripts/reference_model.py" \
  --weights "$weights" --output "$result"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python "$scripts/reference_replay.py" \
  --weights "$weights" --output "$result"
python "$scripts/reference_summary.py" --output "$result"
```

`reference_summary.py` 输出 `final-summary.json`，0=全部通过、1=已知失败、2=尚缺独立模型证据。`PARTIAL` 结合 `outcome=failed/pending` 区分失败与缺证据。原始 `.pt` 包含权重和真实观测，只在私有测试目录保留，公开证据仅包含标量结果、原生 token/logprob、身份与脚本摘要。

本轮原始运行器在分析后正常退出，最终判定由后补的汇总器读取已保存证据完成，返回1。交付版增加了阶段退出码和汇总完整性检查；没有重跑模型来替换原始失败。证据分别保存执行时源码与后补汇总/重放源码。

## 代码与检查

- `reference_acceptance.py`：生成独立派生 adapter、执行24次请求、实际生命周期、加载与层级分析。
- `reference_observer.py`：具名 worker RPC，采集真实输入/输出、raw delta及加载 buffer；不会替换模型结果。观测包含同步和 IO，仅供正确性测试。
- `reference_checks.py`：独立磁盘因子 FP64 参考与误差/信号/负例检查。
- `reference_model.py`：独立 HF teacher forcing。
- `reference_replay.py`：O 分支中间精度诊断。
- `reference_summary.py`：检查目标、阶段及请求覆盖，汇总并拒绝缺失证据假通过。
- `tests/ut/lora/test_reference_checks.py`：9项检查器回归；远端原生 `tests/ut/lora` 合计128 passed。

Python Ruff lint/format、codespell、Markdown 检查分别执行。完整 `format.sh ci` 仍受环境缺少 pre-commit 限制，不能声称完整 CI 通过。
