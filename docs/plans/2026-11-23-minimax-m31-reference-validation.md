# M3.1 LoRA 参考协议用例改造

**Goal:** 参考 MindSpeed-Agent `origin/feat/lora-adaptation` 的固定提交 `a86e85307e87888ad6ab0ac41e4b5925ca45b65c`，增加标准低秩独立参考、零对照、逐层观测及加载生命周期验证，并在真实 NPU 执行。

**Architecture:** 复用原四层 BF16 基座和独立非零 A/B，不改训练目标或原严格用例。新协议单独保存结果，优先单设备 eager、单 adapter，比较原始磁盘因子计算的 FP64 增量，适配五段 QKV 的前三段边界和 indexer 尾部。独立模型参考使用已安装 Transformers 的实际 M3 类，不能直接套用只支持 AutoModelForCausalLM 的默认参考入口。

**Tech Stack:** 固定 vLLM c410543d9、当前 Ascend 个人分支、BF16、pytest、safetensors、NPU、Transformers 5.19.0。

## Task 1：参考审计与冻结

- 保存参考提交、协议差异、实际模型/adapter/源码身份、容差和阶段范围。
- 运行参考仓库 CPU 测试，区分采集器自测和 M3 实机结果。
- 新协议新增 base/zero/A/B、Q/K/V 隔离、至少一次 prefill 和两步 decode、load/remove/load、非法 rank 和错误增量灵敏度。
- 旧 merged8项、跨TP和36项混批不删除、不改名为本协议通过；本轮验收不从单设备外推到TP/DP/EP。

## Task 2：实现和回归

- 在 `tests/e2e/models/minimax_m31_lora/` 增加独立参考验收器和测试用 worker 观测。
- 原始磁盘 A/B 作为参考；已加载 A/B 仅作为待验证对象。记录缩放、rank padding、实际 wrapper、token slot和基座保持不变。
- 分别记录低秩原始 delta、实际加到BF16输出的delta和完整输出；避免把基座舍入误差误报为低秩错误。
- 固定逐元素容差，不根据结果事后放宽。信号不足标为 INVALID_PROBE，缺少HF模型参考或采集能力标为 PARTIAL/blocked，不伪造通过。
- 首轮 BF16 数值合同在实测前冻结为 `atol=rtol=1/128`（一个 BF16 epsilon），不沿用参考仓仅用于微型 FP16 Llama 的0.003容差；另将实际加到 BF16 基座输出的增量与原始分支增量分别记录。
- 对检查器增加能发现漏增量、重复缩放、错slice和非有限数值的必要回归。

## Task 3：实机与交付

- 使用真实四层 BF16 权重和原A/B执行；zero及隔离/非法产物存独立新目录，原资产不变。
- 运行逐层、单adapter生命周期和可用的teacher-forcing模型参考；完整记录失败和恢复。
- 保存脱敏报告和证据到个人分支及ref-doc。仅报告实际完成阶段，不将参考工具的CPU通过视为M3 NPU验收。
