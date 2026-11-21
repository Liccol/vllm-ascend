# M3.1 BF16 LoRA 一致性修复方案

**Goal:** 沿用原四层真实 BF16 权重、独立非零 A/B 和固定 prompt，尝试通过模型8项及混合服务36项 token 全一致验收。

**Architecture:** 默认保持低秩 LoRA。先验证在 adapter 激活阶段构建合并目标层缓存、执行阶段按既有 token→slot 映射选择结果的可选模式；复用上游 set/reset 生命周期、TP 分片和已有 batch-invariant GEMM，不做逐 token CPU 同步。不把串行因果对照当作多 adapter 功能已完成。

**Tech Stack:** 固定配套 vLLM main、BF16、NPU、TP2/DP2/EP、typed additional config、pytest。

## Task 1：可回退实验

- 先用测试 worker extension 实现激活时缓存及真正混批的结果选择。
- 同一服务加载 A/B，收集12项串行参考及三轮36项交错请求，用设备侧 trace 确认真正 base/A/B 混批。
- 与原同拓扑 merged-A/B 比较8项32-token输出；TP1跨拓扑结果单列。
- 任何失败保留原始记录，继续检查 MoE、批次及累加路径。

## Task 2：生产实现（实验有效后）

- 在 `vllm_ascend/ascend_config.py` 增加 typed `lora_execution_mode`，默认 `low_rank`，可选 `merged`。现有配置只有使能、rank、slot等，无法表达执行路径和额外显存成本，因此需要新配置。
- 在独立 M3.1 LoRA 模块实现 QKV/indexer 和 O 投影的缓存 wrapper，仅在该模型及 merged 模式匹配；其他模型及默认路径不变。
- 复用上游 set/reset、QKV和O的分片；缓存空间在 profiling 前预留，reset使槽位失效，partial reload 从原始基座重建，避免旧 adapter 或重复合并残留。
- 首轮只支持 BF16、eager、非 fully-sharded、无 SP；明确拒绝未支持的组合，不静默降级。
- 文档说明每个活跃 slot 额外保存完整目标层，不能称作低秩 kernel 的性能优化；合并使用上游已完成 scaling 的 A/B。

## Task 3：有意义的回归

- 按生产模块归属测试类型匹配、混合 slot、base、TP分片、bias、partial reload/reset，检查 indexer 不受影响。
- 配置测试覆盖非法值和不兼容模式的拒绝行为。
- 重跑原 NPU 模型及服务矩阵，量化实际缓存字节；保留正常低秩路径的既有结果。
- 完成相关 lint、中文结果报告和复现步骤；推送个人开发分支与 ref-doc，保留全层、真实训练 adapter、图及性能的覆盖边界。

## 执行结果

- 原型通过后完成正式 wrapper、typed config 和 profiling 前缓存预留；正式代码独立重启两次均通过同拓扑模型8/8、混合服务36/36。
- LoRA 与配置目录301项 UT通过，包含启动时有效 SP 配置的校验顺序回归；每个 worker 两个 slot 的逻辑缓存0.83203125 GiB。
- 跨 TP1→TP2 仍为4/8，全层与训练 adapter 未验证；默认低秩模型仍为3/8。详细证据和限制见 [验收报告](../../tests/e2e/models/minimax_m31_lora/MERGED_MODE.md)。
