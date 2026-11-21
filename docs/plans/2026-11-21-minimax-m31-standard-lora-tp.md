# M3.1 标准低秩与跨 TP 定位计划

**Goal:** 保持标准低秩执行，定位原8项 merged 对照失败和 TP1/TP2 差异；保留原权重、prompt、32-token及严格判定，不把完整 merged GEMM 当成修复。

**Architecture:** 先用真实输入与独立算术参考区分权重舍入、低秩求值、BF16局部输出及TP归约的贡献。对仍可能修复的累加路径使用测试 worker extension 做因果对照；只将经过原矩阵验证、保持低秩计算且范围明确的修复纳入生产代码。

**Tech Stack:** vLLM c410543d9、BF16 checkpoint、Ascend NPU、TP1及TP2/DP2/EP、eager、batch invariance、pytest。

## Task 1：单层独立对照

- 扩展 `tests/e2e/models/minimax_m31_lora` 诊断工具：读取真实 Q/O 权重及已有共同输入快照，比较低秩和 merged、TP1和TP2局部分片计算。
- 新增完整FP32累加的低秩参考，区别于此前保留BF16基座输出的FP64增量对照。
- 量化 weight rounding 和 partial-output rounding，保留原始结果及全部失败项；不能通过调小adapter或缩短生成使其通过。

## Task 2：因果对照与修复

- 分别验证 FP32基座/增量累加、O投影低秩shrink在FP32归约后expand的效果。
- 跨TP先对相同输入复现第一个分歧模块，核对TP本身和EP/DP拓扑的贡献；不将改变EP拓扑的结果误报为纯TP问题。
- 若发现加载、索引、累加或归约实现错误，局部修改 `vllm_ascend/lora/minimax_m3.py` 及所属测试；其他模型的默认逻辑保持原样。
- 若属于两条不同BF16求值公式不可等价，给出最小实测反例，保留未通过状态，不伪造兼容承诺。

## Task 3：回归与交付

- 原A/B四条prompt、32-token、同拓扑merged8项、混合请求36项以及TP1/TP2各自参考单列。
- 生产修改执行相关UT；诊断工具验证实际NPU运行、结果导出和复现说明。
- 中文报告和脱敏证据推送个人分支与ref-doc；记录有效修复、未解决项及原因。

## 执行结果

- 原标准低秩模型3/8、混合服务36/36再次复现；两种精度控制在TP2均有服务回归，未合入生产代码。
- TP1与TP2相同模式的标准低秩输出均仅3/8一致；真实两卡HCCL单层对照在BI模式下也保留差异。
- 独立FP64单次舍入仍与BF16 merged不同；Q/O有效更新64×64子矩阵均为秩64。详细数据与复现工具见 [定位报告](../../tests/e2e/models/minimax_m31_lora/STANDARD_LOW_RANK.md)。
