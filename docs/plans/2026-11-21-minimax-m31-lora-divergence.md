# M3.1 BF16 LoRA token 分歧定位

沿用原四层 BF16 base、std0.1 的独立 A/B、固定四条 prompt 和32个生成 token。
保持 token ID、长度和 finish_reason 全一致的验收条件，不更换权重或删减失败用例。
vLLM 固定 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`。

1. 在原 TP2/DP2/EP、batch-invariance 环境复现5项模型分歧，保存新结果。
2. 通过测试 worker extension，在共同输入前缀下记录各层输入、QKV/indexer、O、hidden state 与最终 logits。
3. 核对实际加载的全部目标层 A/B、alpha、TP 分片以及 base/merged 权重；逐投影使用独立公式检查，不仅检查第0层 Q。
4. 对首次分歧位置做 teacher forcing，排除生成历史不同造成的后续误差；比较 top1/top2 间隔、路由和数值误差。
5. 用测试专用参考实现替换 LoRA 低秩 kernel 做因果对照；任何替换只作定位，不能冒充生产路径验收通过。
6. 发现实现错误时在对应生产模块局部修复，增加原始失败回归，再重跑模型和混合服务规则。
7. 保留不通过记录；若根因是 BF16 merged 与动态计算无法保证完全相等，给出共同输入下的可复现反例，不伪造通过或暗改验收。

测试辅助代码放在 `tests/e2e/models/minimax_m31_lora/`，不引入生产环境变量，不改上游 vLLM 源码，不影响其他任务进程。
