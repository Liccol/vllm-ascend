# RL MX 低精度权重更新测试设计

## 1. 结论

最精简且有效的测试不需要启动完整 RL 训练。测试应直接覆盖真实的 Ascend MX
量化、vLLM 权重加载和推理布局恢复流程，并采用以下正确性基准：

```text
使用更新后 Actor 权重全新加载的模型 == 热更新到已有模型后的结果
```

必须同时比较低精度权重、scale 和模型输出。只检查 shape、状态标志或者“输出
发生变化”是不充分的，因为在权重丢失、融合层名称映射错误、scale 加载错误或
布局重复转换时，这类弱断言仍可能通过。

建议首个用例放在 `tests/e2e/pull_request/one_card/` 下，使用一张 NPU，参数化
覆盖以下场景：

1. Dense `W8A8_MXFP8`。
2. Dense `W4A8_MXFP`，即 MXFP4 权重和动态 MXFP8 激活。
3. MoE `W8A8_MXFP8`。
4. MoE `W4A8_MXFP`。
5. 同时包含 MX 层和 BF16/跳过量化层的混合模型。

现有的双卡 HCCL 权重传输测试可以在后续复用同一正确性基准，作为传输层集成
测试。它不应成为首个或唯一的测试，否则量化、布局转换、模型启动和跨进程传输
问题会混在一起，难以定位失败原因。

需要明确一个范围边界：当前 verl 中已有的 QAT 实现面向 NVFP4；当前 Ascend
MX 在线权重更新路径支持 `W8A8_MXFP8` 和 `W4A8_MXFP`。因此，本设计现在能
验证 PTQ 式的 MX rollout 权重更新，但在建立 Ascend MX 的假量化训练和导出
协议前，不能宣称它验证了 Ascend MX QAT。

## 2. 调研基线

本文档基于以下本地源码版本：

| 仓库 | 版本 |
|---|---|
| verl | `3e0d79ea8c38264bcaddc64de1b3465589580320` |
| vLLM | `e2ad0ce5f5a944e10ed2ea664e4a24bf85791346` |
| vllm-ascend | `33a8b92bdc00dc83c6088d51a397f0cab2f98761` |

本轮调研没有在 NPU 上执行测试。具体硬件支持、数值容差和可用的小模型需要在
实现阶段验证，因此这些部分目前仍不确定。

## 3. verl 当前的 PTQ 式 rollout 流程

这里所说的“PTQ 训练”实际是：Actor 继续使用 BF16/FP16 训练，在 Actor 权重
同步到低精度 rollout 引擎时进行在线量化。Actor 训练过程本身没有插入假量化。

```text
PPO/GRPO Actor 执行 optimizer step
        |
        v
FSDP/Megatron get_per_tensor_param()
  - 将训练权重转换成完整的 HF 坐标张量
  - 必要时拆分训练侧融合的 MoE 权重
        |
        v
CheckpointEngine / 分桶权重传输
        |
        v
vLLMColocateWorkerExtension.update_weights_from_ipc()
  1. 将 MX 推理布局恢复成 checkpoint 加载布局
  2. 接收一个或多个权重 bucket
  3. 对每个 bucket 的 BF16 权重做 MX 量化并调用 model.load_weights()
  4. 最后一个 bucket 完成后，重新构建 Ascend 推理布局
        |
        v
清理 KV cache，并发布新的 global step
```

### 3.1 Actor 权重导出

PPO/GRPO 的训练循环在 Actor 更新完成后调用 checkpoint manager 进行权重同步。
FSDP 和 Megatron 的 `get_per_tensor_param()` 负责将训练侧权重转换成 rollout
加载器使用的名称和张量形式。

FSDP 路径会从 `state_dict` 取得参数，将 DTensor materialize 为完整张量，并将
训练侧融合 MoE 权重转换为 rollout 需要的权重名称。普通 PTQ 路径输出的是
BF16/FP16 张量，而不是预先量化的 MX 张量。

### 3.2 权重传输

`checkpoint_manager.update_weights()` 将 Actor 权重交给具体 checkpoint engine。
vLLM rollout 侧启动 `update_weights_from_ipc()`，通过 bucket receiver 接收权重。

这里的关键语义是：

- prepare 只能在第一个 bucket 前执行一次；
- 每个 bucket 可以分别调用权重加载；
- post-process 只能在最后一个 bucket 完成后执行一次；
- 更新完成后需要清除旧 KV cache。

如果在每个 bucket 后都执行推理布局转换，后续 bucket 会尝试把 checkpoint
布局数据写入已经转置或格式转换后的参数，从而造成 shape、stride 或数据错误。

### 3.3 在线 MX 量化

实际转换逻辑位于 `verl/utils/vllm/vllm_quant_utils.py::quant_weights`：

- `is_vllm_ascend_mx_quant` 识别包含 `W8A8_MXFP8` 或 `W4A8_MXFP` 的
  `AscendModelSlimConfig`；
- Dense 或融合 MoE 权重被分类为 `mx_fp8` 或 `mx_fp4`；
- 调用 `torch_npu.npu_dynamic_mx_quant(..., axis=-1)` 生成低精度权重和
  E8M0 分组 scale；
- MXFP8 权重使用原权重名称，并额外生成 `<weight_name>_scale`；
- MXFP4 输出被重新解释为 `uint8`，因为一个字节中打包两个 FP4 值，且
  vllm-ascend 的对应参数使用 `uint8` 保存；
- 不参与量化的参数和 buffer 原样透传。

### 3.4 checkpoint 布局与推理布局切换

加载前，`prepare_quanted_weights_for_loading` 调用各 Ascend 量化 scheme 的
`restore_weights_for_rl_loading`。所有 bucket 完成后，
`process_quanted_weights_after_loading` 调用 `process_weights_after_loading`。

这个顺序不能省略：vLLM weight loader 接收 checkpoint 布局，而 Ascend 量化
kernel 使用转置后的布局，部分路径还需要 FRACTAL_NZ 格式。scale 在 checkpoint
和推理阶段的 shape/维度顺序也不同。

MXFP8 路径会保留持久化的转换后 buffer，并在后续更新时原地写入，从而保持
ACL Graph 捕获的 `data_ptr` 不变。Dense 和 MoE 已有单元测试覆盖部分 shape
恢复和指针稳定性，但其中 NPU 格式转换大量使用 mock，不能证明真实低精度数据
已经正确进入实际 kernel。

MXFP4 的 Dense 和 MoE scheme 会在加载前恢复 packed weight 和 scale 的原始
checkpoint shape，加载后再做 format cast、reshape 和 transpose。现有单元测试
主要验证 shape、幂等性和基本生命周期，没有在 NPU 上比较热更新模型与全新
加载模型的实际计算结果。

## 4. verl 当前的 QAT 流程

### 4.1 FSDP QAT

FSDP 路径实现的是 NVFP4 QAT：

```text
构建 BF16 模型
  -> 将符合条件的 nn.Linear 替换为 QATLinear
  -> 训练前向使用 FP4 假量化和 STE
  -> optimizer 更新 BF16 master weight
  -> 使缓存的 weight scale 失效
  -> get_per_tensor_param()
  -> QATQuantizer 打包真实 NVFP4 rollout 权重
  -> 分桶加载到 compressed-tensors vLLM 模型
  -> 最后一个 bucket 后执行 manual_process_weights_after_loading()
```

`QATLinear` 的代码支持 `w4a4` 和 `w4a16` 假量化。权重梯度通过
straight-through estimator 传递。`w4a4` 还维护激活 observer 和
`input_global_scale`。

导出阶段，`QATQuantizer` 为量化层生成：

- `weight_packed`；
- `weight_scale`；
- `weight_global_scale`；
- `w4a4` 模式额外生成 `input_global_scale`。

rollout 配置注入 `compressed-tensors` 量化配置。reload 前，verl 会重新构造
被 vLLM post-load 过程删除或替换的 checkpoint 布局参数；最后一个 bucket 完成
后，再手工触发所有量化层的 post-process。

### 4.2 Megatron QAT

Megatron 路径使用 NVIDIA ModelOpt。当前 ModelOpt 量化配置构造器只接受
`w4a16`：

```text
Megatron modules
  -> 应用 ModelOpt 假量化
  -> optimizer 更新 BF16 master weight
  -> QATWeightExporter 读取量化器 metadata 和 amax
  -> 按 HF 坐标导出 packed NVFP4 权重及 scale
  -> 加载到 vLLM modelopt 量化路径
  -> 执行 ModelOpt 专用 post-load 处理
```

导出器还负责 Megatron 到 HF 的名称映射，以及 PP/EP 场景中的 metadata 同步。
rollout worker 使用单独的 ModelOpt prepare/process patch。

### 4.3 与 Ascend MX 目标之间的缺口

上述 QAT 路径针对 NVIDIA NVFP4 布局以及 vLLM 的
`compressed-tensors`/`modelopt` quant method。vllm-ascend 的 MX 路径使用
E8M0 microscaling、`torch_npu.npu_dynamic_mx_quant` 和
`AscendModelSlimConfig`。

在本次检查的代码中，没有发现将 FSDP 或 Megatron QAT 状态导出成 Ascend
`W8A8_MXFP8` 或 `W4A8_MXFP` 权重的训练和导出协议。因此：

- 现在可以验证 PTQ 式的 Ascend MX 权重更新；
- QAT 中“prepare 一次、加载全部 bucket、process 一次”的 reload 状态机可以
  复用到 MX 测试；
- 真正的“MX QAT”还需要先定义训练侧假量化算法、scale 语义、导出名称和
  Ascend kernel 支持；
- 不能把 NVFP4 packed 权重直接当成 MXFP4 权重使用。

## 5. 测试架构

### 5.1 推荐测试边界

主测试直接实例化最小 vLLM 模型，或者由真实 vLLM Dense/MoE 层构成的最小
模型，并使用 vllm-ascend 的真实量化 scheme。测试必须使用真实 `torch_npu`
操作和实际 `load_weights` 路径。

Ray、PPO/GRPO 完整训练循环、数据集、奖励计算、generation scheduler 和跨进程
权重传输不进入主测试。

```text
确定性的 BF16 W0                         确定性的 BF16 W1
        |                                       |
        | 在线 MX 量化                          | 在线 MX 量化
        v                                       v
初始量化模型                         预期的 checkpoint 布局张量
        |                                       |
        | 转换为推理布局                         |
        v                                       |
固定输入 X -> Y0                               |
        |                                       |
        +----- restore -> load -> process <------+ 
                               |
                               v
                     热更新模型 -> Yhot

全新模型 <- 正常加载同一份 W1 payload -> Yfresh

必须满足：Yhot == Yfresh，量化张量一致，并且 Yhot != Y0
```

测试侧只保留一个很小的 producer helper，调用与 verl 相同的公开 NPU 算子：

```python
low_precision, scale = torch_npu.npu_dynamic_mx_quant(
    bf16_weight,
    axis=-1,
    dst_type=dst_type,
)
scale = scale.flatten(-2, -1).squeeze(-1)
```

MXFP4 场景将 `low_precision.view(torch.uint8)` 交给 loader。该 helper 需要检查
输出名称、dtype 和 shape 协议，但不能复制另一套模型加载或布局转换实现。

### 5.2 正确性基准选择

| 方案 | 优点 | 缺陷 | 结论 |
|---|---|---|---|
| 只检查权重或输出发生变化 | 实现最小 | 丢失部分权重或名称映射错误时仍可能通过 | 不采用 |
| 与 BF16 输出比较 | 能发现严重数值错误 | 需要格式相关反量化和容差，且无法分离 reload 错误与量化误差 | 只作辅助指标 |
| 热更新与同 payload 的全新加载比较 | 覆盖名称、loader、派生状态、布局转换和 kernel 执行 | 需要两个模型或顺序重建 | 采用 |

现有 vllm-ascend RLHF E2E 已经使用“正常启动加载作为 reference”的原则验证
BF16 权重更新。新的单卡 MX 用例应使用更轻量的进程内版本。后续双卡测试可以
复用同一判定标准。

### 5.3 单个测试事务

每个参数化场景执行以下步骤：

1. 固定随机种子并构造小型输入张量 `X`。
2. 构造 BF16 Actor 权重 `W0`。数值需要覆盖不同 MX 量化区间，不能全部为零
   或常量。
3. 量化 `W0`，通过正常加载路径加载，并执行 scheme 的 post-load 处理。
4. 保存输出 `Y0`、checkpoint 布局元信息、转换后 shape 和所有持久推理 buffer
   的地址。
5. 构造 `W1 = W0 + delta`。`delta` 必须足够大，确保 packed value 和 block
   scale 都发生变化。
6. 在修改已有模型前，独立生成预期的 W1 payload。
7. 调用生产 reload 生命周期：
   `restore_weights_for_rl_loading -> load_weights ->
   process_weights_after_loading`。
8. 使用相同输入得到 `Yhot`。
9. 创建全新模型，正常加载完全相同的 W1 payload，得到 `Yfresh`。
10. 在原热更新模型上使用 `W2` 再执行一轮，捕获只在首次 reload 正常的问题。

测试还应将 weight 和 scale 拆成两个逻辑 bucket。只有最后一个 bucket 完成后
才能执行 post-process，以检测过早转换推理布局的问题。

## 6. 必须检查的断言

### 6.1 Payload 协议

- 所有预期权重名称都由 loader 报告为已加载；
- 没有量化权重或 scale 缺失；
- ignored/BF16 张量保持 BF16 dtype，并准确接收 W1 数值；
- MXFP8 权重和 scale dtype 符合 scheme 协议；
- MXFP4 在 loader 边界为 `uint8`，packed 维度存储宽度减半；
- scale shape 等于 `ceil(input_size / group_size)`；
- row-parallel shard 保持全局 MX group 相位，不能从错误 block 开始取 scale。

### 6.2 生命周期和存储

- 加载前恢复的 shape 等于记录的 checkpoint shape；
- 加载后的 shape 和 NPU format 等于全新加载模型的推理布局；
- post-process 后 `_mxfp8_transformed` 或 `_mxfp4_transformed` 恢复为 `True`；
- 未执行 restore 时，重复调用 post-process 不产生二次转换；
- 对 scheme 承诺地址稳定的持久 buffer，W1 和 W2 更新后 `data_ptr` 不变；
- post-process 在全部 bucket 完成后只执行一次。

### 6.3 数值和执行结果

- checkpoint 布局中的 W1 量化 weight 字节与 producer 预期完全一致；
- W1 scale 字节与预期完全一致；
- `Yhot` 与 `Yfresh` 一致；
- 相同 kernel 路径可确定执行时，优先要求完全相等；
- 如果硬件 kernel 存在非 bitwise 稳定性，需要先通过重复执行测得波动，再记录
  并使用严格的最小容差；
- `Yhot` 与 `Y0` 的差异超过最小 norm 阈值，证明更新真正生效；
- W2 热更新同样与 W2 全新加载 reference 一致；
- 输出中没有 NaN 或 Inf。

MoE 场景必须控制路由，使每个被更新的 expert 至少接收一个 token，否则陈旧
expert 可能逃过输出比较。同时需要直接比较 `w13_weight`、`w2_weight` 和两个
scale 张量。

## 7. 最小测试矩阵

| 场景 | 层类型 | 权重格式 | 激活格式 | 特殊覆盖点 |
|---|---|---|---|---|
| `dense_mxfp8` | Dense Linear | MXFP8 | 动态 MXFP8 | 持久图 buffer |
| `dense_mxfp4` | Dense Linear | packed MXFP4 | 动态 MXFP8 | packed dtype 和 shape |
| `moe_mxfp8` | Routed Experts | MXFP8 | 动态 MXFP8 | 融合 expert 名称和 scale |
| `moe_mxfp4` | Routed Experts | packed MXFP4 | 动态 MXFP8 | `w13`/`w2` 及 expert 布局 |
| `mixed_mxfp8_bf16` | 两层最小模型 | MXFP8 + BF16 | 混合 | 跳过量化层透传 |

以下内容作为后续扩展，不阻塞首版：

- tensor parallel 大于 1；
- row-parallel 分片边界不能被 MX group size 整除；
- MegaMoE 和非 MegaMoE 两套布局；
- ACL Graph 捕获、权重更新和 replay；
- HCCL 与共享内存传输；
- 真实 packed QKV/gate-up projection 名称映射；
- cache sleep/resume 打开时的权重更新。

vllm-ascend 中还存在 `W4A4_MXFP4` 和 `W4A16_MXFP4`，但当前检查到的 verl
在线 MX classifier 没有导出这两种格式。生产者协议支持它们之前，不应把它们
加入 RL 权重同步测试并声称已经覆盖。

## 8. 建议文件位置

主测试：

```text
tests/e2e/pull_request/one_card/test_rl_mx_weight_update.py
```

如果测试无法保持在一个文件内，可增加共享 helper：

```text
tests/e2e/pull_request/rlhf/mx_weight_update_test_utils.py
```

后续传输层扩展：

```text
tests/e2e/pull_request/two_card/rlhf/state_transitions/
    test_hccl_mx_weight_transfer.py
```

verl 中可以增加一个小型 CPU 单元测试，验证 `vllm_quant_utils.py` 的权重分类
和名称协议。但它不能替代 NPU 测试，因为关键量化和布局转换是设备相关操作。

## 9. 失败定位

| 失败表现 | 优先检查范围 |
|---|---|
| 加载前预期量化字节已经不同 | producer 量化、dtype 或 view 协议 |
| loader 报告权重名称缺失 | HF 到 vLLM 或融合 MoE 名称映射 |
| process 前字节一致，模型输出与 fresh model 不同 | restore/process 布局或派生状态 |
| 第一次更新通过，第二次失败 | 非幂等转换或陈旧 reload state |
| eager 通过，ACL Graph replay 失败 | buffer 地址改变或图捕获状态陈旧 |
| Dense 通过，MoE 失败 | expert 融合、路由覆盖、EP/MegaMoE 布局 |
| 单卡通过，HCCL 测试失败 | 传输 metadata、分片或事务顺序 |
| MX 通过但所谓 QAT 测试无法启动 | NVFP4 到 Ascend MX 协议尚不存在，不属于 MX reload 回归 |

失败日志应打印张量名称、checkpoint shape/dtype、推理 shape/dtype、预期和实际
数据摘要，以及错误发生在 restore、load 还是 process 阶段。不需要打印完整
张量，否则 CI 日志会过大并降低可读性。

## 10. 非功能要求

- 主测试只使用一张 NPU，不启动网络服务；
- 输入和 payload 完全确定；
- 依赖和模型结构可用后，目标运行时间小于两分钟；
- 不依赖完整 PPO/GRPO trainer、Ray、数据集、token generation 或 reward model；
- 不依赖私有 checkpoint；
- 优先使用合成的小型真实层；
- 如果必须使用真实模型类，通过 config 减少层数和 expert 数量；
- 失败信息必须定位到具体张量和生命周期阶段；
- 测试数据在内存中生成，并自动清理临时文件。

## 11. 风险与缓解措施

| 风险 | 缓解措施 |
|---|---|
| fake layer 绕过真实 vLLM loader | 使用真实 vLLM Linear/MoE 模块和实际 `load_weights` 路径 |
| 输出断言没有覆盖全部 expert | 强制确定性路由，并直接比较 expert 权重 |
| 测试用另一份相同实现作为 oracle | 只生成一份 payload，比较热更新模型与独立全新模型 |
| 量化噪声掩盖陈旧权重 | W1/W2 必须改变 payload 摘要，并断言 `Yhot != Y0` |
| 输出完全相等在硬件上不稳定 | 先测重复执行波动，再制定有依据的最小容差 |
| 混合量化 metadata 依赖模型名称 | 首版使用明确的小型 quant description，后续补真实模型映射 |
| 将 QAT 错标为 Ascend MX | 在 MX QAT 导出协议建立前，始终与 NVFP4 QAT 分开描述 |

## 12. 架构决策记录

### ADR-001：在 RL trainer 以下测试 reload 事务

**状态：** 待确认。

**背景：** 完整 RL 运行包含优化、分布式调度、数据、奖励、生成和权重传输。
目标问题只关注更新后的 Actor 权重是否正确转换并安装到 vllm-ascend。

**决策：** 使用确定性的 Actor 风格 BF16 张量，直接驱动 MX 量化以及真实的
vLLM/vllm-ascend reload 生命周期。

**影响：** 测试更小、失败更容易定位，但它不证明 PPO 或 checkpoint transport
backend 正确。传输边界由现有和后续传输测试覆盖。

### ADR-002：使用全新加载作为正确性基准

**状态：** 待确认。

**背景：** 内部状态标志和 shape round trip 不能证明权重语义正确。

**决策：** 热更新模型和正常全新加载同一 W1/W2 payload 的模型，必须得到一致
的张量与输出。

**影响：** 相比纯单元测试，需要重建模型并增加一些显存和时间开销，但不需要
实现自定义反量化器，就能检测陈旧派生状态和错误布局转换。

### ADR-003：分开处理 PTQ 式 MX 与 NVFP4 QAT

**状态：** 待确认。

**背景：** 名称容易让人误以为任意 4-bit QAT 产物都属于 MX。实际上它们的
scale 语义、packing、vLLM quant method 和平台 kernel 不同。

**决策：** 首个 vllm-ascend 测试只以 `W8A8_MXFP8` 和 `W4A8_MXFP` 为合入
门禁。记录 NVFP4 QAT 的现状，但不将其作为 Ascend MX 正确性的证据。

**影响：** 首版可以对当前生产 MX PTQ reload 路径给出可靠结论。未来增加 MX
QAT 时，需要新增训练/导出协议测试，但可以复用全新加载 reference 的判定方法。

## 13. 实现验收标准

满足以下条件时，可认为设计已经实现：

1. 五个最小矩阵场景均可在一张 NPU 上运行；
2. 每个场景执行两轮热更新；
3. 量化权重、scale、跳过量化的 BF16 权重和输出均与全新加载 reference 一致；
4. 初始输出与更新后输出不同；
5. 所有承诺持久化的 buffer 地址保持稳定；
6. 人为删除 scale 或跳过 post-process 时，测试会失败并给出明确定位；
7. 测试不启动 RL trainer，也不需要下载完整模型权重。
