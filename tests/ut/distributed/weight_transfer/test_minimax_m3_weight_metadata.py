# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Checkpoint coverage for MiniMax-M3's IPC weight-update case."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from tests.e2e.pull_request.rlhf import weight_transfer_test_utils as utils


@pytest.fixture
def text_config():
    # Original dimensions from MiniMax/MiniMax-M3 config.json, with the same
    # layer/expert overrides used by the IPC server. No checkpoint download.
    return PretrainedConfig(
        hidden_size=6144,
        vocab_size=200064,
        head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=4,
        intermediate_size=3072,
        dense_intermediate_size=12288,
        n_shared_experts=1,
        use_routing_bias=True,
        **utils.MINIMAX_M3_CASE.hf_overrides["text_config"],
    )


def test_minimax_metadata_covers_the_complete_text_checkpoint(text_config):
    metadata = utils.minimax_m3_parameter_metadata(text_config)
    actual = {meta.name: (meta.shape, meta.dtype) for meta in metadata}
    expected = {
        "language_model.model.embed_tokens.weight": ((200064, 6144), torch.bfloat16),
        "language_model.model.norm.weight": ((6144,), torch.bfloat16),
        "language_model.lm_head.weight": ((200064, 6144), torch.bfloat16),
    }
    # Independent expected checkpoint schema: dense layers 0..2, sparse/MoE
    # layer 3. In particular index_k is NOT num_index_heads * index_dim.
    for layer in range(4):
        prefix = f"language_model.model.layers.{layer}"
        shapes = {
            "input_layernorm.weight": (6144,),
            "post_attention_layernorm.weight": (6144,),
            "self_attn.q_proj.weight": (8192, 6144),
            "self_attn.k_proj.weight": (512, 6144),
            "self_attn.v_proj.weight": (512, 6144),
            "self_attn.o_proj.weight": (6144, 8192),
            "self_attn.q_norm.weight": (128,),
            "self_attn.k_norm.weight": (128,),
        }
        if layer < 3:
            shapes.update(
                {
                    "mlp.gate_proj.weight": (12288, 6144),
                    "mlp.up_proj.weight": (12288, 6144),
                    "mlp.down_proj.weight": (6144, 12288),
                }
            )
        else:
            shapes.update(
                {
                    "self_attn.index_q_proj.weight": (512, 6144),
                    "self_attn.index_k_proj.weight": (128, 6144),
                    "self_attn.index_q_norm.weight": (128,),
                    "self_attn.index_k_norm.weight": (128,),
                    "block_sparse_moe.shared_experts.gate_proj.weight": (3072, 6144),
                    "block_sparse_moe.shared_experts.up_proj.weight": (3072, 6144),
                    "block_sparse_moe.shared_experts.down_proj.weight": (6144, 3072),
                }
            )
            for expert in range(8):
                shapes[f"block_sparse_moe.experts.{expert}.w1.weight"] = (3072, 6144)
                shapes[f"block_sparse_moe.experts.{expert}.w3.weight"] = (3072, 6144)
                shapes[f"block_sparse_moe.experts.{expert}.w2.weight"] = (6144, 3072)
            expected[f"{prefix}.block_sparse_moe.gate.weight"] = ((8, 6144), torch.float32)
            expected[f"{prefix}.block_sparse_moe.e_score_correction_bias"] = ((8,), torch.float32)
        expected.update({f"{prefix}.{name}": (shape, torch.bfloat16) for name, shape in shapes.items()})
    assert len(metadata) == len(actual)  # no duplicate names
    assert actual == expected


def test_minimax_source_avoids_transformers_model_construction(text_config):
    config = SimpleNamespace(text_config=text_config)
    with (
        patch.object(utils.AutoConfig, "from_pretrained", return_value=config),
        patch.object(utils.AutoModelForCausalLM, "from_config") as build_model,
    ):
        source = utils.FixedRandomWeightSource(utils.MINIMAX_M3_CASE, torch.device("cpu"))
    build_model.assert_not_called()
    assert source.metadata() == utils.minimax_m3_parameter_metadata(text_config)
    # The original embedding exceeds the default 1 GiB packed buffer.
    assert utils.packed_buffer_size_for(source) > 200064 * 6144 * torch.bfloat16.itemsize


@pytest.mark.parametrize("duplicate", [False, True], ids=["empty", "duplicate"])
def test_metadata_factory_rejects_invalid_parameter_sets(text_config, duplicate):
    metadata = utils.minimax_m3_parameter_metadata(text_config)
    case = replace(utils.MINIMAX_M3_CASE, parameter_metadata_factory=lambda _: [metadata[0]] * 2 if duplicate else [])
    with (
        patch.object(utils.AutoConfig, "from_pretrained", return_value=SimpleNamespace(text_config=text_config)),
        pytest.raises(AssertionError, match="not unique" if duplicate else "no parameters"),
    ):
        utils.FixedRandomWeightSource(case, torch.device("cpu"))


def test_minimax_is_added_only_to_ipc_matrix():
    assert utils.MINIMAX_M3_CASE not in utils.MODEL_CASES
    cases = utils.pytest_model_cases((*utils.MODEL_CASES, utils.MINIMAX_M3_CASE))
    assert cases[-1].values == (utils.MINIMAX_M3_CASE,)
    assert cases[-1].marks[0].args == ("MiniMax/MiniMax-M3",)
    assert utils.MINIMAX_M3_CASE.skip_reason is None
