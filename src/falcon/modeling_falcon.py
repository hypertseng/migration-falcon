"""
Falcon model
"""

# pylint: disable=C0103


import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np

import mindspore
from mindspore import nn, ops
from mindspore.nn import Dense as FalconLinear
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp._legacy import functional as F
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)


from ...modeling_utils import PreTrainedModel
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from .configuration_falcon import FalconConfig


logger = logging.get_logger(__name__)

FALCON_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tiiuae/falcon-40b",
    "tiiuae/falcon-40b-instruct",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-rw-7b",
    "tiiuae/falcon-rw-1b",
]
_CHECKPOINT_FOR_DOC = "Rocketknight1/falcon-rw-1b"
_CONFIG_FOR_DOC = "FalconConfig"

def rotate_half(x):
    """
    Rotates the input tensor by half along the last dimension.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The rotated tensor."""

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(padding_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`mindspore.Tensor`): The query tensor.
        k (`mindspore.Tensor`): The key tensor.
        cos (`mindspore.Tensor`): The cosine part of the rotary embedding.
        sin (`mindspore.Tensor`): The sine part of the rotary embedding.
        position_ids (`mindspore.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(mindspore.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FalconRotaryEmbedding(nn.Cell):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with `[batch_size,
    n_heads_per_partition, seq_len, head_dim]` (e.g. MinGPTAttention format).
    """

    def __init__(self, dim: int, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq = 1.0 / (self.base ** (ops.arange(0, dim, 2).float() / dim))

        # mindspore does not support get_default_dtype()
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=self.inv_freq.dtype
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # freqs = ops.matmul()(t.reshape(-1, 1), self.inv_freq.reshape(1, -1))
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().astype(dtype)
        self.sin_cached = emb.sin().astype(dtype)

    def construct(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].astype(dtype=x.dtype),
            self.sin_cached[:seq_len].astype(dtype=x.dtype),
        )


class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self, dim: int, max_position_embeddings=2048, base=10000, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(seq_len, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = ops.outer(t, self.inv_freq)
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().astype(dtype)
        self.sin_cached = emb.sin().astype(dtype)


class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """
    FalconRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self, dim: int, max_position_embeddings=2048, base=10000, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base ** (ops.arange(0, self.dim, 2).float() / self.dim)
            )

        t = ops.arange(seq_len)
        freqs = ops.outer(t, self.inv_freq)
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().astype(dtype)
        self.sin_cached = emb.sin().astype(dtype)


def _prepare_4d_attention_mask(
    mask: mindspore.Tensor, past_key_values_length: int
) -> mindspore.Tensor:
    """
    Expands attention_mask from `[batch_size, seq_length]` to `[batch_size, 1, seq_length, seq_length + past_length]`.
    """
    batch_size, total_length = mask.shape
    seq_length = (
        total_length - past_key_values_length
        if past_key_values_length is not None
        else total_length
    )

    expanded_mask = ~(mask[:, None, None, :].bool())
    return expanded_mask.broadcast_to((batch_size, 1, seq_length, total_length))


def build_alibi_tensor(
    attention_mask: mindspore.Tensor, num_heads: int, dtype: mindspore.dtype
) -> mindspore.Tensor:
    """
    Builds the alibi tensor used for attention bias in the Falcon model.

    Args:
        attention_mask (mindspore.Tensor): The attention mask tensor.
        num_heads (int): The number of attention heads.
        dtype (mindspore.dtype): The data type of the tensor.

    Returns:
        mindspore.Tensor: The alibi tensor of shape (batch_size * num_heads, 1, seq_length).
    """

    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = mindspore.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=mindspore.float32
    )
    powers = ops.arange(1, 1 + closest_power_of_2, dtype=mindspore.int32)
    slopes = ops.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = mindspore.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=mindspore.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = ops.arange(
            1, 1 + 2 * num_remaining_heads, 2, dtype=mindspore.int32
        )
        slopes = ops.cat([slopes, ops.pow(extra_base, extra_powers)], axis=0)

    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None].astype(mindspore.float32) * arange_tensor
    return ops.reshape(alibi, (batch_size * num_heads, 1, seq_length)).astype(dtype)


# Copied from transformers.models.bloom.modeling_bloom.dropout_add
def dropout_add(
    x: mindspore.Tensor, residual: mindspore.Tensor, prob: float, training: bool
) -> mindspore.Tensor:
    """
    Dropout add function

    Args:
        x (`mindspore.tensor`, *required*):
            input tensor
        residual (`mindspore.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = ops.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class FalconAttention(nn.Cell):
    """
    FalconAttention is a module that implements the attention mechanism used in the Falcon model.

    Args:
        config (FalconConfig): The configuration object that contains various hyperparameters for the Falcon model.

    Raises:
        ValueError: If `hidden_size` is not divisible by `num_heads`.

    Attributes:
        config (FalconConfig): The configuration object that contains various hyperparameters for the Falcon model.
        hidden_size (int): The size of the hidden state.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        split_size (int): The size of the split dimension.
        hidden_dropout (float): The dropout rate for the hidden states.
        max_position_embeddings (int): The maximum number of position embeddings.
        rope_theta (float): The theta value for the RoPE (Rotary Position Embedding).
        is_casual (bool): Whether the attention is causal or not.
        inv_norm_factor (float): The inverse normalization factor for layer-wise attention scaling.
        beta (float): The beta value for layer-wise attention scaling.
        new_decoder_architecture (bool): Whether to use the new decoder architecture or not.
        multi_query (bool): Whether to use multi-query attention or not.
        num_kv_heads (int): The number of key-value attention heads.

    """

    def __init__(self, config: FalconConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        if config.rotary:
            self._init_rope()

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            qkv_out_dim = (
                config.num_kv_heads * 2 + config.num_attention_heads
            ) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(
            self.hidden_size, qkv_out_dim, has_bias=config.bias
        )
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(
            self.hidden_size, self.hidden_size, has_bias=config.bias
        )
        self.attention_dropout = nn.Dropout(p=config.attention_dropout)
        self.num_kv_heads = (
            config.num_kv_heads
            if (self.new_decoder_architecture or not self.multi_query)
            else 1
        )

    def _init_rope(self):
        """
        Initialize the Rotary Position Embedding (RoPE) based on the configuration.

        Raises:
            ValueError: If the RoPE scaling type is unknown.

        """
        if self.config.rope_scaling is None:
            self.rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                base=self.config.rope_theta,
                max_position_embeddings=self.config.max_position_embeddings,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _split_heads(
        self, fused_qkv: mindspore.Tensor
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`mindspore.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]

        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(
                batch,
                seq_len,
                -1,
                self.num_heads // self.num_kv_heads + 2,
                self.head_dim,
            )
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = ops.broadcast_to(key, query.shape)
            value = ops.broadcast_to(value, query.shape)

            query, key, value = [
                x.flatten(start_dim=2, end_dim=3) for x in (query, key, value)
            ]
            return query, key, value
        if not self.multi_query:
            batch_size, seq_length, _ = fused_qkv.shape
            fused_qkv = fused_qkv.view(
                batch_size, seq_length, self.num_heads, 3, self.head_dim
            )
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        batch_size, seq_length, _ = fused_qkv.shape
        fused_qkv = fused_qkv.view(
            batch_size, seq_length, self.num_heads + 2, self.head_dim
        )
        return (
            fused_qkv[..., :-2, :],
            fused_qkv[..., [-2], :],
            fused_qkv[..., [-1], :],
        )

    def _merge_heads(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`mindspore.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            mindspore.tensor: [batch_size, seq_length, num_heads * head_dim]

        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        alibi: Optional[mindspore.Tensor],
        attention_mask: mindspore.Tensor,
        position_ids: Optional[mindspore.Tensor] = None,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Apply the FalconAttention mechanism to the input hidden states.

        Args:
            hidden_states (mindspore.Tensor): The input hidden states of shape [batch_size, seq_length, hidden_size].
            alibi (mindspore.Tensor, optional): The alibi tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask (mindspore.Tensor): The attention mask tensor of shape [batch_size, seq_length].
            position_ids (mindspore.Tensor, optional): The position ids tensor of shape [batch_size, seq_length].
            layer_past (Tuple[mindspore.Tensor, mindspore.Tensor], optional): The past key-value states of the layer.
            head_mask (mindspore.Tensor, optional): The head mask tensor of shape [num_heads].
            use_cache (bool, optional): Whether to use the cache or not.
            output_attentions (bool, optional): Whether to output the attention scores or not.

        Returns:
            Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]], Optional[mindspore.Tensor]]:
                - output_tensor (mindspore.Tensor): The output tensor of shape [batch_size, seq_length, hidden_size].
                - present (Tuple[mindspore.Tensor, mindspore.Tensor], optional): The present key-value states of the layer.
                - attention_scores (mindspore.Tensor, optional): The attention scores tensor of shape [batch_size, num_heads, seq_length, seq_length].

        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = (
            self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        )
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.swapaxes(1, 2).reshape(
            batch_size, self.num_heads, query_length, self.head_dim
        )
        key_layer = key_layer.swapaxes(1, 2).reshape(
            batch_size,
            num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.swapaxes(1, 2).reshape(
            batch_size, num_kv_heads, query_length, self.head_dim
        )

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer, key_layer, cos, sin, position_ids
            )

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size, self.num_heads, kv_length, head_dim]
            #  - value: [batch_size, self.num_heads, kv_length, head_dim]
            key_layer = ops.cat((past_key, key_layer), axis=-2)
            value_layer = ops.cat((past_value, value_layer), axis=-2)

        kv_length = key_layer.shape[-2]
        present = (key_layer, value_layer) if use_cache else None

        if alibi is None:
            if hasattr(F, "_scaled_dot_product_attention") and not output_attentions:
                attn_output, attention_scores = F._scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask,
                    0.0,
                    is_causal=False,
                    is_training=self.training,
                )
            else:
                attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = ops.softmax(
                    attention_scores + attention_mask,
                    axis=-1,
                    dtype=hidden_states.dtype,
                )
                attn_output = ops.matmul(attention_scores, value_layer)

            attn_output = attn_output.view(
                batch_size, self.num_heads, query_length, self.head_dim
            )
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(
                batch_size, query_length, self.num_heads * self.head_dim
            )

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, present, attention_scores
            return output_tensor, present

        matmul_result = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(
            batch_size, self.num_heads, query_length, kv_length
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype in [
            mindspore.float16,
            mindspore.bfloat16,
        ]:  # from bfloat32 change to float32
            attention_scores = attention_scores.astype(mindspore.float32)
        # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
        # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
        # equivalent and more performant, but there might be a numerical difference. If you're reading this
        # and you'd like to experiment and maybe file a PR, feel free!
        attention_logits = attention_scores + alibi.view(
            batch_size, self.num_heads, 1, -1
        )
        attention_logits *= self.inv_norm_factor
        attention_probs = ops.softmax(
            attention_logits + attention_mask, axis=-1, dtype=hidden_states.dtype
        )
        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size, num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size, self.num_heads, query_length, kv_length
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = ops.matmul(attention_probs_reshaped, value_layer)
        context_layer = context_layer.flatten(start_dim=0, end_dim=1)
        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        if output_attentions:
            return output_tensor, present, attention_probs
        return output_tensor, present


class FalconMLP(nn.Cell):
    """
    FalconMLP is a multi-layer perceptron (MLP) module for the Falcon model.

    Args:
        config (FalconConfig): The configuration for the Falcon model.

    Returns:
        Tensor: The output tensor after applying the MLP transformation."""

    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = FalconLinear(
            hidden_size, 4 * hidden_size, has_bias=config.bias
        )
        self.act = nn.GELU(approximate=False)
        self.dense_4h_to_h = FalconLinear(
            4 * hidden_size, hidden_size, has_bias=config.bias
        )
        self.hidden_dropout = config.hidden_dropout

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class FalconDecoderLayer(nn.Cell):
    """
    FalconDecoderLayer is a class that represents a single layer of the Falcon decoder model.

    Args:
        config (FalconConfig): The configuration for the Falcon model.

    Attributes:
        num_heads (int): The number of attention heads in the self-attention mechanism.
        self_attention (FalconAttention): The self-attention module.
        mlp (FalconMLP): The MLP module.
        hidden_dropout (float): The dropout rate for the hidden states.
        config (FalconConfig): The configuration for the Falcon model.
        ln_attn (nn.LayerNorm): The layer normalization module before self-attention (only used in new decoder architecture).
        ln_mlp (nn.LayerNorm): The layer normalization module before the MLP (only used in new decoder architecture).
        input_layernorm (nn.LayerNorm): The layer normalization module before the self-attention (only used in old decoder architecture).
        post_attention_layernorm (nn.LayerNorm): The layer normalization module after the self-attention (only used in old decoder architecture).

    Methods:
        construct: Forward pass of the FalconDecoderLayer.
    """

    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.self_attention = FalconAttention(config)
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.new_decoder_architecture:
            # The layer norm before self-attention
            self.ln_attn = nn.LayerNorm(
                [hidden_size], epsilon=config.layer_norm_epsilon
            )
            # The layer norm before the MLP
            self.ln_mlp = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)
        else:
            self.input_layernorm = nn.LayerNorm(
                [hidden_size], epsilon=config.layer_norm_epsilon
            )
            if not config.parallel_attn:
                self.post_attention_layernorm = nn.LayerNorm(
                    [hidden_size], epsilon=config.layer_norm_epsilon
                )

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        alibi: Optional[mindspore.Tensor],
        attention_mask: mindspore.Tensor,
        position_ids: Optional[mindspore.Tensor] = None,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Forward pass of the FalconDecoderLayer.

        Args:
            hidden_states (mindspore.Tensor): The input hidden states.
            alibi (Optional[mindspore.Tensor]): The alibi tensor.
            attention_mask (mindspore.Tensor): The attention mask tensor.
            position_ids (Optional[mindspore.Tensor]): The position ids tensor.
            layer_past (Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]): The past layer tensor.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor.
            use_cache (bool): Whether to use cache.
            output_attentions (bool): Whether to output attentions.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[mindspore.Tensor]: The output tensor(s).
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output,
                    residual,
                    self.config.attention_dropout,
                    training=self.training,
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(
            mlp_output, residual, self.config.hidden_dropout, training=self.training
        )

        return (output,) + outputs if use_cache else (output,) + outputs[1:]


class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # convert_torch_to_mindspore = torch_to_mindspore
    config_class = FalconConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["FalconDecoderLayer"]
    _supports_flash_attn_2 = False  # change to False

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, (nn.Dense, FalconLinear)):
            # 使用正态分布初始化权重
            cell.weight.set_data(
                initializer(
                    Normal(0.0, self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.has_bias:
                cell.bias.set_data(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(
                0.0, self.config.initializer_range, cell.weight.shape
            )
            if cell.padding_idx:
                weight[cell.padding_idx] = 0.0

            cell.weight.set_data(mindspore.Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))
            cell.weight.set_data(
                initializer("ones", cell.weight.shape, cell.weight.dtype)
            )


class FalconModel(FalconPreTrainedModel):
    """
    FalconModel is a class representing the Falcon model architecture.

    Args:
        config (FalconConfig): The configuration object specifying the model architecture.

    Attributes:
        embed_dim (int): The dimensionality of the word embeddings.
        num_heads (int): The number of attention heads.
        use_alibi (bool): Whether to use alibi tensor.
        word_embeddings (nn.Embedding): The word embedding layer.
        h (nn.CellList): The list of FalconDecoderLayer instances representing the transformer blocks.
        ln_f (nn.LayerNorm): The final layer normalization.
        gradient_checkpointing (bool): Whether to use gradient checkpointing.

    Methods:
        get_input_embeddings(): Returns the word embedding layer.
        set_input_embeddings(new_embeddings: mindspore.Tensor): Sets the word embedding layer with new embeddings.
        construct(...): The forward pass of the FalconModel.

    Returns:
        Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        The output of the forward pass, which includes the last hidden state, past key values,
        hidden states, and self-attention matrices.
    """

    def __init__(self, config: FalconConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.CellList(
            [FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final Layer Norm
        self.ln_f = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: mindspore.Tensor):
        self.word_embeddings = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]
        ] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]

        if self.use_alibi:
            mask = (
                ops.ones(
                    (batch_size, seq_length + past_key_values_length),
                    dtype=mindspore.int64,
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                position_ids = ops.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=mindspore.int64,
                )
                position_ids = position_ids.unsqueeze(0)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class FalconForCausalLM(FalconPreTrainedModel):
    """
    Falcon model for causal language modeling.

    Args:
        config (FalconConfig): The configuration object that defines the model architecture and hyperparameters.

    Attributes:
        transformer (FalconModel): The Falcon model.
        lm_head (nn.Dense): The linear layer for language modeling.

    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.transformer = FalconModel(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: mindspore.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: mindspore.Tensor,
        past_key_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # Note: versions of Falcon with alibi do not use position_ids. It is used with RoPE.
        if (
            not self.transformer.use_alibi
            and attention_mask is not None
            and position_ids is None
        ):
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]
        ] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss = ops.cross_entropy(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length),
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self,
        past: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...],
        beam_idx: mindspore.Tensor,
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """

        return tuple(
            (
                layer_past[0].index_select(0, beam_idx),
                layer_past[1].index_select(0, beam_idx),
            )
            for layer_past in past
        )


class FalconForSequenceClassification(FalconPreTrainedModel):
    """
    Falcon model for sequence classification tasks.

    Args:
        config (FalconConfig): The configuration object that defines the model architecture and hyperparameters.

    """

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = FalconModel(config)
        self.score = nn.Dense(config.hidden_size, config.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]
        ] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is None:
            sequence_lengths = -1
            logger.warning(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        else:
            sequence_lengths = (
                ops.ne(input_ids, self.config.pad_token_id).sum(axis=-1) - 1
            )
        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in [
                    mindspore.int64,
                    mindspore.int32,
                ]:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(pooled_logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class FalconForTokenClassification(FalconPreTrainedModel):
    """
    Falcon model for token classification.

    Args:
        config (FalconConfig): The configuration object of the Falcon model.

    Attributes:
        num_labels (int): The number of labels for token classification.
        transformer (FalconModel): The Falcon model transformer.
        dropout (nn.Dropout): The dropout layer.
        classifier (nn.Dense): The dense layer for classification.

    """

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = FalconModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]
        ] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        """
        Forward pass of the FalconForTokenClassification model.

        Args:
            input_ids (mindspore.Tensor, optional): The input token IDs. Shape: (batch_size, sequence_length).
            past_key_values (Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...], optional): The past key-value pairs for
                the self-attention mechanism. Shape: (batch_size, num_layers, 2, sequence_length, hidden_size).
            attention_mask (mindspore.Tensor, optional): The attention mask to avoid performing attention on padding tokens.
                Shape: (batch_size, sequence_length).
            head_mask (mindspore.Tensor, optional): The head mask to mask specific attention heads. Shape: (batch_size, num_heads).
            inputs_embeds (mindspore.Tensor, optional): The embedded input tokens. Shape: (batch_size, sequence_length, hidden_size).
            labels (mindspore.Tensor, optional): The labels for computing the sequence classification/regression loss.
                Indices should be in [0, ..., config.num_labels - 1]. If config.num_labels == 1, a regression loss is computed
                (Mean-Square loss). If config.num_labels > 1, a classification loss is computed (Cross-Entropy).
                Shape: (batch_size, sequence_length).
            use_cache (bool, optional): Whether to use the cache for the self-attention mechanism.
            output_attentions (bool, optional): Whether to output the attentions weights.
            output_hidden_states (bool, optional): Whether to output the hidden states.
            return_dict (bool, optional): Whether to return a dictionary as the output.

        Returns:
            Union[Tuple[mindspore.Tensor], TokenClassifierOutput]: The model output. If return_dict is False, returns a tuple
            of (logits, hidden_states, attentions). If labels is not None, also returns the loss.

        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            batch_size, seq_length = labels.shape
            loss = ops.cross_entropy(
                logits.view(batch_size * seq_length, self.num_labels),
                labels.view(batch_size * seq_length),
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class FalconForQuestionAnswering(FalconPreTrainedModel):
    """
    Falcon model for question answering tasks.

    Args:
        config (FalconConfig): The configuration object that defines the model architecture and hyperparameters.

    Attributes:
        transformer (FalconModel): The underlying Falcon model.
        qa_outputs (nn.Dense): The dense layer for question answering outputs.

    """

    def __init__(self, config):
        super().__init__(config)
        self.transformer = FalconModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
        Forward pass of the FalconForQuestionAnswering model.

        Args:
            input_ids (mindspore.Tensor, optional): The input token IDs. Shape: (batch_size, sequence_length).
            attention_mask (mindspore.Tensor, optional): The attention mask. Shape: (batch_size, sequence_length).
            head_mask (mindspore.Tensor, optional): The head mask. Shape: (num_heads, sequence_length, sequence_length).
            inputs_embeds (mindspore.Tensor, optional): The embedded inputs. Shape: (batch_size, sequence_length, hidden_size).
            start_positions (mindspore.Tensor, optional): The start positions of the labeled span. Shape: (batch_size,).
            end_positions (mindspore.Tensor, optional): The end positions of the labeled span. Shape: (batch_size,).
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.
            return_dict (bool, optional): Whether to return a dictionary as the output. Default: None.

        Returns:
            Union[Tuple, QuestionAnsweringModelOutput]: The model output, which includes the start logits, end logits,
            hidden states, and attentions.

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.cross_entropy(
                start_logits, start_positions, ignore_index=ignored_index
            )
            end_loss = ops.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "FALCON_PRETRAINED_MODEL_ARCHIVE_LIST",
    "FalconForCausalLM",
    "FalconModel",
    "FalconPreTrainedModel",
    "FalconForSequenceClassification",
    "FalconForTokenClassification",
    "FalconForQuestionAnswering",
]
