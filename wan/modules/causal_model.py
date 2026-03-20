from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d,
)
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import (
    create_block_mask as create_block_mask_uncompiled,
)
from torch.nn.attention.flex_attention import (
    flex_attention as flex_attention_uncompiled,
)
from torch.nn.attention.flex_attention import and_masks
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from typing import Any, Callable, Optional, Union

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
# if __name__ == "__main__":

_COMPILED_FLEX_ATTENTION = None
MAX_SEQ_LEN = 64000


def flex_attention(*args, **kwargs):

    global _COMPILED_FLEX_ATTENTION

    # 首次调用时进行编译
    if _COMPILED_FLEX_ATTENTION is None:
        print(
            f"Compiling flex_attention for the first time in process {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}..."
        )
        # 编译的是我们导入的原始函数
        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_attention_uncompiled, dynamic=True
        )  # mode="max-autotune-no-cudagraphs")
        print("Compilation finished.")

    # 将所有参数原封不动地传递给编译后的函数
    return _COMPILED_FLEX_ATTENTION(*args, **kwargs)


_COMPILED_BLOCK_MASK = None


def create_block_mask(*args, **kwargs):

    global _COMPILED_BLOCK_MASK

    # 首次调用时进行编译
    if _COMPILED_BLOCK_MASK is None:
        print(
            f"Compiling block_mask for the first time in process {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}..."
        )
        # 编译的是我们导入的原始函数
        _COMPILED_BLOCK_MASK = torch.compile(create_block_mask_uncompiled, dynamic=True)
        print("Compilation finished.")

    # 将所有参数原封不动地传递给编译后的函数
    return _COMPILED_BLOCK_MASK(*args, **kwargs)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def mf_rope_apply(x, grid_sizes, freqs, start_index=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        x_len = x.size(1)
        f = x_len // h // w
        # precompute multipliers
        x_i = torch.view_as_complex(x[i].to(torch.float64).reshape(x_len, n, -1, 2))
        freqs_f = freqs[0][start_index:start_index+f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_i = torch.cat(
            [
                freqs_f,  # freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(x_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        is_inference_mode=False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = (
            327600  # if local_attn_size == -1 else local_attn_size * 1560
        )
        self.is_inference_mode = is_inference_mode
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        start_index=0,
        update_cache=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        # self.is_inference_mode 为false时，cache start一定是None
        assert (
            self.is_inference_mode or cache_start is None
        ), "When is_inference_mode is False, cache_start must be None."
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = False  # (s == seq_lens[0].item() * 2)
            is_mf = s > seq_lens[0].item()
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
            elif is_mf:

                noisy_seq_len = seq_lens[0].item()
                clean_seq_len = s - noisy_seq_len
                q_chunk = torch.split(q, [clean_seq_len, noisy_seq_len], dim=1)
                k_chunk = torch.split(k, [clean_seq_len, noisy_seq_len], dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                # for ii in range(2):
                c_rq = mf_rope_apply(q_chunk[0], grid_sizes, freqs,start_index).type_as(v)
                c_rk = mf_rope_apply(k_chunk[0], grid_sizes, freqs,start_index).type_as(v)
                roped_query.append(c_rq)
                roped_key.append(c_rk)

                n_rq = mf_rope_apply(q_chunk[1], grid_sizes, freqs,start_index).type_as(v)
                n_rk = mf_rope_apply(k_chunk[1], grid_sizes, freqs,start_index).type_as(v)
                roped_query.append(n_rq)
                roped_key.append(n_rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs, start_index).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs, start_index).type_as(v)

            padded_length = 0
            padded_roped_query = torch.cat(
                [
                    roped_query,
                    torch.zeros(
                        [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                        device=q.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            ).transpose(2, 1)

            padded_roped_key = torch.cat(
                [
                    roped_key,
                    torch.zeros(
                        [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                        device=k.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            ).transpose(2, 1)

            padded_v = torch.cat(
                [
                    v,
                    torch.zeros(
                        [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                        device=v.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            ).transpose(2, 1)
            x = flex_attention(
                query=padded_roped_query,
                key=padded_roped_key,
                value=padded_v,
                block_mask=block_mask,
            )
            # assert isinstance(x, torch.Tensor)
            x = x[:, :, : -padded_length if padded_length > 0 else None].transpose(2, 1)
            # x = x.transpose(2, 1)
        else:

            # =====================================================================
            # --- BRANCH 1: INFERENCE MODE ---
            # `kv_cache` 尺寸较小，执行高效的动态滑窗更新。
            # =====================================================================
            if self.is_inference_mode:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)

                current_end = cache_start + roped_query.shape[1]
                sink_tokens = self.sink_size * frame_seqlen
                # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
                kv_cache_size = kv_cache["k"].shape[1]
                num_new_tokens = roped_query.shape[1]
                if (
                    self.local_attn_size != -1
                    and (current_end > kv_cache["global_end_index"].item())
                    and (
                        num_new_tokens + kv_cache["local_end_index"].item()
                        > kv_cache_size
                    )
                ):
                    # Calculate the number of new tokens added in this step
                    # Shift existing cache content left to discard oldest tokens
                    # Clone the source slice to avoid overlapping memory error
                    num_evicted_tokens = (
                        num_new_tokens
                        + kv_cache["local_end_index"].item()
                        - kv_cache_size
                    )
                    num_rolled_tokens = (
                        kv_cache["local_end_index"].item()
                        - num_evicted_tokens
                        - sink_tokens
                    )
                    kv_cache["k"][
                        :, sink_tokens : sink_tokens + num_rolled_tokens
                    ] = kv_cache["k"][
                        :,
                        sink_tokens
                        + num_evicted_tokens : sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()
                    kv_cache["v"][
                        :, sink_tokens : sink_tokens + num_rolled_tokens
                    ] = kv_cache["v"][
                        :,
                        sink_tokens
                        + num_evicted_tokens : sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()
                    # Insert the new keys/values at the end
                    local_end_index = (
                        kv_cache["local_end_index"].item()
                        + current_end
                        - kv_cache["global_end_index"].item()
                        - num_evicted_tokens
                    )
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                else:
                    # Assign new keys/values directly up to current_end
                    local_end_index = (
                        kv_cache["local_end_index"].item()
                        + current_end
                        - kv_cache["global_end_index"].item()
                    )  # add num new token
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][
                        :, local_start_index:local_end_index
                    ] = roped_key  # baocuo
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                x = attention(
                    roped_query,
                    kv_cache["k"][
                        :,
                        max(
                            0, local_end_index - self.max_attention_size
                        ) : local_end_index,
                    ],
                    kv_cache["v"][
                        :,
                        max(
                            0, local_end_index - self.max_attention_size
                        ) : local_end_index,
                    ],
                )
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

            # =====================================================================
            # --- BRANCH 2: TRAINING MODE ---
            # `kv_cache` 尺寸很大，执行适用于梯度检查点的静态写入和动态构建。
            # =====================================================================
            else:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
                current_start_frame = (
                    current_start // frame_seqlen
                )  # chunk为第一个chunk为21帧，后续都为18帧

                # 1. 对当前的 q 和 k 应用旋转位置编码 (RoPE)
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)

                # 根据当前帧计算chunk索引
                chunk_index = (
                    0
                    if current_start_frame < 21
                    else 1 + (current_start_frame - 21) // 18
                )

                local_cache_start_frame = (
                    current_start_frame
                    if chunk_index == 0
                    else current_start_frame - 21 - 18 * (chunk_index - 1)
                )
                local_cache_start = local_cache_start_frame * frame_seqlen
                num_new_tokens = roped_query.shape[1]
                local_cache_start_end = local_cache_start + num_new_tokens

                # a. 【写操作】如果需要，更新大的KV缓存
                if update_cache:
                    kv_cache["k"][
                        :, local_cache_start:local_cache_start_end
                    ] = roped_key
                    kv_cache["v"][:, local_cache_start:local_cache_start_end] = v

                    if (
                        current_start_frame == 0
                    ):  # 记录sink token 前3帧，但不一定用这么多
                        kv_cache["sink_k"][
                            :, : self.sink_size * frame_seqlen
                        ] = roped_key[:, : self.sink_size * frame_seqlen]
                        kv_cache["sink_v"][:, : self.sink_size * frame_seqlen] = v[:, : self.sink_size * frame_seqlen]
                    kv_cache["global_end_index"].fill_(current_start + num_new_tokens)
                    kv_cache["local_end_index"].fill_(local_cache_start_end)

                # b. 【读操作】构建本次注意力计算所需的K和V
                sink_len = min(current_start, self.sink_size * frame_seqlen)
                sink_k = kv_cache["sink_k"][:, :sink_len]
                sink_v = kv_cache["sink_v"][:, :sink_len]

                # if self.local_attn_size != -1:
                local_history_len = min(
                    self.local_attn_size * frame_seqlen, current_start - sink_len
                )

                history_k, history_v = self.read_history_token(
                    kv_cache, local_cache_start, local_history_len, chunk_index
                )

                current_k = roped_key
                current_v = v

                keys_to_attend = torch.cat([sink_k, history_k, current_k], dim=1)
                values_to_attend = torch.cat([sink_v, history_v, current_v], dim=1)

                # c. 执行注意力计算
                x = attention(roped_query, keys_to_attend, values_to_attend)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def read_history_token(
        self, kv_cache, local_cache_start, local_history_len, chunk_index
    ):
        # 对于chunk index = 0 直接在主cache里取local history len
        if chunk_index == 0:
            history_k = kv_cache["k"][
                :, local_cache_start - local_history_len : local_cache_start
            ]
            history_v = kv_cache["v"][
                :, local_cache_start - local_history_len : local_cache_start
            ]
        else:  # 全局注意力
            window_cache_require_len = max(local_history_len - local_cache_start, 0)
            window_cache_k = (
                kv_cache["window_k"][:, -window_cache_require_len:]
                if window_cache_require_len > 0
                else kv_cache["window_k"][:, :0]
            )
            window_cache_v = (
                kv_cache["window_v"][:, -window_cache_require_len:]
                if window_cache_require_len > 0
                else kv_cache["window_v"][:, :0]
            )

            local_history_len = local_history_len - window_cache_require_len

            history_k = kv_cache["k"][
                :, local_cache_start - local_history_len : local_cache_start
            ]
            history_v = kv_cache["v"][
                :, local_cache_start - local_history_len : local_cache_start
            ]

            history_k = torch.cat([window_cache_k, history_k], dim=1)
            history_v = torch.cat([window_cache_v, history_v], dim=1)

        return history_k, history_v


class CausalWanSelfAttention_block_cache(CausalWanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        is_inference_mode=False,
        cache_mode="block_compress",
    ):
        super().__init__(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, is_inference_mode
        )
        self.cache_mode = cache_mode

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        start_index=0,
        update_cache=False,
    ):
        if kv_cache is not None:
            return self.forward_cache(
                x,
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                start_index,
                update_cache,
            )
        else:
            return self.forward_no_cache(
                x,
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                start_index,
                update_cache,
            )

    def forward_no_cache(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        start_index=0,
        update_cache=False,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        # self.is_inference_mode 为false时，cache start一定是None
        assert (
            self.is_inference_mode or cache_start is None
        ), "When is_inference_mode is False, cache_start must be None."
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        is_tf = False  # (s == seq_lens[0].item() * 2)
        is_mf = s > seq_lens[0].item()
        if is_tf:
            q_chunk = torch.chunk(q, 2, dim=1)
            k_chunk = torch.chunk(k, 2, dim=1)
            roped_query = []
            roped_key = []
            # rope should be same for clean and noisy parts
            for ii in range(2):
                rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                roped_query.append(rq)
                roped_key.append(rk)

            roped_query = torch.cat(roped_query, dim=1)
            roped_key = torch.cat(roped_key, dim=1)
        elif is_mf:

            noisy_seq_len = seq_lens[0].item()
            clean_seq_len = s - noisy_seq_len
            q_chunk = torch.split(q, [clean_seq_len, noisy_seq_len], dim=1)
            k_chunk = torch.split(k, [clean_seq_len, noisy_seq_len], dim=1)
            roped_query = []
            roped_key = []
            # rope should be same for clean and noisy parts
            # for ii in range(2):
            c_rq = mf_rope_apply(q_chunk[0], grid_sizes, freqs,start_index).type_as(v)
            c_rk = mf_rope_apply(k_chunk[0], grid_sizes, freqs,start_index).type_as(v)
            roped_query.append(c_rq)
            roped_key.append(c_rk)

            n_rq = mf_rope_apply(q_chunk[1], grid_sizes, freqs,start_index).type_as(v)
            n_rk = mf_rope_apply(k_chunk[1], grid_sizes, freqs,start_index).type_as(v)
            roped_query.append(n_rq)
            roped_key.append(n_rk)

            roped_query = torch.cat(roped_query, dim=1)
            roped_key = torch.cat(roped_key, dim=1)

        else:
            roped_query = rope_apply(q, grid_sizes, freqs, start_index).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs, start_index).type_as(v)

        padded_length = 0
        padded_roped_query = torch.cat(
            [
                roped_query,
                torch.zeros(
                    [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                    device=q.device,
                    dtype=v.dtype,
                ),
            ],
            dim=1,
        ).transpose(2, 1)

        padded_roped_key = torch.cat(
            [
                roped_key,
                torch.zeros(
                    [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                    device=k.device,
                    dtype=v.dtype,
                ),
            ],
            dim=1,
        ).transpose(2, 1)

        padded_v = torch.cat(
            [
                v,
                torch.zeros(
                    [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                    device=v.device,
                    dtype=v.dtype,
                ),
            ],
            dim=1,
        ).transpose(2, 1)
        x = flex_attention(
            query=padded_roped_query,
            key=padded_roped_key,
            value=padded_v,
            block_mask=block_mask,
        )
        # assert isinstance(x, torch.Tensor)
        x = x[:, :, : -padded_length if padded_length > 0 else None].transpose(2, 1)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def forward_cache(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        start_index=0,
        update_cache=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        # self.is_inference_mode 为false时，cache start一定是None
        assert (
            self.is_inference_mode or cache_start is None
        ), "When is_inference_mode is False, cache_start must be None."
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        assert (
            kv_cache is not None
        ), "kv_cache must be provided for block_cache attention."

        if self.is_inference_mode:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = (
                current_start // frame_seqlen
            )  # chunk为第一个chunk为21帧，后续都为18帧

            # 1. 对当前的 q 和 k 应用旋转位置编码 (RoPE)
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)

            # 根据当前帧计算chunk索引
            num_new_tokens = roped_query.shape[1]
            num_frame_of_block = num_new_tokens // frame_seqlen
            chunk_index = (
                0 if current_start_frame < 21 else 1 + (current_start_frame - 21) // 18
            )
            block_index = current_start_frame // num_frame_of_block
            # local_cache_start = current_start_frame * frame_seqlen
            # local_cache_start_end = local_cache_start + num_new_tokens  # 每次就只保留3帧
            local_cache_start = block_index * frame_seqlen
            local_cache_start_end = local_cache_start + frame_seqlen  # 每次就只保留一帧

            sink_len = min(current_start, self.sink_size * frame_seqlen)
            sink_k = kv_cache["sink_k"][:, :sink_len]
            sink_v = kv_cache["sink_v"][:, :sink_len]
            history_k = kv_cache["k"][:, :local_cache_start]
            history_v = kv_cache["v"][:, :local_cache_start]

            current_k = roped_key
            current_v = v

            keys_to_attend = torch.cat([sink_k, history_k, current_k], dim=1)
            values_to_attend = torch.cat([sink_v, history_v, current_v], dim=1)

            if update_cache:
                x = attention(roped_query, keys_to_attend, values_to_attend)

                kv_len = kv_cache["k"].shape[1]
                local_cache_roll_start = local_cache_start % kv_len
                local_cache_roll_end = local_cache_roll_start + frame_seqlen
                kv_cache["k"][:, local_cache_roll_start:local_cache_roll_end] = roped_key[
                    :, 2*frame_seqlen : 3* frame_seqlen
                ]
                kv_cache["v"][:, local_cache_roll_start:local_cache_roll_end] = v[
                    :, 2*frame_seqlen : 3* frame_seqlen
                ]
                kv_cache["global_end_index"].fill_(current_start + num_new_tokens) 
                kv_cache["local_end_index"].fill_(local_cache_roll_end)
                if block_index == 0 and self.sink_size != 0:
                    kv_cache["sink_k"][:, : self.sink_size * frame_seqlen] = roped_key[
                        :, : self.sink_size * frame_seqlen
                    ]
                    kv_cache["sink_v"][:, : self.sink_size * frame_seqlen] = v[
                        :, : self.sink_size * frame_seqlen
                    ]
            else:
                x = attention(roped_query, keys_to_attend, values_to_attend)
                

        # =====================================================================
        # --- BRANCH 2: TRAINING MODE ---
        # `kv_cache` 尺寸很大，执行适用于梯度检查点的静态写入和动态构建。
        # =====================================================================
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = (
                current_start // frame_seqlen
            )  # chunk为第一个chunk为21帧，后续都为18帧

            # 1. 对当前的 q 和 k 应用旋转位置编码 (RoPE)
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)

            # 根据当前帧计算chunk索引
            num_new_tokens = roped_query.shape[1]
            num_frame_of_block = num_new_tokens // frame_seqlen
            chunk_index = (
                0 if current_start_frame < 21 else 1 + (current_start_frame - 21) // 18
            )
            block_index = current_start_frame // num_frame_of_block
            local_cache_start = block_index * frame_seqlen
            local_cache_start_end = local_cache_start + frame_seqlen  # 每次就只保留一帧

            # a. 【写操作】如果需要，更新大的KV缓存
            if update_cache:
                kv_cache["k"][:, local_cache_start:local_cache_start_end] = roped_key[
                    :, 2 * frame_seqlen : 3 * frame_seqlen
                ]
                kv_cache["v"][:, local_cache_start:local_cache_start_end] = v[
                    :, 2 * frame_seqlen : 3 * frame_seqlen
                ]
                kv_cache["global_end_index"].fill_(current_start + num_new_tokens) 
                kv_cache["local_end_index"].fill_(local_cache_start_end)
                if block_index == 0 and self.sink_size != 0:
                    kv_cache["sink_k"][:, : self.sink_size * frame_seqlen] = roped_key[
                        :, : self.sink_size * frame_seqlen
                    ]
                    kv_cache["sink_v"][:, : self.sink_size * frame_seqlen] = v[
                        :, : self.sink_size * frame_seqlen
                    ]

            sink_len = min(current_start, self.sink_size * frame_seqlen)
            sink_k = kv_cache["sink_k"][:, :sink_len]
            sink_v = kv_cache["sink_v"][:, :sink_len]
            history_k = kv_cache["k"][:, :local_cache_start]
            history_v = kv_cache["v"][:, :local_cache_start]

            current_k = roped_key
            current_v = v

            keys_to_attend = torch.cat([sink_k, history_k, current_k], dim=1)
            values_to_attend = torch.cat([sink_v, history_v, current_v], dim=1)

            # c. 执行注意力计算
            x = attention(roped_query, keys_to_attend, values_to_attend)

        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        is_inference_mode=False,
        cache_mode='block_compress'
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        if cache_mode == 'block_compress':
            self.self_attn = CausalWanSelfAttention_block_cache(
                dim, num_heads, local_attn_size, sink_size, qk_norm, eps, is_inference_mode,cache_mode
            )
        else:
            self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size,qk_norm, eps,is_inference_mode)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        start_index=None,
        update_cache=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            start_index,
            update_cache,
        )

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(
            1, 2
        )

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            y = self.ffn(
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * (1 + e[4])
                    + e[3]
                ).flatten(1, 2)
            )
            x = x + (
                y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]
            ).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        # torch.cuda.empty_cache()
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1])
            + e[0]
        )
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        num_frame_per_block=3,
        is_inference_mode=False,
        cache_mode='block_compress'
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.cache_mode = cache_mode

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    is_inference_mode,
                    cache_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask_cache_df = {}
        self.block_mask_cache_tf = {}
        self.block_mask_cache = {}

        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = False
        self.is_inference_mode = is_inference_mode

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def generate_block_compress_mask(
        self,
        num_frames: int,
        frame_seqlen: int,
        device: torch.device | str,
        num_frame_per_block: int = 1,
        local_attn_size: int = 1,
    ):
        """
        为视频帧生成 "memory forcing" 掩码，旨在增强模型的记忆和重建能力。
        优化版本：利用block结构，最小化内存占用。
        """
        # 1. 计算核心参数
        # local_attn_in_blocks = math.ceil(local_attn_size / num_frame_per_block)

        clean_len = num_frames * frame_seqlen
        noisy_len = clean_len
        total_len = clean_len + noisy_len

        block_size_in_tokens = num_frame_per_block * frame_seqlen
        clean_blocks = math.ceil(clean_len / block_size_in_tokens)
        noisy_blocks = math.ceil(noisy_len / block_size_in_tokens)

        if block_size_in_tokens == 0:
            raise ValueError("`num_frame_per_block` 和 `frame_seqlen` 必须为正数。")
        if noisy_len <= 0:
            raise ValueError(
                f"noisy_len{noisy_len}小于等于零, local_attn_size:{local_attn_size},clean_len:{clean_len}"
            )

        # 3. 创建用于noisy-clean区域的start和end向量（1D张量）
        noise_clean_start = torch.zeros(total_len, device=device, dtype=torch.long)
        noise_clean_end = torch.zeros(total_len, device=device, dtype=torch.long)

        cache_frame_idx = num_frame_per_block - 1 # num_frame_per_block // 2
        for i in range(noisy_blocks):

            block_start = (i + clean_blocks) * block_size_in_tokens
            block_end = block_start + block_size_in_tokens

            cache_frame_start = i * block_size_in_tokens + cache_frame_idx * frame_seqlen
            cache_frame_end = cache_frame_start + frame_seqlen
            noise_clean_start[block_start:block_end] = cache_frame_start
            noise_clean_end[block_start:block_end] = cache_frame_end

        # --- 定义优化的 attention_mask 函数 ---
        def attention_mask(b, h, q_idx, kv_idx):
            """
            优化的注意力掩码函数：
            - clean-clean: 在内部计算
            - noisy-noisy: 利用对角线结构在内部计算
            - noisy-clean: 使用预先计算的start和end向量
            """
            # --- 区域判断 ---
            is_q_clean = q_idx < clean_len
            is_q_noisy = ~is_q_clean
            is_kv_clean = kv_idx < clean_len
            is_kv_noisy = ~is_kv_clean

            # --- 索引计算 ---
            q_block_idx = q_idx // block_size_in_tokens
            kv_block_idx = kv_idx // block_size_in_tokens

            # --- 区域 1: Clean Query -> Clean KV (clean-clean) ---
            # 块状因果自注意力
            mask_cc = is_q_clean & is_kv_clean & (kv_block_idx == q_block_idx)
            mask_cc = mask_cc
            # --- 区域 2: Clean Query -> Noisy KV (clean-noise) ---
            mask_cn = torch.zeros_like(q_idx, dtype=torch.bool)

            # --- 区域 3: Noisy Query -> Clean KV (noise-clean) ---
            # 使用预先计算的start和end向量
            mask_nc = (
                is_q_noisy
                & is_kv_clean
                & (kv_idx >= noise_clean_start[q_idx])
                & (kv_idx < noise_clean_end[q_idx])
            )

            # --- 区域 4: Noisy Query -> Noisy KV (noise-noise) ---
            # 利用对角线结构：同一block内的双向注意力
            mask_nn = is_q_noisy & is_kv_noisy & (q_block_idx == kv_block_idx)

            # --- 组合所有区域 ---
            final_mask = mask_cc | mask_cn | mask_nc | mask_nn
            return final_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_len,
            KV_LEN=total_len,
            _compile=False,
            device=device,
        )

        import imageio
        import numpy as np
        from torch.nn.attention.flex_attention import create_mask

        create_mask = torch.compile(create_mask)
        mask = create_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_len,
            KV_LEN=total_len,
            device=device,
        )
        import cv2

        mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        imageio.imwrite("compress_mask_%d.jpg" % (0), np.uint8(255.0 * mask))
        return block_mask

    def generate_all_history_block_compress_mask(
        self,
        num_frames: int,
        frame_seqlen: int,
        device: torch.device | str,
        num_frame_per_block: int = 1,
        local_attn_size: int = 1,
    ):
        """
        为视频帧生成 "memory forcing" 掩码，旨在增强模型的记忆和重建能力。
        清晰版本：使用条纹状缓存帧掩码，每个block缓存一帧，后续block可见前面所有缓存帧。
        """
        # 1. 计算核心参数
        clean_len = num_frames * frame_seqlen
        noisy_len = clean_len 
        total_len = clean_len + noisy_len
        
        block_size_in_tokens = num_frame_per_block * frame_seqlen
        clean_blocks = math.ceil(clean_len / block_size_in_tokens)
        noisy_blocks = math.ceil(noisy_len / block_size_in_tokens)

        if block_size_in_tokens == 0:
            raise ValueError("`num_frame_per_block` 和 `frame_seqlen` 必须为正数。")
        if noisy_len <= 0:
            raise ValueError(f"noisy_len{noisy_len}小于等于零, local_attn_size:{local_attn_size},clean_len:{clean_len}")

        # 2. 创建缓存帧标记张量（条纹状掩码）
        # is_cached_frame[i] = True 表示第i个token是某个block的缓存中间帧
        is_cached_frame = torch.zeros(total_len, device=device, dtype=torch.bool)
        
        # 为每个clean block标记缓存帧（中间帧）
        for block_idx in range(clean_blocks):
            block_start = block_idx * block_size_in_tokens
            # 计算当前block的中间帧位置
            if num_frame_per_block == 1:
                # 每个block只有一帧，缓存这一帧
                cache_frame_start = block_start
            else:
                # 每个block有多帧，缓存中间那一帧
                cache_frame_idx = num_frame_per_block - 1 # num_frame_per_block // 2
                cache_frame_start = block_start + cache_frame_idx * frame_seqlen
            
            cache_frame_end = cache_frame_start + frame_seqlen
            is_cached_frame[cache_frame_start:cache_frame_end] = True

        # --- 定义优化的 attention_mask 函数 ---
        def attention_mask(b, h, q_idx, kv_idx):
            """
            清晰的注意力掩码函数：
            - clean-clean: 当前block内所有token + 前面所有block的缓存帧
            - noisy-noisy: 当前block内所有token
            - noisy-clean: 前面所有block的缓存帧（block index < 当前noisy block index）
            - clean-noisy: 不允许
            """
            # 获取query和kv的block索引
            # q_block = token_block_idx[q_idx]
            # kv_block = token_block_idx[kv_idx]
            q_block = q_idx // block_size_in_tokens
            kv_block = kv_idx // block_size_in_tokens
            # `q_noisy_equiv_block_idx` 指的是 noisy query 对应的 clean block 索引
            # 这是 memory task 的核心，决定了 "未来" 窗口的相对位置
            # 区域判断
            is_q_clean = q_idx < clean_len
            is_q_noisy = ~is_q_clean
            is_kv_clean = kv_idx < clean_len
            is_kv_noisy = ~is_kv_clean

            q_block = torch.where(
                is_q_clean,
                q_block,
                (q_idx - clean_len) // block_size_in_tokens
            )
            kv_block = torch.where(
                is_kv_clean,
                kv_block,
                (kv_idx - clean_len) // block_size_in_tokens
            )
            # 1. Clean Query -> Clean KV (clean-clean)
            # - 当前block内的所有token
            # - 前面所有block的缓存帧
            same_block = (q_block == kv_block)
            prev_block_cache = (kv_block < q_block) & is_cached_frame[kv_idx]
            mask_cc = is_q_clean & is_kv_clean & (same_block | prev_block_cache)

            # 2. Clean Query -> Noisy KV (clean-noise)
            mask_cn = torch.zeros_like(q_idx, dtype=torch.bool)

            # 3. Noisy Query -> Clean KV (noise-clean)
            # - 前面所有block的缓存帧（block index < 当前noisy block index）
            mask_nc = is_q_noisy & is_kv_clean & (kv_block <= q_block) & is_cached_frame[kv_idx]

            # 4. Noisy Query -> Noisy KV (noise-noise)
            # - 当前block内的所有token
            mask_nn = is_q_noisy & is_kv_noisy & (q_block == kv_block)

            # 组合所有区域
            final_mask = mask_cc | mask_cn | mask_nc | mask_nn
            return final_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_len,
            KV_LEN=total_len,
            _compile=False,
            device=device,
        )

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask
        # create_mask = torch.compile(create_mask)
        # mask = create_mask(
        #     attention_mask,
        #     B=None,
        #     H=None,
        #     Q_LEN=total_len,
        #     KV_LEN=total_len,
        #     device=device,
        # )
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("compress_mask_%d.jpg" % (0), np.uint8(255.0 * mask))
        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start=None,
        update_cache=False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=None,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "update_cache": update_cache,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "update_cache": update_cache,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        start_index=0,
        block_mask_type="mf",
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if clean_x is not None:  # 临时将tf改成mf先
            num_clean_frames = clean_x.shape[2]
        if block_mask_type == "compress":
            if x.shape[2] in self.block_mask_cache:
                block_mask = self.block_mask_cache[x.shape[2]]
            else:
                block_mask = self.generate_block_compress_mask(
                    num_frames=clean_x.shape[2],
                    frame_seqlen=x.shape[-2]* x.shape[-1]// (self.patch_size[1] * self.patch_size[2]),
                    device=device,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )
                self.block_mask_cache[x.shape[2]] = block_mask
        elif block_mask_type == "history_compress":
            if x.shape[2] in self.block_mask_cache:
                block_mask = self.block_mask_cache[x.shape[2]]
            else:
                block_mask = self.generate_all_history_block_compress_mask(
                    num_frames=clean_x.shape[2],
                    frame_seqlen=x.shape[-2]* x.shape[-1]// (self.patch_size[1] * self.patch_size[2]),
                    device=device,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )
                self.block_mask_cache[x.shape[2]] = block_mask

        # num_frames = x[0].shape[1]
        # frame_seqlen = x[0].shape[2] // self.patch_size[1] * x[0].shape[3] // self.patch_size[2]
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(
            [
                torch.cat(
                    [u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1
                )
                for u in x
            ]
        )

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )  # [2, 39, 6, 1536] # 39其实是外部补齐的长度，实际视频不一定是这个
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor(
                [u.size(1) for u in clean_x], dtype=torch.long
            )
            # assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(
                [
                    torch.cat(
                        [u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))],
                        dim=1,
                    )
                    for u in clean_x
                ]
            )

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                t_shape = list(t.shape)
                t_shape[1] = num_clean_frames
                aug_t = torch.zeros(t_shape, device=t.device)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e0_clean = (
                self.time_projection(e_clean)
                .unflatten(1, (6, self.dim))
                .unflatten(dim=0, sizes=aug_t.shape)
            )
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=block_mask,
            start_index=start_index,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, clean_x.shape[1] :]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)  # 不定长序列只能返回list
        # return x

    def forward(
        self,
        *args,
        # kv_cache=None,
        # crossattn_cache=None,
        **kwargs,
    ):

        if kwargs.get("kv_cache", None) is not None:
            # if kv_cache is not None:
            # kwargs['kv_cache'] = kv_cache
            # kwargs['crossattn_cache'] = crossattn_cache
            return self._forward_inference(*args, **kwargs)
        else:
            # kwargs['kv_cache'] = None
            # kwargs['crossattn_cache'] = crossattn_cache
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # u = u[:v[0]].view(*v, *self.patch_size, c)
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
