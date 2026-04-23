# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

logger = logging.getLogger(__name__)

try:
    from transformers.integrations.flash_attention import get_target_dtype
    from transformers.modeling_flash_attention_utils import (
        _flash_attention_forward,
        flash_attn_supports_top_left_mask,
    )
except Exception:
    get_target_dtype = None
    _flash_attention_forward = None
    flash_attn_supports_top_left_mask = None


@lru_cache(maxsize=1)
def _load_flash_attn_with_kvcache():
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except Exception as exc:
        logger.debug("flash_attn_with_kvcache is unavailable: %s", exc)
        return None
    return flash_attn_with_kvcache


def _is_transformers_flash_attention_forward(func) -> bool:
    return (
        getattr(func, "__module__", "") == "transformers.integrations.flash_attention"
        and getattr(func, "__name__", "") == "flash_attention_forward"
    )


def reset_flashdecode_tracking(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_kvpress_flashdecode_used"):
            module._kvpress_flashdecode_used = False


def flashdecode_used_layers(model: torch.nn.Module) -> list[int]:
    used_layers: list[int] = []
    for module in model.modules():
        if getattr(module, "_kvpress_flashdecode_used", False):
            layer_idx = getattr(module, "layer_idx", None)
            if isinstance(layer_idx, int):
                used_layers.append(layer_idx)
    return sorted(set(used_layers))


def _flashdecode_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float | None,
    sliding_window: int | None,
    softcap: float | None,
    is_causal: bool | None,
    s_aux: torch.Tensor | None,
) -> torch.Tensor | None:
    flash_attn_with_kvcache = _load_flash_attn_with_kvcache()
    if flash_attn_with_kvcache is None:
        return None
    if query.shape[2] != 1:
        return None
    if attention_mask is not None or s_aux is not None:
        return None

    q = query.transpose(1, 2).contiguous()
    k_cache = key.transpose(1, 2).contiguous()
    v_cache = value.transpose(1, 2).contiguous()
    cache_seqlens = torch.full(
        (q.shape[0],),
        k_cache.shape[1],
        dtype=torch.int32,
        device=q.device,
    )
    window_size = (-1, -1) if sliding_window is None else (sliding_window - 1, 0)
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        softmax_scale=scaling,
        causal=bool(is_causal),
        window_size=window_size,
        softcap=0.0 if softcap is None else softcap,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]
    if not getattr(module, "_kvpress_flashdecode_logged", False):
        logger.info(
            "Using flash_attn_with_kvcache decode fast path for %s",
            module.__class__.__name__,
        )
        module._kvpress_flashdecode_logged = True
    module._kvpress_flashdecode_used = True
    return attn_output


def _flash_attention_forward_without_forced_sink(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Compatibility shim for Transformers FA wrappers that assume s_aux is always set."""
    if get_target_dtype is None or _flash_attention_forward is None or flash_attn_supports_top_left_mask is None:
        raise RuntimeError("Transformers FlashAttention utilities are unavailable.")

    seq_len = query.shape[2]
    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has a zero dimension. FlashAttention does not support inputs with dim=0."
        )

    max_head_dim = max(query.shape[-1], key.shape[-1], value.shape[-1])
    if max_head_dim > 256:
        raise RuntimeError(
            "FlashAttention-3 requires head_dim <= 256 for this path, "
            f"but received head_dim={max_head_dim} in {module.__class__.__name__}."
        )

    flashdecode_output = _flashdecode_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        is_causal=is_causal,
        s_aux=s_aux,
    )
    if flashdecode_output is not None:
        return flashdecode_output, None

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    target_dtype = get_target_dtype(query, module)
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=flash_attn_supports_top_left_mask(),
        target_dtype=target_dtype,
        attn_implementation=module.config._attn_implementation,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        s_aux=s_aux.to(query.dtype) if s_aux is not None else None,
        **kwargs,
    )

    return attn_output, None


def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for a hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns - 1e5 * Y / ||Y|| ** 2 to ensure exp(<X, Y>) = 0
    Raises a ValueError if no such hyperplane is found

    Parameters
    ----------
    X : torch.Tensor
        Query tensor with shape (batch_size, seq_len, head_dim) representing
        the query vectors for which we want to find a nullifying hyperplane.
    max_iter : int, default=1000
        Maximum number of iterations to search for the hyperplane. If no valid
        hyperplane is found within this limit, a ValueError is raised.

    Returns
    -------
    torch.Tensor
        Hyperplane tensor with shape (batch_size, head_dim) scaled by -1e5 / ||Y||²
        to ensure that exp(<X, Y>) ≈ 0 for all queries in X.

    Raises
    ------
    ValueError
        If no valid hyperplane is found within max_iter iterations.
    """
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def attention_patch(func):
    """
    Decorator to update the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k such that exp(<q, k>) = 0 to fake head-wise compression
    This solution is not optimal as it does not reduce peak memory and slightly increases runtime

    Parameters
    ----------
    func : callable
        The original attention function to be patched. Should accept parameters
        (module, query, key, value, attention_mask, dropout, **kwargs).

    Returns
    -------
    callable
        The wrapped attention function that supports head-wise key masking.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.masked_key_indices = None
        elif getattr(module, "masked_key_indices", None) is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        # see https://github.com/NVIDIA/kvpress/pull/115#issuecomment-3183785597
        # cu_seq_lens_k are only in kwargs if model.generate is used.
        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Apply attention patching to all transformer attention functions.

    This function automatically patches all attention functions registered in
    transformers' ALL_ATTENTION_FUNCTIONS to support head-wise key masking.
    It enables KVPress compression methods that require head-specific masking
    (like AdaKV) to work correctly during text generation.

    The patching is applied globally and affects all transformer models loaded
    after this function is called. It's automatically called when importing
    kvpress to ensure compatibility with head-wise compression methods.

    Notes
    -----
    This function modifies the global attention functions in the transformers
    library. The modifications do not affect models that don't use head-wise compression (i.e. don't have
    module.masked_key_indices).
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        if _is_transformers_flash_attention_forward(func):
            func = _flash_attention_forward_without_forced_sink
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
