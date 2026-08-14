from torch import Tensor


def masked_mean_pool(
    hidden_states: Tensor,
    attention_mask: Tensor,
    special_tokens_mask: Tensor | None = None,
) -> Tensor:
    """Mean-pool valid residue tokens and exclude tokenizer special tokens."""
    mask = attention_mask.bool()
    if special_tokens_mask is not None:
        mask = mask & ~special_tokens_mask.bool()
    weights = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denominator
