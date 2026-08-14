import torch

from terpedia_esm2.pooling import masked_mean_pool


def test_masked_mean_excludes_padding_and_special_tokens() -> None:
    hidden = torch.tensor([[[100.0], [2.0], [4.0], [200.0], [999.0]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    pooled = masked_mean_pool(hidden, attention, special)
    assert pooled.tolist() == [[3.0]]


def test_masked_mean_is_finite_for_empty_residue_mask() -> None:
    hidden = torch.ones((1, 2, 4))
    pooled = masked_mean_pool(hidden, torch.ones((1, 2)), torch.ones((1, 2)))
    assert pooled.tolist() == [[0.0, 0.0, 0.0, 0.0]]
