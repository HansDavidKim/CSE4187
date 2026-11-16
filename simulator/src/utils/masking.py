"""Masked-token sampling shared by the sequence encoder."""
import torch

MASK = 6


def mask_tokens(tokens: torch.Tensor, lengths: torch.Tensor, probability: float,
                generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask valid positions, selecting at least one target per nonempty row."""
    if not 0 < probability < 1:
        raise ValueError("mask probability must be between zero and one")
    positions = torch.arange(tokens.shape[1]).expand_as(tokens)
    valid = positions < lengths.unsqueeze(1)
    selected = valid & (torch.rand(tokens.shape, generator=generator) < probability)
    for row, length in enumerate(lengths.tolist()):
        if length and not selected[row, :length].any():
            selected[row, length - 1] = True
    masked = tokens.clone()
    masked[selected] = MASK
    return masked, selected
