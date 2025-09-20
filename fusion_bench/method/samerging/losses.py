import torch
from torch import Tensor


def compute_kl_loss(merged_logits, expert_logits, temperature=1.0):
    """Compute KL divergence loss between merged and expert logits."""
    merged_probs = torch.softmax(merged_logits / temperature, dim=-1)
    expert_probs = torch.softmax(expert_logits / temperature, dim=-1)
    return torch.nn.functional.kl_div(
        merged_probs.log(),
        expert_probs.log(),
        reduction="batchmean",
        log_target=True,
    )


def compute_jsd_loss(merged_logits, expert_logits):
    """Compute JSD loss between merged and expert logits."""
    mixed_logits = (merged_logits + expert_logits) / 2
    mixed_probs = torch.softmax(mixed_logits, dim=-1)
    merged_probs = torch.softmax(merged_logits, dim=-1)
    expert_probs = torch.softmax(expert_logits, dim=-1)
    loss = torch.nn.functional.kl_div(
        mixed_probs.log(),
        merged_probs.log(),
        reduction="batchmean",
        log_target=True,
    )
    loss += torch.nn.functional.kl_div(
        mixed_probs.log(),
        expert_probs.log(),
        reduction="batchmean",
        log_target=True,
    )
    return loss


def compute_ce_loss(merged_logits, expert_logits):
    """Compute cross-entropy loss between merged and expert logits."""
    return torch.nn.functional.cross_entropy(
        merged_logits, expert_logits.argmax(dim=-1)
    )


def entropy_loss(logits: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.
        eps (float): A small value to avoid log(0). Default is 1e-8.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    # Ensure the logits tensor has 2 dimensions
    assert (
        logits.dim() == 2
    ), f"Expected logits to have 2 dimensions, found {logits.dim()}, {logits.size()=}"

    # Compute the softmax probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute the entropy loss
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()
