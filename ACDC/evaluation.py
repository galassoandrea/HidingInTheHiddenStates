import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, roc_curve

"""Evaluation functions for circuit discovery and model outputs."""

def kl_divergence(clean_logits, corrupted_logits, dim: int = -1):
    """Compute KL divergence between two logit distributions: KL(P || Q)."""
    # Convert logits to probability distributions
    log_probs_clean = F.log_softmax(clean_logits, dim=dim)  # log P
    log_probs_corrupted = F.log_softmax(corrupted_logits, dim=dim)  # log Q
    probs_clean = log_probs_clean.exp()  # P

    # KL(P || Q) = sum P * (logP - logQ)
    kl = torch.sum(probs_clean * (log_probs_clean - log_probs_corrupted), dim=dim)

    if kl.dim() > 1:
        kl = kl.mean(dim=-1)

    return kl.mean()

def evaluate_factuality(all_logits: torch.Tensor, all_labels, model):
    """
    Optimized version that works directly with batches.
    """

    # Get token IDs for '0' and '1'
    token_0_id = model.to_tokens("0", prepend_bos=False)[0, 0].item()
    token_1_id = model.to_tokens("1", prepend_bos=False)[0, 0].item()

    # Extract logits at the last position for tokens "0" and "1"
    logits_0 = all_logits[:, token_0_id]
    logits_1 = all_logits[:, token_1_id]

    # Stack logits for "0" and "1" into a 2D tensor
    binary_logits = torch.stack([logits_0, logits_1], dim=1)

    # Compute probabilities using softmax
    probs = F.softmax(binary_logits, dim=1)
    probs_1 = probs[:, 1].numpy()

    # Get predictions: 1 if logit_1 > logit_0, else 0
    predictions = (logits_1 > logits_0).long().numpy()

    # Convert labels to numpy if needed
    labels_np = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(labels_np, predictions)
    roc_auc = roc_auc_score(labels_np, probs_1)

    # Compute NLL (Negative Log-Likelihood)
    # Convert labels to tensor for loss computation
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    nll = F.cross_entropy(binary_logits, labels_tensor).item()

    print(f"Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, NLL: {nll:.4f}")
