from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, roc_curve

"""Evaluation functions for circuit discovery and model outputs."""

def kl_divergence(clean_logits, corrupted_logits, dim: int = -1):
    """Compute KL divergence between two logit distributions: KL(P || Q)."""
    # Convert logits to probability distributions
    log_probs_a = F.log_softmax(clean_logits, dim=dim)  # log P
    log_probs_b = F.log_softmax(corrupted_logits, dim=dim)  # log Q
    probs_a = log_probs_a.exp()  # P

    # KL(P || Q) = sum P * (logP - logQ)
    kl = torch.sum(probs_a * (log_probs_a - log_probs_b), dim=dim)

    return kl.mean()

def evaluate_factuality(all_logits: List[torch.Tensor], all_labels, model):
    """
    Optimized version that works directly with batches.
    """

    all_predictions = []
    all_probs_positive = []

    # Get token IDs for '0' and '1' - moved outside loop for efficiency
    token_0_id = model.to_tokens("0", prepend_bos=False)[0, 0].item()
    token_1_id = model.to_tokens("1", prepend_bos=False)[0, 0].item()

    # Process each logit tensor
    for logits in all_logits:
        # logits_batch shape: [batch_size, seq_len, vocab_size]
        batch_size = logits.shape[0]

        # Extract logits for the next token (last position) for all samples in batch
        next_token_logits = logits[0, -1, :]

        # Extract logits for tokens '0' and '1' for all samples
        binary_logits = torch.stack([
            next_token_logits[token_0_id],  # Logits for '0' across batch
            next_token_logits[token_1_id]  # Logits for '1' across batch
        ])

        # Convert to probabilities for the entire batch
        probs = torch.softmax(binary_logits, dim=0).cpu().numpy()

        # Get predictions for the entire batch
        predictions = np.argmax(probs)
        probs_positive = probs[1]

        all_predictions.append(predictions)
        all_probs_positive.append(probs_positive)

    # Convert labels to numpy array
    ground_truths = np.array(all_labels)

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    probs_positive = np.array(all_probs_positive)

    # Create probability matrix for log_loss
    probs_matrix = np.column_stack([1 - probs_positive, probs_positive])

    # Calculate metrics
    accuracy = accuracy_score(ground_truths, predictions)

    # ROC-AUC (only if both classes are present)
    try:
        if len(np.unique(ground_truths)) > 1:
            roc_auc = roc_auc_score(ground_truths, probs_positive)
            fpr, tpr, _ = roc_curve(ground_truths, probs_positive)
            plt.plot(fpr, tpr)
        else:
            roc_auc = float('nan')
            print("Warning: Only one class present in labels, cannot compute ROC-AUC")
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        roc_auc = float('nan')

    # Negative Log-Likelihood
    try:
        nll = log_loss(ground_truths, probs_matrix, labels=[0, 1])
    except Exception as e:
        print(f"Warning: Could not compute NLL: {e}")
        nll = float('nan')

    print(f"Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, NLL: {nll:.4f}")

    return {'accuracy': accuracy, 'roc_auc': roc_auc, 'nll': nll}

