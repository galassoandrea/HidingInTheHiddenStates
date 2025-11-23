from typing import Set

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, roc_curve, auc

"""Evaluation functions for circuit discovery and model outputs."""

def kl_divergence(clean_logits, corrupted_logits, dim: int = -1):
    """Compute KL divergence between two logit distributions: KL(P || Q)."""
    # Convert to log-probabilities and probabilities
    clean_log_probs = F.log_softmax(clean_logits, dim=-1)
    clean_probs = F.softmax(clean_logits, dim=-1)

    patched_log_probs = F.log_softmax(corrupted_logits, dim=-1)

    # Compute KL Divergence: D_KL(Clean || Patched)
    kl_div = F.kl_div(
        patched_log_probs,  # The "new" distribution (Q)
        clean_probs,  # The "target" distribution (P)
        reduction='batchmean',
        log_target=False
    )

    return kl_div

def compute_logit_difference(logits, true_id, false_id, labels):

    # Extract logits for true and false
    true_logits = logits[:, true_id]
    false_logits = logits[:, false_id]

    # Calculate logit-diff
    logit_diff = true_logits - false_logits

    # When answer is false (0), flip the sign
    sign = 2 * labels.float() - 1  # Maps 0->-1, 1->1
    logit_diff = logit_diff * sign

    return logit_diff

def compute_acdc_score(clean_logits, corrupted_logits, true_id, false_id, labels):
    clean_logit_diff = compute_logit_difference(clean_logits, true_id, false_id, labels)
    corrupted_logit_diff = compute_logit_difference(corrupted_logits, true_id, false_id, labels)
    difference = clean_logit_diff - corrupted_logit_diff
    return difference.mean()

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculates the Jaccard similarity coefficient between two sets."""

    # Handle the case of two empty sets (Jaccard is 1, they are identical)
    if not set1 and not set2:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Handle division by zero (if union is 0 but not both sets, similarity is 0)
    if union == 0:
        return 0.0

    return intersection / union

def evaluate_factuality(model, list_of_datasets, average_only=False):

    all_labels = []
    all_scores = []
    all_nlls = []
    all_predictions = []

    model.eval()

    # Get token IDs for 'true' and 'false'
    true_id = model.to_single_token("true")
    false_id = model.to_single_token("false")

    for dataset_to_use in list_of_datasets:
        dataset_labels = []
        dataset_scores = []
        dataset_nlls = []
        dataset_predictions = []

        df = pd.read_csv(f"resources/{dataset_to_use}_few_shot_prompt.csv").head(500)

        for i in range(len(df)):
            prompt = str(df.iloc[i]['prompt'])
            label = int(df.at[i, 'label'])

            with torch.no_grad():
                inputs = model.to_tokens(prompt, prepend_bos=True)
                logits = model(inputs)
                # Get the logits for the last token
                next_token_logits = logits[0, -1, :]

                # Calculate Probabilities (as the paper does)
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                prob_true = probs[true_id].item()
                prob_false = probs[false_id].item()

                # Calculate score as ratio P(true) / P(false)
                # Add a small epsilon to avoid division by zero
                score_ratio = prob_true / (prob_false + 1e-9)

                # Calculate NLL
                log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                correct_token_id = true_id if label == 1 else false_id
                nll_score = -log_probs[correct_token_id].item()

                dataset_labels.append(label)
                dataset_scores.append(score_ratio)
                dataset_nlls.append(nll_score)
                dataset_predictions.append(1 if score_ratio > 1.0 else 0)

        if not average_only:
            acc = accuracy_score(dataset_labels, dataset_predictions)
            # AUC is calculated with the raw scores (ratios)
            fpr, tpr, _ = roc_curve(dataset_labels, dataset_scores)
            roc_auc_val = auc(fpr, tpr)
            nll = np.mean(dataset_nlls)

            print(f"Accuracy for topic {dataset_to_use}: {acc:.4f}")
            print(f"AUC for topic {dataset_to_use}: {roc_auc_val:.4f}")
            print(f"NLL for topic {dataset_to_use}: {nll:.4f}")

        all_labels.extend(dataset_labels)
        all_scores.extend(dataset_scores)
        all_nlls.extend(dataset_nlls)
        all_predictions.extend(dataset_predictions)

    acc = accuracy_score(all_labels, all_predictions)
    # AUC is calculated with the raw scores (ratios)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc_val = auc(fpr, tpr)
    nll = np.mean(all_nlls)

    print("-" * 30)
    print(f"Average accuracy: {acc:.4f}")
    print(f"Average AUC: {roc_auc_val:.4f}")
    print(f"Average NLL: {nll:.4f}")
    print("-" * 30)

    return acc, roc_auc_val, nll


def evaluate_factuality_it_is_true_baseline(model, list_of_datasets, average_only=False):

    all_predictions = []
    all_labels = []
    all_scores = []
    all_nlls = []

    model.eval()

    for dataset_to_use in list_of_datasets:
        predictions = []
        labels = []
        scores = []
        nll_scores = []

        df = pd.read_csv(f"resources/{dataset_to_use}_true_false.csv").head(500)

        for i in range(len(df)):
            # Get the original statement 'X' and its label
            original_statement = str(df.iloc[i]['statement'])
            label = int(df.at[i, 'label'])

            # Create the two prompt contexts
            prompt_true_context = f"It is true that {original_statement}"
            prompt_false_context = f"It is false that {original_statement}"

            with torch.no_grad():

                # Tokenize the full prompts (transformer-lens adds BOS by default)
                tokens_true = model.to_tokens(prompt_true_context)
                tokens_false = model.to_tokens(prompt_false_context)

                # Tokenize the statement 'X' *without* BOS to get its tokens
                tokens_x = model.to_tokens(original_statement, prepend_bos=False)

                # Tokenize the prefixes *with* BOS to find the length
                tokens_prefix_true = model.to_tokens("It is true that ")
                tokens_prefix_false = model.to_tokens("It is false that ")

                # Get the number of tokens
                len_x = tokens_x.shape[1]
                # Length of prefix *including* BOS
                len_prefix_true = tokens_prefix_true.shape[1]
                len_prefix_false = tokens_prefix_false.shape[1]

                # Get logits: shape [batch, seq_len, d_vocab]
                logits_true = model(tokens_true)
                logits_false = model(tokens_false)

                # Get log_probs: shape [batch, seq_len, d_vocab]
                # We use [0] to remove the batch dimension
                log_probs_true = torch.nn.functional.log_softmax(logits_true[0], dim=-1)
                log_probs_false = torch.nn.functional.log_softmax(logits_false[0], dim=-1)

                score_true = 0.0
                score_false = 0.0

                for j in range(len_x):
                    # Get the j-th token ID of the statement 'X'
                    token_id = tokens_x[0, j]

                    true_logit_idx = (len_prefix_true - 1) + j
                    false_logit_idx = (len_prefix_false - 1) + j

                    # Ensure we are not indexing out of bounds
                    if true_logit_idx < log_probs_true.shape[0] and \
                            false_logit_idx < log_probs_false.shape[0]:

                        # Add the log-probability of the *correct* token
                        score_true += log_probs_true[true_logit_idx, token_id].item()
                        score_false += log_probs_false[false_logit_idx, token_id].item()
                    else:
                        print(f"Warning: Index mismatch for dataset {dataset_to_use}, row {i}. Skipping token {j}.")

                # The continuous score for ROC-AUC is the log-prob difference
                score_diff = score_true - score_false

                # Binary prediction for accuracy
                prediction = 1 if score_diff > 0 else 0

                # Calculate NLL
                # We normalize the two scores into a probability distribution
                # and take the NLL of the correct label.
                scores_for_nll = torch.tensor([score_false, score_true])  # [P(false), P(true)]
                log_probs_nll = torch.nn.functional.log_softmax(scores_for_nll, dim=-1)
                nll_score = -log_probs_nll[label].item()  # label is 0 or 1

                predictions.append(prediction)
                labels.append(label)
                scores.append(score_diff)
                nll_scores.append(nll_score)

        if not average_only:
            # Calculate metrics
            acc = accuracy_score(labels, predictions)
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc_val = auc(fpr, tpr)
            nll = np.mean(nll_scores)

            print(f"Accuracy for topic {dataset_to_use}: {acc:.4f}")
            print(f"AUC for topic {dataset_to_use}: {roc_auc_val:.4f}")
            print(f"NLL for topic {dataset_to_use}: {nll:.4f}")

        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_scores.extend(scores)
        all_nlls.extend(nll_scores)

    # Compute average performance
    acc = accuracy_score(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc_val = auc(fpr, tpr)
    nll = np.mean(all_nlls)

    print("-" * 30)
    print(f"Average accuracy: {acc:.4f}")
    print(f"Average AUC: {roc_auc_val:.4f}")
    print(f"Average NLL: {nll:.4f}")
    print("-" * 30)

    return acc, roc_auc_val, nll


