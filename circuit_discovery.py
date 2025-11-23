import os
from typing import List, Tuple, Optional
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from ACDC.ACDC import ACDCNode
from ACDC.utils import add_circuit_hooks, get_node_id
from ACDC.evaluation import evaluate_factuality
from ACDC.ComputationalGraph import build_computational_graph, ComputationalGraph
from ACDC.visualization import visualize_computational_graph, plot_circuit_discovery_heatmap, plot_circuit_convergence, \
    plot_scores_by_threshold, plot_scores_by_n_samples, plot_results_multiple_samples
import pandas as pd


def run_circuit_discovery_multiple_samples(model, n_samples_per_iteration, threshold, topics = None):
    num_runs = len(n_samples_per_iteration)
    # All topics except facts
    if topics is None:
        topics = [ "animals", "cities", "elements", "companies", "inventions"]
    file_path = f"removed_components/{model_name.replace('/', '-')}-t{threshold}.json"
    if os.path.exists(file_path):
        os.remove(file_path)
    for i, n_samples in enumerate(n_samples_per_iteration):
        print(f"\nRun {i+1}/{num_runs} with {n_samples} samples per iteration\n")
        algorithm = ACDCNode(model, model_name, mode="greedy", method="patching", threshold=threshold,
                             num_samples=n_samples, topics=topics)
        _, _ = algorithm.run(batch_size=8)


def evaluate_circuit_performance_multiple_samples(model, n_samples_per_iteration, threshold, topics=None):

    results_data = []
    if topics is None:
        topics = ["animals", "cities", "elements", "companies", "inventions", "facts"]

    file_path = f"removed_components/{model_name.replace('/', '-')}-t{threshold}.json"
    if os.path.exists(file_path):
        os.remove(file_path)

    for i, n_samples in enumerate(n_samples_per_iteration):
        print(f"\nEvaluating circuit for {n_samples} samples (Run {i + 1}/{len(n_samples_per_iteration)})\n")

        # Add hooks for this specific experiment
        add_circuit_hooks(model, model_name, threshold, i)

        # Compute performance and store data
        accuracy, roc_auc, nll = evaluate_factuality(model, topics, average_only=True)
        results_data.append({
            "N_samples": n_samples,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
            "NLL": nll
        })
        model.reset_hooks()

    # Create results dataframe
    df_results = pd.DataFrame(results_data)
    plot_scores_by_n_samples(df_results)
    print(df_results)
    df_results.to_csv(f"removed_components/{model_name.replace('/', '-')}-t{threshold}.csv")


#model_name = "EleutherAI/pythia-14m"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "Qwen/Qwen3-0.6B"

# Load the model
model = HookedTransformer.from_pretrained(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model.set_use_attn_result(True)

# ---------------------------------------------------
# ---- RUN EXPERIMENTS USING THE CODE BELOW ----
# If you want to run ACDC over a specific topic, insert it in a list as the last argument of the function:
# e.g. run_circuit_discovery_multiple_samples(model, n_samples_per_iteration, threshold=0.1, topics=['animals'])
# ---- DE-COMMENT THE FUNCTION FOR THE EXPERIMENTS YOU WANT TO EXECUTE ----
# ---------------------------------------------------

# Run ACDC over different numbers of dataset elements
n_samples_per_iteration = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
run_circuit_discovery_multiple_samples(model, n_samples_per_iteration, threshold=0.1)
plot_results_multiple_samples(model_name, threshold=0.1)
evaluate_circuit_performance_multiple_samples(model, n_samples_per_iteration, threshold=0.1)

# Run ACDC over different thresholds
#run_circuit_discovery_multiple_thresholds(model, thresholds=[0.1, 0.2], n_samples=50)

