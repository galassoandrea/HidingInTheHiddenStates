import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from ACDC.ACDC import ACDC
from ACDC.utils import add_circuit_hooks
from ACDC.evaluation import evaluate_factuality
from ACDC.ComputationalGraph import build_computational_graph
from ACDC.visualization import visualize_computational_graph
import pandas as pd

model_name = "EleutherAI/pythia-70m-deduped"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "google/gemma-2-2b-it"
#model_name = "Qwen/Qwen3-0.6B"

model = HookedTransformer.from_pretrained(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

list_of_datasets = [
    "animals",
    "cities",
    "elements",
    "companies",
    "inventions",
    "facts"
]

# Get full-model performance for factuality task both on single datasets and overall
all_logits = []
all_labels = []
for dataset_to_use in list_of_datasets:
    logits = []
    labels = []
    # Read the CSV file
    df = pd.read_csv("resources/" + dataset_to_use + "_true_false.csv")
    # Get full-model performance for factuality task
    for example in tqdm(df, desc="Evaluating examples"):
        prompt = example['statement']
        prompt = prompt.rstrip(".")
        with torch.no_grad():
            prompt_tokens = model.to_tokens(prompt).to(model.cfg.device)
            output = model(prompt_tokens)
            if hasattr(output, 'logits'):
                logit = output.logits
            else:
                logit = output
        logits.append(logit.cpu())
        labels.append(example.label)
    print(f"Factuality evaluation for {dataset_to_use} dataset: ")
    evaluate_factuality(logits, labels, model)
    all_logits.extend(logits)
    all_labels.extend(labels)
print(f"Overall factuality evaluation: ")
evaluate_factuality(all_logits, all_labels, model)

# Run ACDC and extract a circuit
algorithm = ACDC(model, model_name, mode="greedy", method="patching", threshold=0.05)
#initial_graph = build_computational_graph(model, model_name)
circuit = algorithm.discover_circuit()
#visualize_computational_graph(initial_graph)
#visualize_computational_graph(circuit)

# Add hooks for removed (unimportant) nodes to the model, to run the model without those nodes
add_circuit_hooks(model, model_name)

# Get ablated-model performance
all_logits = []
all_labels = []
for dataset_to_use in list_of_datasets:
    logits = []
    labels = []
    # Read the CSV file
    df = pd.read_csv("resources/" + dataset_to_use + "_true_false.csv")
    # Get full-model performance for factuality task
    for example in tqdm(df, desc="Evaluating examples"):
        prompt = example['statement']
        prompt = prompt.rstrip(".")
        with torch.no_grad():
            prompt_tokens = model.to_tokens(prompt).to(model.cfg.device)
            output = model(prompt_tokens)
            if hasattr(output, 'logits'):
                logit = output.logits
            else:
                logit = output
        logits.append(logit.cpu())
        labels.append(example.label)
    print(f"Factuality evaluation for {dataset_to_use} dataset: ")
    evaluate_factuality(logits, labels, model)
    all_logits.extend(logits)
    all_labels.extend(labels)
print(f"Overall factuality evaluation: ")
evaluate_factuality(all_logits, all_labels, model)