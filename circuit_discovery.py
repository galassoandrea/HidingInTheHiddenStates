import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from ACDC.ACDC import ACDC
from ACDC.utils import add_circuit_hooks
from ACDC.evaluation import evaluate_factuality
from ACDC.ComputationalGraph import build_computational_graph, ComputationalGraph
from ACDC.visualization import visualize_computational_graph
import pandas as pd

def get_task_performance(list_of_datasets, batch_size=32):
    all_logits = []
    all_labels = []

    for dataset_to_use in list_of_datasets:
        logits = []

        # Read the CSV file
        df = pd.read_csv("resources/" + dataset_to_use + "_true_false.csv", nrows=100)

        # Prepare data
        statements = df['statement'].str.rstrip(".").tolist()
        labels = df['label'].tolist()

        # Process in batches
        for i in tqdm(range(0, len(statements), batch_size), desc=f"Evaluating {dataset_to_use}"):
            batch_statements = statements[i:i + batch_size]

            with torch.no_grad():
                # Tokenize batch with padding
                prompt_tokens = model.to_tokens(batch_statements, padding_side="left").to(model.cfg.device)
                output = model(prompt_tokens)

                if hasattr(output, 'logits'):
                    batch_logits = output.logits
                else:
                    batch_logits = output

                # Store individual logits
                for j in range(batch_logits.shape[0]):
                    logits.append(batch_logits[j:j + 1].cpu())

        print(f"Factuality evaluation for {dataset_to_use} dataset: ")
        evaluate_factuality(logits, labels, model)
        all_logits.extend(logits)
        all_labels.extend(labels)
    print(f"Overall factuality evaluation: ")
    evaluate_factuality(all_logits, all_labels, model)

model_name = "EleutherAI/pythia-14m"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "google/gemma-2-2b-it"
#model_name = "Qwen/Qwen3-0.6B"

model = HookedTransformer.from_pretrained(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

list_of_datasets = [
    "animals",
    "cities",
    "elements",
    "companies",
    "inventions",
    "facts"
]

# Get full-model performance for factuality task both on single dataframes and overall
#get_task_performance(list_of_datasets)

## Run ACDC and extract a circuit
algorithm = ACDC(model, model_name, mode="greedy", method="patching", target="edge", threshold=0.05)
initial_graph = build_computational_graph(model, model_name, granularity="head")
visualize_computational_graph(initial_graph)

circuit = algorithm.run()
##visualize_computational_graph(circuit)
#
# Add hooks for removed (unimportant) nodes to the model, to run the model without those nodes
#add_circuit_hooks(model, model_name)

# Get ablated-model performance
#get_task_performance(list_of_datasets)