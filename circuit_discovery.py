import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from ACDC.ACDC import ACDCNode
from ACDC.utils import add_circuit_hooks
from ACDC.evaluation import evaluate_factuality
from ACDC.ComputationalGraph import build_computational_graph, ComputationalGraph
from ACDC.visualization import visualize_computational_graph
import pandas as pd

def get_task_performance(list_of_datasets, batch_size=32):
    all_logits = []
    all_labels = []

    df_all = pd.DataFrame()
    for d in list_of_datasets:
        data = pd.read_csv("resources/" + d + "_true_false.csv", nrows=100)
        data["topic"] = d
        df_all = pd.concat([df_all, data], ignore_index=True)

    for topic in list_of_datasets:
        df = df_all[df_all["topic"] == topic].copy()
        logits = []

        prompts = df['statement'].tolist()
        labels = df['label'].tolist()

        # Tokenize with padding
        prompt_tokens = model.to_tokens(prompts, padding_side="left").to(model.cfg.device)

        # Process in batches
        for i in tqdm(range(0, len(prompt_tokens), batch_size), desc=f"Evaluating {topic}"):
            batch = prompt_tokens[i:i + batch_size]

            with torch.no_grad():
                output = model(batch)

                if hasattr(output, 'logits'):
                    batch_logits = output.logits
                else:
                    batch_logits = output

                last_token_logits = batch_logits[:, -1, :]
                logits.append(last_token_logits.cpu())

        # Concatenate all batches at once
        logits = torch.cat(logits, dim=0)

        print(f"Factuality evaluation for {topic} dataset: ")
        evaluate_factuality(logits, labels, model)
        all_logits.append(logits)
        all_labels.extend(labels)

    # Concatenate all datasets
    all_logits = torch.cat(all_logits, dim=0)

    print(f"Overall factuality evaluation: ")
    evaluate_factuality(all_logits, all_labels, model)

#model_name = "EleutherAI/pythia-14m"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "google/gemma-2-2b-it"
model_name = "Qwen/Qwen3-0.6B"

# Load the model
model = HookedTransformer.from_pretrained(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

model.set_use_attn_result(True)
#model.set_use_split_qkv_input(True)

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
algorithm = ACDCNode(model, model_name, mode="greedy", method="patching", threshold=0.1)
#initial_graph = build_computational_graph(model, model_name, granularity="block")
#visualize_computational_graph(initial_graph)

circuit = algorithm.run()
##visualize_computational_graph(circuit)
#
# Add hooks for removed (unimportant) nodes to the model, to run the model without those nodes
#add_circuit_hooks(model, model_name)

# Get ablated-model performance
get_task_performance(list_of_datasets)