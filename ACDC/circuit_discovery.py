import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from ACDC import ACDC
from utils import add_circuit_hooks
from evaluation import evaluate_factuality
from ComputationalGraph import build_computational_graph
from visualization import visualize_computational_graph

#model_name = "EleutherAI/pythia-70m-deduped"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "google/gemma-2-2b-it"
model_name = "Qwen/Qwen3-0.6B"

model = HookedTransformer.from_pretrained(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Run ACDC and extract a circuit

algorithm = ACDC(model, model_name, mode="greedy", method="patching", threshold=0.05)

all_logits = []
all_labels = []

# Get full-model performance for factuality task
for example in tqdm(algorithm.dataset, desc="Evaluating examples"):
    prompt_tokens = torch.tensor(example.clean_tokens).unsqueeze(0).to(model.cfg.device)
    with torch.no_grad():
        output = model(prompt_tokens)
        if hasattr(output, 'logits'):
            logit = output.logits
        else:
            logit = output
    all_logits.append(logit.cpu())
    all_labels.append(example.label)
evaluate_factuality(all_logits, all_labels, model)

# Clean memory
del all_logits
del all_labels

#initial_graph = build_computational_graph(model, model_name)
circuit = algorithm.discover_circuit()
#visualize_computational_graph(initial_graph)
#visualize_computational_graph(circuit)