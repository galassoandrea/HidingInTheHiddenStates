"""
Based on the original code provided by Azaria & Mitchell, 2023. (The Internal State of an LLM Knows When It's Lying)

@inproceedings{azaria-mitchell-2023-internal,
    title = "The Internal State of an {LLM} Knows When It`s Lying",
    author = "Azaria, Amos  and
      Mitchell, Tom",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.68/",
    doi = "10.18653/v1/2023.findings-emnlp.68",
    pages = "967--976",
    abstract = "While Large Language Models (LLMs) have shown exceptional performance in various tasks, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. In this paper, we provide evidence that the LLM`s internal state can be used to reveal the truthfulness of statements. This includes both statements provided to the LLM, and statements that the LLM itself generates. Our approach is to train a classifier that outputs the probability that a statement is truthful, based on the hidden layer activations of the LLM as it reads or generates the statement. Experiments demonstrate that given a set of test sentences, of which half are true and half false, our trained classifier achieves an average of 71{\%} to 83{\%} accuracy labeling which sentences are true versus false, depending on the LLM base model. Furthermore, we explore the relationship between our classifier`s performance and approaches based on the probability assigned to the sentence by the LLM. We show that while LLM-assigned sentence probability is related to sentence truthfulness, this probability is also dependent on sentence length and the frequencies of words in the sentence, resulting in our trained classifier providing a more reliable approach to detecting truthfulness, highlighting its potential to enhance the reliability of LLM-generated content and its practical applicability in real-world scenarios."
}
"""
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
import torch
import pandas as pd
import numpy as np
from typing import Dict
from ACDC.utils import add_circuit_hooks


#layers_to_use = [-1, -4, -8, -12, -16]
layers_to_use = [-1, -2, -4]
list_of_datasets = [
    "animals",
    "cities",
    "elements",
    "companies",
    "inventions",
    "facts"
]

#model_name = "meta-llama/Llama-2-7b-hf"
#model_name = "Qwen/Qwen3-0.6B"
#model_name = "facebook/opt-6.7b"
model_name = "EleutherAI/pythia-70m-deduped"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

remove_period = True

# Load model as HookedTransformer from transformer lens
model = HookedTransformer.from_pretrained(
    model_name,
    device=device,
    torch_dtype=torch.float16
)

# Add hooks for removed (unimportant) nodes to the model, to run the model without those nodes
add_circuit_hooks(model, model_name)

dfs: Dict[int, pd.DataFrame] = {}

with torch.no_grad():
    for dataset_to_use in list_of_datasets:
        # Read the CSV file
        df = pd.read_csv("resources/" + dataset_to_use + "_true_false.csv")
        df['embeddings'] = pd.Series(dtype='object')
        df['next_id'] = pd.Series(dtype=float)
        for layer in layers_to_use:
            dfs[layer] = df.copy()

        for i, row in df.iterrows():
            prompt = row['statement']
            if remove_period:
                prompt = prompt.rstrip(". ")
            inputs = model.to_tokens(prompt)

            logits, cache = model.run_with_cache(inputs)
            next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            for layer in layers_to_use:
                last_hidden_state = cache["resid_post", layer][0, -1].cpu()
                dfs[layer].at[i,'embeddings'] = [last_hidden_state.numpy().tolist()]
                dfs[layer].at[i, 'next_id'] = next_id
            print("processing: " + str(i) + ", next_token:" + str(next_id))

        for layer in layers_to_use:
            dfs[layer].to_csv("embeddings/" + "embeddings_with_labels_" + dataset_to_use + "_" + model_name.split("/")[-1] + "_" + str(abs(layer)) + "_rmv_period.csv", index=False)
