import json
import os
import torch
from typing import List, Optional, Callable
from .evaluation import kl_divergence
import numpy as np
from .ComputationalGraph import Node, Edge

def precompute_node_contributions(graph, device, granularity, clean_caches: Optional = None, corrupted_caches: Optional = None):
    """Precompute all node contributions for all examples."""
    clean_node_contributions = {} if clean_caches is not None else None
    corrupted_node_contributions = {} if corrupted_caches is not None else None
    for node in graph.nodes:
        node_id = get_node_id(node)
        if node.component_type == "embedding":
            if granularity == "head":
                clean_node_contributions[node_id] = clean_caches[node.full_activation].to(device)
                if corrupted_caches is not None:
                    corrupted_node_contributions[node_id] = corrupted_caches[node.full_activation].to(device)
        elif node.component_type == "attention":
            if clean_caches is not None:
                clean_activation = clean_caches[node.full_activation]
                clean_node_contributions[node_id] = clean_activation[:, :, node.head_idx, :].to(device)
            if corrupted_caches is not None:
                corrupted_activation = corrupted_caches[node.full_activation]
                corrupted_node_contributions[node_id] = corrupted_activation[:, :, node.head_idx, :].to(
                    device)
        else:
            if clean_caches is not None:
                clean_node_contributions[node_id] = clean_caches[node.full_activation].to(device)
            if corrupted_caches is not None:
                corrupted_node_contributions[node_id] = corrupted_caches[node.full_activation].to(device)
    return clean_node_contributions, corrupted_node_contributions

def create_node_patching_hook(
        method,
        node: Node,
        corrupted_contributions: Optional[torch.Tensor] = None
) -> Callable:
    """Create a hook function for node ablation/patching."""

    def patching_hook(activation, hook):
        if node.name == "hook_result":
            # activation shape: [batch_size, seq_len, n_heads, d_head]
            patched_activation = activation.clone()
            if method == "patching":
                # batched_corrupted_contribution: [batch_size, seq_len, d_head]
                patched_activation[:, :, node.head_idx, :] = corrupted_contributions
            else:
                patched_activation[:, :, node.head_idx, :] = 0
        else:
            # activation shape: [batch_size, seq_len, d_model] or similar
            if method == "patching":
                # batched_corrupted_contribution: [batch_size, seq_len, d_model]
                patched_activation = corrupted_contributions
            else:
                patched_activation = torch.zeros_like(activation)

        return patched_activation

    return patching_hook

def create_edge_patching_hook(
        method,
        node: Node,
        clean_sender_contribution: torch.Tensor,
        corrupted_sender_contribution: Optional[torch.Tensor] = None
) -> Callable:
    """Create a hook function for edge pruning/patching."""

    def patching_hook(activation, hook):
        # Subtract sender's contribution from the output
        if node.component_type == "attention":
            patched_activation = activation
            if method == "patching":
                patched_activation[:,:,node.head_idx, :] = activation[:,:,node.head_idx, :] - clean_sender_contribution + corrupted_sender_contribution
            else:
                patched_activation[:,:,node.head_idx, :] = activation[:,:,node.head_idx, :] - clean_sender_contribution
        else:
            if method == "patching":
                patched_activation = activation - clean_sender_contribution + corrupted_sender_contribution
            else:
                patched_activation = activation - clean_sender_contribution

        return patched_activation

    return patching_hook

def add_all_hooks(
        model,
        i,
        clean_node_contributions: Optional = None,
        ablated_nodes: Optional[List[Node]] = None,
        ablated_edges: Optional[List[Edge]] = None
):
    """Add hooks for all activations in the circuit."""
    if ablated_nodes != [] and ablated_nodes is not None:
        for node in ablated_nodes:
            hook = create_node_patching_hook(
                method="pruning",
                node=node
            )
            if hasattr(model, 'add_hook'):
                model.add_hook(node.full_activation, hook)

    if ablated_edges != [] and ablated_edges is not None:
        for edge in ablated_edges:
            node_id = get_node_id(edge.sender)
            hook = create_edge_patching_hook(
                method="pruning",
                node=edge.receiver,
                clean_sender_contribution=clean_node_contributions[node_id][i]
            )
            if hasattr(model, 'add_hook'):
                model.add_hook(edge.receiver.full_activation, hook)

def get_final_performance(
        model,
        clean_tokens,
        clean_labels,
        clean_logits,
        clean_node_contributions: Optional = None,
        tokenize = True,
        ablated_nodes: Optional[List[Node]] = None,
        ablated_edges: Optional[List[Edge]] = None
):
    """Get final performance after all edges/nodes have been evaluated."""
    kl_divs = []
    logits = []
    labels = []

    for i, example in enumerate(clean_tokens):
        # Clear previous hooks
        model.reset_hooks()

        if not tokenize: # Embedding dataset - skip embedding layer
            # Dummy tokens
            inputs = torch.zeros_like(clean_tokens[i]).long().to(model.cfg.device)

            def embedding_replacement_hook(activations, hook):
                return clean_tokens

            # Register hook on the node
            if hasattr(model, 'add_hook'):
                model.add_hook("hook_embed", embedding_replacement_hook)
        else:
            inputs = clean_tokens[i].to(model.cfg.device)

        # Add all hooks for patched edges/nodes
        add_all_hooks(model, i, clean_node_contributions, ablated_nodes, ablated_edges)

        with torch.no_grad():
            ablated_logits = model(inputs)
        kl_div = kl_divergence(clean_logits[i].to(model.cfg.device), ablated_logits)
        kl_divs.append(kl_div.item())
        logits.append(ablated_logits)
        labels.append(clean_labels[i])
    avg_kl_div = np.mean(kl_divs)
    # Free some memory
    model.reset_hooks()
    del clean_node_contributions
    del clean_logits
    logits = [t.cpu() for t in logits]
    torch.cuda.empty_cache()
    # Compute metrics for factuality evaluation
    #evaluate_factuality(logits, labels, model)
    return avg_kl_div

def get_activations_name(model_name, layers, target):
    names = []
    if target == "node":
        if "pythia" in model_name:
            hook_list = ["attn.hook_result", "hook_resid_pre", "hook_resid_post", "hook_mlp_out"]
        else:
            hook_list = ["attn.hook_result", "hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_mlp_out"]
        for hook_name in hook_list:
            hooks = [f"blocks.{l}.{hook_name}" for l in range(layers)]
            names.extend(hooks)
    else:
        hook_list = ["attn.hook_q_input", "attn.hook_result", "hook_resid_post", "hook_mlp_out", "hook_mlp_in"]
        for hook_name in hook_list:
            hooks = [f"blocks.{l}.{hook_name}" for l in range(layers)]
            names.extend(hooks)
            # Add embedding node
            names.append("hook_embed")
    return names

def get_node_id(node: Node) -> str:
    """Get the ID string for a given node."""
    if node.component_type == "attention":
        node_id = f"L{node.layer}-Head{node.head_idx}"
    elif node.component_type == "embedding":
        node_id = node.name
    elif node.component_type == "mlp":
        node_id = f"L{node.layer}-MLP"
    else:  # residual
        if "pre" in node.name:
            node_id = f"L{node.layer}-ResPre"
        elif "mid" in node.name:
            node_id = f"L{node.layer}-ResMid"
        else: # post
            node_id = f"L{node.layer}-ResPost"
    return node_id

def save_removed_components(model_name, threshold, num_samples, topics, ablated_nodes: Optional[List[Node]] = None, ablated_edges: Optional[List[Edge]] = None):
    # Store removed edges/nodes metadata in a json file
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to root, then into the save folder
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    save_dir = os.path.join(ROOT_DIR, "removed_components")
    os.makedirs(save_dir, exist_ok=True)
    if ablated_edges is not None and ablated_edges != []:
        experiment_data = {
            "ablated_edges": [
                {
                    "sender": {edge.sender.full_activation: edge.sender.head_idx},
                    "receiver": {edge.receiver.full_activation: edge.receiver.head_idx}
                }
                for edge in ablated_edges
            ]
        }
    elif ablated_nodes is not None and ablated_nodes != []:
        experiment_data = {
            "components": [get_node_id(node) for node in ablated_nodes],
            "samples": num_samples
        }
    # Build full file path
    if len(topics) == 1:
        save_path = os.path.join(save_dir, f"{model_name.replace('/', '-')}-t{threshold}-{topics[0]}.json")
    else:
        save_path = os.path.join(save_dir, f"{model_name.replace('/', '-')}-t{threshold}.json")
    # Initialize the data structure
    all_data = {"experiments": []}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            existing_data = json.load(f)
            # Check if it has the expected "experiments" list
            if isinstance(existing_data, dict) and "experiments" in existing_data:
                all_data = existing_data
            else:
                print(f"Warning: '{save_path}' had an unexpected format. Starting fresh.")
    # Append the new experiment's data
    all_data["experiments"].append(experiment_data)

    # Write the updated data back to the file
    with open(save_path, "w") as f:
        json.dump(all_data, f, indent=2)

def add_circuit_hooks(model, model_name, threshold, experiment_index: Optional = None):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to root, then into the save folder
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    path = os.path.join(ROOT_DIR, "removed_components", f"{model_name.replace('/', '-')}-t{threshold}.json")
    with open(path, "r") as f:
        params = json.load(f)
    if experiment_index is not None:
        experiment_data = params["experiments"][experiment_index]  # Load specific experiment
    else:
        experiment_data = params["experiments"][-1]  # Load the last experiment
    components = experiment_data["components"]
    for node in components:
        layer = int(node.split('-')[0][1:])
        if "Head" in node:
            # Attention node
            head_idx = int(node.split('-')[1][4:])
            full_activation = f"blocks.{layer - 1}.attn.hook_result"
            act_name = "hook_result"
            node = Node(
                name=act_name,
                layer=layer,
                component_type="attention",
                head_idx=head_idx,
                full_activation=full_activation
            )
        elif "MLP" in node:
            # MLP node
            full_activation = f"blocks.{layer - 1}.hook_mlp_out"
            act_name = "hook_mlp_out"
            node = Node(
                name=act_name,
                layer=layer,
                component_type="mlp",
                full_activation=full_activation
            )
        elif "ResPre" in node:
            # Residual pre node
            full_activation = f"blocks.{layer - 1}.hook_resid_pre"
            act_name = "hook_resid_pre"
            node = Node(
                name=act_name,
                layer=layer,
                component_type="residual",
                full_activation=full_activation
            )
        elif "ResMid" in node:
            # Residual mid node
            full_activation = f"blocks.{layer - 1}.hook_resid_mid"
            act_name = "hook_resid_mid"
            node = Node(
                name=act_name,
                layer=layer,
                component_type="residual",
                full_activation=full_activation
            )
        else:
            # Residual post node
            full_activation = f"blocks.{layer - 1}.hook_resid_post"
            act_name = "hook_resid_post"
            node = Node(
                name=act_name,
                layer=layer,
                component_type="residual",
                full_activation=full_activation
            )
        # Add hook to the model
        model.add_hook(node.full_activation, create_node_patching_hook(
            method="pruning",
            node=node
        ))
    print(f"Added ablation hooks for nodes: {components}")
