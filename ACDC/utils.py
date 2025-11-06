import json
import os
import torch
from typing import List, Optional, Callable
from .evaluation import kl_divergence, evaluate_factuality
import numpy as np
from .ComputationalGraph import Node, Edge


def precompute_node_contributions(graph, method, device, granularity, clean_caches, corrupted_caches: Optional = None):
    """Precompute all node contributions for all examples."""
    clean_node_contributions = {}
    corrupted_node_contributions = {}
    for node in graph.nodes:
        node_id = get_node_id(node)
        if node.component_type == "embedding":
            if granularity == "head":
                clean_node_contributions[node_id] = clean_caches[node.full_activation].to(device)
                if method == "patching":
                    corrupted_node_contributions[node_id] = corrupted_caches[node.full_activation].to(device)
        elif node.component_type == "attention":
            clean_activation = clean_caches[node.full_activation]
            clean_node_contributions[node_id] = clean_activation[:, :, node.head_idx, :].to(device)
            if method == "patching":
                corrupted_activation = corrupted_caches[node.full_activation]
                corrupted_node_contributions[node_id] = corrupted_activation[:, :, node.head_idx, :].to(
                    device)
        else:
            clean_node_contributions[node_id] = clean_caches[node.full_activation].to(device)
            if method == "patching":
                corrupted_node_contributions[node_id] = corrupted_caches[node.full_activation].to(device)
    return clean_node_contributions, corrupted_node_contributions

def create_node_patching_hook(
        method,
        node: Node,
        corrupted_node_contribution: Optional[torch.Tensor] = None
) -> Callable:
    """Create a hook function for node ablation/patching."""

    def patching_hook(activation, hook):
        # Check the number of dimensions of the activation
        if node.name == "hook_result":
            patched_activation = activation
            if method == "patching":
                patched_activation[:,:,node.head_idx, :] = corrupted_node_contribution
            else:
                patched_activation[:,:,node.head_idx, :] = 0
        else:
            if method == "patching":
                patched_activation = corrupted_node_contribution
            else:
                patched_activation = 0
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
    else:
        node_id = f"L{node.layer}-{node.name.split('_', 1)[1]}"
    return node_id

def save_removed_components(model_name, ablated_nodes: Optional[List[Node]] = None, ablated_edges: Optional[List[Edge]] = None):
    # Store removed edges/nodes metadata in a json file
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to root, then into the save folder
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    save_dir = os.path.join(ROOT_DIR, "removed_components")
    if ablated_edges is not None and ablated_edges != []:
        params_to_save = {
            "ablated_edges": [
                {
                    "sender": {edge.sender.full_activation: edge.sender.head_idx},
                    "receiver": {edge.receiver.full_activation: edge.receiver.head_idx}
                }
                for edge in ablated_edges
            ]
        }
    elif ablated_nodes is not None and ablated_nodes != []:
        params_to_save = {
            "ablated_nodes": [
                {node.full_activation: node.head_idx}
                for node in ablated_nodes
            ]
        }
    os.makedirs(save_dir, exist_ok=True)

    # Build full file path
    save_path = os.path.join(save_dir, f"{model_name.replace('/', '-')}.json")
    with open(save_path, "w") as f:
        json.dump(params_to_save, f, indent=2)

def load_removed_components(model_name):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to root, then into the save folder
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    path = os.path.join(ROOT_DIR, "removed_components", f"{model_name.replace('/', '-')}.json")
    with open(path, "r") as f:
        params = json.load(f)
    return params

def add_circuit_hooks(model, model_name):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to root, then into the save folder
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    path = os.path.join(ROOT_DIR, "removed_components", f"{model_name.replace('/', '-')}.json")
    with open(path, "r") as f:
        params = json.load(f)
    for node in params["ablated_nodes"]:
        for full_activation, head_idx in node.items():
            if "attn" in full_activation:
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                head_idx = head_idx
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="attention",
                    head_idx=head_idx,
                    full_activation=full_activation
                )
            elif "mlp_out" in full_activation:
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="mlp",
                    full_activation=full_activation
                )
            elif "resid" in full_activation:
                # Residual nodes
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="residual",
                    full_activation=full_activation
                )
            model.add_hook(node.full_activation, create_node_patching_hook(
                method="pruning",
                node=node
            ))
    print(f"Added ablation hooks for nodes: {params['ablated_nodes']}")

def evaluate_pruned_model(model, model_name, test_data):
    edges_to_prune = load_removed_components(model_name)
    for edge in edges_to_prune["ablated_edges"]:

        for full_activation, head_idx in edge.items():
            if "attn" in full_activation:
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                head_idx = head_idx
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="attention",
                    head_idx=head_idx,
                    full_activation=full_activation
                )
            elif "mlp_out" in full_activation:
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="mlp",
                    full_activation=full_activation
                )
            elif "resid" in full_activation:
                # Residual nodes
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="residual",
                    full_activation=full_activation
                )
            model.add_hook(edge.receiver.full_activation, create_node_patching_hook(
                method="pruning",
                node=node
            ))
