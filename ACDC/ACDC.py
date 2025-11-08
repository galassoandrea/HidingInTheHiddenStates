import ast
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from .utils import *
from .evaluation import kl_divergence
import numpy as np
from tqdm import tqdm
from .ComputationalGraph import Node, build_computational_graph


class ACDCNode:
    """
    Automatic Circuit Discovery (ACDC) Algorithm
    for finding minimal circuits responsible for specific tasks.
    Node-level version.
    """

    def __init__(self, model, model_name,
                 mode: str = "greedy",
                 method: str = "patching",
                 threshold: float = 0.1):

        self.model = model
        self.model_name = model_name
        self.threshold = threshold
        self.device = model.cfg.device
        self.mode = mode
        self.method = method

        # Initialize graphs
        self.full_graph = None
        self.circuit = None

        # Cache for model activations and logits
        self.ablated_nodes = []

        # Create dataset
        print("Loading Factuality dataset...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        self.dataset = pd.DataFrame()
        for topic in ["animals", "cities", "elements", "companies", "inventions"]:
            df_path = os.path.join(root_dir, "resources", f"{topic}_clean_corrupted.csv")
            df = pd.read_csv(df_path, nrows=10)
            df["topic"] = topic
            self.dataset = pd.concat([self.dataset, df], ignore_index=True)

    def run(self):
        """ Main method to perform circuit discovery. """

        print(f"Building computational graph for {self.model_name}...")
        granularity = "block"
        self.full_graph = build_computational_graph(self.model, self.model_name, granularity)
        self.circuit = self.full_graph.copy()
        ordered_nodes = self.circuit.topological_sort()

        print(f"Total nodes: {len(ordered_nodes)}")
        print(f"Total edges: {len(self.circuit.edges)}")

        # Pre-tokenize dataset examples
        clean_tokens = self.model.to_tokens(self.dataset['clean_statement'].tolist()).to(self.device)
        corrupted_tokens = self.model.to_tokens(self.dataset['corrupted_statement'].tolist()).to(self.device)
        # Pad to max length
        max_len = max(clean_tokens.shape[1], corrupted_tokens.shape[1])
        clean_tokens = torch.nn.functional.pad(
            clean_tokens,
            (0, max_len - clean_tokens.shape[1]),
            value=self.model.tokenizer.pad_token_id
        )
        corrupted_tokens = torch.nn.functional.pad(
            corrupted_tokens,
            (0, max_len - corrupted_tokens.shape[1]),
            value=self.model.tokenizer.pad_token_id
        )

        # Collect clean and corrupted reference outputs and caches
        batch_size = 8
        clean_logits_list = []
        corrupted_caches_list = [] if self.method == "patching" else None
        act_names = get_activations_name(self.model_name, self.model.cfg.n_layers, target="node")
        with (torch.no_grad()):
            for i in range(0, len(self.dataset), batch_size):
                batch_clean = clean_tokens[i:i + batch_size]
                batch_corrupted = corrupted_tokens[i:i + batch_size]

                # Run a forward pass and collect clean logits (we don't need clean caches for node patching)
                batch_logits = self.model(batch_clean, return_type="logits")
                clean_logits_list.append(batch_logits.cpu())

                if self.method == "patching":
                    # Run a forward pass and collect corrupted caches (these are needed for node patching)
                    _, batch_corrupted_cache = self.model.run_with_cache(
                        batch_corrupted,
                        return_type="logits",
                        names_filter=act_names
                    )
                    corrupted_caches_list.append({k: v.cpu() for k, v in batch_corrupted_cache.items()})

                # Clear GPU cache
                torch.cuda.empty_cache()

            # Concatenate results
            clean_logits = torch.cat(clean_logits_list, dim=0)
            if self.method == "patching":
                corrupted_caches = {k: torch.cat([cache[k] for cache in corrupted_caches_list], dim=0)
                                    for k in corrupted_caches_list[0].keys()}
        # Precompute node contributions for all examples
        if self.method == "patching":
            _, corrupted_node_contributions = precompute_node_contributions(self.full_graph, self.device, granularity, corrupted_caches=corrupted_caches)
            del corrupted_caches
        # Clear gpu
        torch.cuda.empty_cache()
        self.circuit_discovery(ordered_nodes, clean_tokens, clean_logits, corrupted_node_contributions=corrupted_node_contributions if self.method == "patching" else None)

        # Clear gpu
        torch.cuda.empty_cache()

        # Compute final KL divergence (add all hooks at the same time)
        kl_score = get_final_performance(
            model=self.model,
            clean_tokens=clean_tokens,
            clean_labels=self.dataset['label'].tolist(),
            clean_logits=clean_logits,
            ablated_nodes=self.ablated_nodes
        )

        print(f"Final KL divergence: {kl_score:.6f}")
        save_removed_components(self.model_name, self.ablated_nodes)
        return self.circuit

    def circuit_discovery(self, ordered_nodes, clean_tokens, clean_logits, corrupted_node_contributions: Optional = None):
        print(f"Starting node evaluation with threshold: {self.threshold}")
        nodes_removed_this_iter = 1
        total_nodes_removed = 0
        batch_size = 8
        while nodes_removed_this_iter > 0:
            nodes_removed_this_iter = 0
            # Run circuit discovery based on the model
            if "pythia" in self.model_name:
                # Iterate through nodes and ablate them
                for node in tqdm(list(ordered_nodes), desc="Evaluating nodes"):
                    if node.name in ['hook_resid_pre', 'hook_resid_post', 'hook_mlp_out', 'hook_result']:
                        node_id = get_node_id(node)
                        print(f"Evaluating node: {node_id}")
                        # Temporarily remove the node
                        kl_divs = []
                        for i in range(0, len(self.dataset), batch_size):
                            if self.method == "patching":
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens[i:i + batch_size],
                                    i=i,
                                    node_to_patch=node,
                                    corrupted_node_contributions=corrupted_node_contributions,
                                    ablated_nodes=self.ablated_nodes
                                )
                            else:
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens[i:i + batch_size],
                                    i=i,
                                    node_to_patch=node,
                                    ablated_nodes=self.ablated_nodes
                                )
                            kl_div = kl_divergence(clean_logits[i:i+batch_size].to(self.device), patched_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_node(node)
                            ordered_nodes.remove(node)
                            nodes_removed_this_iter += 1
                            self.ablated_nodes.append(node)
                            print(f"Node removed.")
                total_nodes_removed += nodes_removed_this_iter
                print(f"Nodes removed this iteration: {nodes_removed_this_iter}")
            else:
                # Iterate through nodes and ablate them
                for node in tqdm(list(ordered_nodes), desc="Evaluating nodes"):
                    if node.name in ['hook_resid_pre', 'hook_resid_mid', 'hook_resid_post', 'hook_mlp_out',
                                     'hook_result']:
                        node_id = get_node_id(node)
                        print(f"Evaluating node: {node_id}")
                        # Temporarily remove the node
                        kl_divs = []
                        for i in range(0, len(self.dataset), batch_size):
                            if self.method == "patching":
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens[i:i + batch_size],
                                    i=i,
                                    node_to_patch=node,
                                    corrupted_node_contributions=corrupted_node_contributions,
                                    ablated_nodes=self.ablated_nodes
                                )
                            else:
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens[i:i + batch_size],
                                    i=i,
                                    node_to_patch=node,
                                    ablated_nodes=self.ablated_nodes
                                )
                            kl_div = kl_divergence(clean_logits[i:i+batch_size].to(self.device), patched_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_node(node)
                            ordered_nodes.remove(node)
                            nodes_removed_this_iter += 1
                            self.ablated_nodes.append(node)
                            print(f"Node removed.")
                total_nodes_removed += nodes_removed_this_iter
                print(f"Nodes removed this iteration: {nodes_removed_this_iter}")

        # Print summary of results
        print(f"\nCircuit discovery complete!")
        print(f"Nodes removed: {total_nodes_removed}")
        print(f"Final circuit nodes: {len(self.circuit.nodes)}")

    def run_with_node_patching(
            self,
            inputs: torch.Tensor,
            i,
            node_to_patch: Node,
            corrupted_node_contributions: Optional = None,
            ablated_nodes: Optional[List[Node]] = None
    ) -> torch.Tensor:
        """Run model with node patching."""
        batch_size = inputs.shape[0]

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are nodes previously patched and, if so, re-add the hooks for them
        if ablated_nodes and self.mode == "greedy":
            for node in ablated_nodes:
                hook = create_node_patching_hook(
                    method="pruning",
                    node=node
                )
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(node.full_activation, hook)

        # Create patching hook
        node_id = get_node_id(node_to_patch)
        patching_hook = create_node_patching_hook(
            self.method,
            node_to_patch,
            corrupted_node_contributions[node_id][i:i+batch_size] if corrupted_node_contributions else None
        )

        # Register hook on the node
        if hasattr(self.model, 'add_hook'):
            self.model.add_hook(node_to_patch.full_activation, patching_hook)

        # Run forward pass
        with torch.no_grad():
            patched_logits = self.model(inputs)

        return patched_logits

    def evaluate_circuit(self, test_data):
        all_preds = []
        all_probs = []
        statements, labels = test_data
        # Tokenize dataset
        clean_tokens = self.model.to_tokens(statements).to(self.device)
        # Pad to max length
        max_len = max(clean_tokens.shape[1])
        clean_tokens = torch.nn.functional.pad(
            clean_tokens,
            (0, max_len - clean_tokens.shape[1]),
            value=self.model.tokenizer.pad_token_id
        )
        print("Evaluating circuit on test data...")
        with torch.no_grad():
            # Cache activations and keep only needed ones
            act_names = get_activations_name(self.model_name, self.model.cfg.n_layers, target="edge")
            _, clean_caches = self.model.run_with_cache(clean_tokens, return_type="logits", names_filter=act_names)
        # Precompute node contributions for all examples
        clean_node_contributions, _ = precompute_node_contributions(self.full_graph, "pruning", self.device, "block", clean_caches)
        # Clear memory from caches since we don't need them anymore
        del clean_caches
        # Clear gpu
        torch.cuda.empty_cache()
        # Iterate over examples, run with ablation hooks and compute metrics
        for i in range(len(clean_tokens)):
            # Add all hooks for patched edges
            add_all_hooks(self.model, i, clean_node_contributions, ablated_nodes=self.ablated_nodes)
            # Run forward pass
            with torch.no_grad():
                inputs = clean_tokens[i]
                patched_logits = self.model(inputs)
                logits = patched_logits[0, -1, :]  # last token's logits (shape: [2])
                probs = torch.softmax(logits, dim=-1)[1].item()  # class 1 probability
                pred = int(probs > 0.5)
                all_probs.append(probs)
                all_preds.append(pred)
        # Reset all hooks
        self.model.reset_hooks()
        # Compute metrics
        accuracy = accuracy_score(labels, all_preds)
        roc_auc = roc_auc_score(labels, all_probs)
        print(f"Accuracy of the circuit on the test set: {accuracy:.4f}")
        print(f"ROC-AUC of the circuit on the test set: {roc_auc:.4f}")


class ACDCEdge:
    """
    Automatic Circuit Discovery (ACDC) Algorithm
    for finding minimal circuits responsible for specific tasks.
    Edge-level version.
    """

    def __init__(self, model, model_name,
                 mode: str = "greedy",
                 method: str = "patching", embedding_dataset: pd.DataFrame = None,
                 threshold: float = 0.05):

        self.model = model
        self.model_name = model_name
        self.threshold = threshold
        self.device = model.cfg.device
        self.mode = mode
        self.method = method

        # Initialize graphs
        self.full_graph = None
        self.circuit = None

        self.ablated_edges = []

        # Create dataset
        print("Loading Factuality dataset...")
        if embedding_dataset is not None:
            self.dataset = embedding_dataset
            self.tokenize = False
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            self.dataset = pd.DataFrame()
            for topic in ["animals", "cities", "elements", "companies", "inventions"]:
                df_path = os.path.join(root_dir, "resources", f"{topic}_clean_corrupted.csv")
                df = pd.read_csv(df_path, nrows=10)
                df["topic"] = topic
                self.dataset = pd.concat([self.dataset, df], ignore_index=True)
            self.tokenize = True

    def run(self):
        """ Main method to perform circuit discovery. """

        print(f"Building computational graph for {self.model_name}...")
        granularity = "head"
        self.full_graph = build_computational_graph(self.model, self.model_name, granularity)
        self.circuit = self.full_graph.copy()
        ordered_nodes = self.circuit.topological_sort()

        print(f"Total nodes: {len(ordered_nodes)}")
        print(f"Total edges: {len(self.circuit.edges)}")

        # Pre-tokenize dataset examples
        # Check the type of clean_statement and corrupted_statement columns and tokenize only if they are strings
        if self.tokenize:
            clean_tokens = self.model.to_tokens(self.dataset['clean_statement'].tolist()).to(self.device)
            corrupted_tokens = self.model.to_tokens(self.dataset['corrupted_statement'].tolist()).to(self.device)
            # Pad to max length
            max_len = max(clean_tokens.shape[1], corrupted_tokens.shape[1])
            clean_tokens = torch.nn.functional.pad(
                clean_tokens,
                (0, max_len - clean_tokens.shape[1]),
                value=self.model.tokenizer.pad_token_id
            )
            corrupted_tokens = torch.nn.functional.pad(
                corrupted_tokens,
                (0, max_len - corrupted_tokens.shape[1]),
                value=self.model.tokenizer.pad_token_id
            )
            # Chek type to ensure they are tensors
            if not isinstance(clean_tokens, torch.Tensor):
                raise ValueError("Clean tokens are not a torch.Tensor")
        else:
            clean_tokens = [ast.literal_eval(x) if isinstance(x, str) else x
                          for x in self.dataset['clean_statement'].tolist()]
            corrupted_tokens = [ast.literal_eval(x) if isinstance(x, str) else x
                              for x in self.dataset['corrupted_statement'].tolist()]
            clean_tokens = torch.tensor(clean_tokens).to(self.device)
            corrupted_tokens = torch.tensor(corrupted_tokens).to(self.device)

        # Collect clean and corrupted reference outputs and caches
        act_names = get_activations_name(self.model_name, self.model.cfg.n_layers, target="edge")
        with torch.no_grad():
            if self.tokenize:
                # Cache activations and keep only needed ones
                clean_logits, clean_caches = self.model.run_with_cache(clean_tokens, return_type="logits", names_filter=act_names)
                clean_logits = clean_logits.cpu()
                if self.method == "patching":
                    # Also collect corrupted outputs for activation patching
                    _, corrupted_caches = self.model.run_with_cache(corrupted_tokens, return_type="logits",
                                                                    names_filter=act_names)
            else:
                # In case of embeddings dataset, since the inputs are embeddings,
                # we need to replace the embedding layer activations with our embeddings.
                # To do this, we need to create dummy tokens and use a forward hook to replace the embeddings.
                dummy_tokens = torch.zeros_like(clean_tokens).long().to(self.device)

                def embedding_replacement_hook(activations, hook):
                    return clean_tokens

                # Run with cache, replacing embeddings via hook
                with self.model.hooks(fwd_hooks=[("hook_embed", embedding_replacement_hook)]):
                    clean_logits, clean_caches = self.model.run_with_cache(dummy_tokens)
                clean_logits = clean_logits.cpu()
                if self.method == "patching":
                    # Also collect corrupted outputs for activation patching
                    # Run with cache, replacing embeddings via hook
                    def embedding_replacement_hook(activations, hook):
                        return corrupted_tokens

                    with self.model.hooks(fwd_hooks=[("hook_embed", embedding_replacement_hook)]):
                        _, corrupted_caches = self.model.run_with_cache(dummy_tokens)
        # Precompute node contributions for all examples
        if self.method == "patching":
            clean_node_contributions, corrupted_node_contributions = precompute_node_contributions(self.full_graph, self.device, granularity, clean_caches, corrupted_caches)
        else:
            clean_node_contributions, corrupted_node_contributions = precompute_node_contributions(self.full_graph, self.device, granularity, clean_caches)
        # Clear memory from caches since we don't need them anymore
        del clean_caches
        del corrupted_caches
        # Clear gpu
        torch.cuda.empty_cache()

        self.circuit_discovery(ordered_nodes, clean_tokens, clean_logits, clean_node_contributions, corrupted_node_contributions)

        # Clear gpu
        torch.cuda.empty_cache()

        # Compute final KL divergence (add all hooks at the same time)
        kl_score = get_final_performance(
            model=self.model,
            tokenize=self.tokenize,
            clean_tokens=clean_tokens,
            clean_labels=self.dataset['label'].tolist(),
            clean_logits=clean_logits,
            clean_node_contributions=clean_node_contributions,
            ablated_edges=self.ablated_edges
        )

        print(f"Final KL divergence: {kl_score:.6f}")
        return self.circuit

    def circuit_discovery(self, ordered_nodes, clean_tokens, clean_logits, clean_node_contributions, corrupted_node_contributions):
        print(f"Starting edge evaluation with threshold: {self.threshold}")
        # Iterate through nodes and prune edges
        edges_removed_this_iter = 1
        total_edges_removed = 0
        iteration = 0
        while edges_removed_this_iter > 0:
            edges_removed_this_iter = 0
            iteration += 1
            print(f"--- Starting iteration {iteration} ---")
            for receiver in tqdm(list(ordered_nodes), desc="Evaluating edges"):
                receiver_id = get_node_id(receiver)
                senders = self.circuit.get_senders(receiver).copy()
                if senders != [] and senders is not None:
                    for sender in senders:
                        sender_id = get_node_id(sender)
                        if receiver.name == "hook_q":
                            print(f"Evaluating edge: {sender_id} -> {receiver_id}")
                        else:
                            print(f"Evaluating edge: {sender_id} -> {receiver_id}")
                        edge = Edge(sender, receiver)
                        # Temporarily remove the edge
                        kl_divs = []
                        for i in range(len(clean_tokens)):
                            # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                            if self.method == "patching":
                                patched_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens[i],
                                    i=i,
                                    edge_to_patch=edge,
                                    clean_node_contributions=clean_node_contributions,
                                    corrupted_node_contributions=corrupted_node_contributions,
                                    ablated_edges=self.ablated_edges
                                )
                            else:
                                patched_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens[i],
                                    i=i,
                                    edge_to_patch=edge,
                                    clean_node_contributions=clean_node_contributions,
                                    ablated_edges=self.ablated_edges
                                )
                            kl_div = kl_divergence(clean_logits[i].to(self.device), patched_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_edge(edge)
                            edges_removed_this_iter += 1
                            self.ablated_edges.append(edge)
                            # Only remove sender from ordered_nodes if it has no more receivers
                            if len(self.circuit.get_receivers(edge.sender)) == 0:
                                ordered_nodes.remove(edge.sender)
                            print(f"Edge removed.")
            total_edges_removed += edges_removed_this_iter
            print(f"Edges removed this iteration: {edges_removed_this_iter}")

        # Print summary of results
        print(f"\nCircuit discovery complete!")
        print(f"Edges removed: {total_edges_removed}")
        print(f"Final circuit edges: {len(self.circuit.edges)}")

    def run_with_edge_patching(
            self,
            inputs: torch.Tensor,
            i,
            edge_to_patch: Edge,
            clean_node_contributions,
            corrupted_node_contributions: Optional = None,
            ablated_edges: Optional[List[Edge]] = None
    ) -> torch.Tensor:
        """Run model with edge patching."""

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are edges previously patched and, if so, re-add the hooks for them
        if ablated_edges and self.mode == "greedy":
            for edge in ablated_edges:
                node_id = get_node_id(edge.sender)
                hook = create_edge_patching_hook(
                    method="pruning",
                    node=edge.receiver,
                    clean_sender_contribution=clean_node_contributions[node_id][i].unsqueeze(0)
                )
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(edge.receiver.full_activation, hook)

        # Create patching hook
        node_id = get_node_id(edge_to_patch.sender)
        patching_hook = create_edge_patching_hook(
            method=self.method,
            node=edge_to_patch.receiver,
            clean_sender_contribution=clean_node_contributions[node_id][i].unsqueeze(0),
            corrupted_sender_contribution=corrupted_node_contributions[node_id][i].unsqueeze(0) if corrupted_node_contributions else None
        )

        if not self.tokenize:
            # In case of embeddings dataset, since the inputs are embeddings,
            # we need to replace the embedding layer activations with our embeddings.
            # To do this, we need to create dummy tokens and use a forward hook to replace the embeddings.
            dummy_tokens = torch.zeros_like(inputs).long().to(self.device)
            def embedding_replacement_hook(activations, hook):
                return inputs
            # Register hook on the node
            if hasattr(self.model, 'add_hook'):
                self.model.add_hook("hook_embed", embedding_replacement_hook)
                self.model.add_hook(edge_to_patch.receiver.full_activation, patching_hook)
            # Run forward pass
            with torch.no_grad():
                patched_logits = self.model(dummy_tokens)
        else:
            # Register hook on the node
            if hasattr(self.model, 'add_hook'):
                self.model.add_hook(edge_to_patch.receiver.full_activation, patching_hook)

            # Run forward pass
            with torch.no_grad():
                patched_logits = self.model(inputs)

        return patched_logits

    def evaluate_circuit(self, test_data):
        all_preds = []
        all_probs = []
        statements, labels = test_data
        # Tokenize dataset if not already tokenized
        if self.tokenize:
            clean_tokens = self.model.to_tokens(statements).to(self.device)
            # Pad to max length
            max_len = max(clean_tokens.shape[1])
            clean_tokens = torch.nn.functional.pad(
                clean_tokens,
                (0, max_len - clean_tokens.shape[1]),
                value=self.model.tokenizer.pad_token_id
            )
        else:
            clean_tokens = [ast.literal_eval(x) if isinstance(x, str) else x
                            for x in statements]
            clean_tokens = torch.tensor(np.array(clean_tokens)).to(self.device)
        print("Evaluating circuit on test data...")
        with torch.no_grad():
            if self.tokenize:
                # Cache activations and keep only needed ones
                act_names = get_activations_name(self.model_name, self.model.cfg.n_layers, target="edge")
                _, clean_caches = self.model.run_with_cache(clean_tokens, return_type="logits", names_filter=act_names)
            else:
                # In case of embeddings dataset, since the inputs are embeddings,
                # we need to replace the embedding layer activations with our embeddings.
                # To do this, we need to create dummy tokens and use a forward hook to replace the embeddings.
                dummy_tokens = torch.zeros_like(clean_tokens).long().to(self.device)

                def embedding_replacement_hook(activations, hook):
                    return clean_tokens

                # Run with cache, replacing embeddings via hook
                with self.model.hooks(fwd_hooks=[("hook_embed", embedding_replacement_hook)]):
                    _, clean_caches = self.model.run_with_cache(dummy_tokens)

        # Precompute node contributions for all examples
        clean_node_contributions, _ = precompute_node_contributions(self.full_graph, "pruning", self.device, "head", clean_caches)
        # Clear memory from caches since we don't need them anymore
        del clean_caches
        # Clear gpu
        torch.cuda.empty_cache()
        # Iterate over examples, run with ablation hooks and compute metrics
        for i in range(len(clean_tokens)):
            if not self.tokenize:
                inputs = torch.zeros_like(clean_tokens[i]).long().to(self.device)
                def embedding_replacement_hook(activations, hook):
                    return clean_tokens[i]
                # Register embedding replacement hook on the node
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook("hook_embed", embedding_replacement_hook)
            # Add all hooks for patched edges
            add_all_hooks(self.model, i, clean_node_contributions, ablated_edges=self.ablated_edges)
            # Run forward pass
            with torch.no_grad():
                patched_logits = self.model(inputs)
                logits = patched_logits[0, -1, :]  # last token's logits (shape: [2])
                probs = torch.softmax(logits, dim=-1)[1].item()  # class 1 probability
                pred = int(probs > 0.5)
                all_probs.append(probs)
                all_preds.append(pred)
        # Reset all hooks
        self.model.reset_hooks()
        # Compute metrics
        accuracy = accuracy_score(labels, all_preds)
        roc_auc = roc_auc_score(labels, all_probs)
        print(f"Accuracy of the circuit on the test set: {accuracy:.4f}")
        print(f"ROC-AUC of the circuit on the test set: {roc_auc:.4f}")


