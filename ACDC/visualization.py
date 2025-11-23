import json
import os
from typing import List, Tuple, Set, Dict, Any
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from ACDC.evaluation import jaccard_similarity


def visualize_computational_graph(
        graph,
        title: str = "Circuit Discovery: Computational Graph",
        width: int = 2000,
        height: int = 1400,
        show_edge_labels: bool = False
):
    """ Visualize a computational graph using Plotly to get an interactive visualization. """

    # Define color scheme for different component types
    color_map = {
        'embedding': '#FF6B6B',  # Red
        'residual_pre': '#4ECDC4',  # Teal
        'residual_mid': '#45B7D1',  # Light Blue
        'residual_post': '#96CEB4',  # Green
        'attention': '#FFEAA7',  # Yellow
        'mlp': '#DDA0DD'  # Plum
    }

    # Get number of layers and heads from the model configuration
    n_layers = graph.model.cfg.n_layers
    layers = list(range(n_layers + 1))
    n_heads = graph.model.cfg.n_heads

    # Calculate positions for nodes
    node_positions = {}

    # Define vertical ordering of component types within each layer
    component_order = {
        'embedding': 0,
        'residual_pre': 1,
        'attention': 2,
        'residual_mid': 3,
        'mlp': 4,
        'residual_post': 5
    }

    # Assign positions
    for node in graph.nodes:
        layer_idx = layers.index(node.layer)
        x_pos = layer_idx * 2  # Horizontal spacing between layers

        # Vertical position based on component type
        if node.component_type == 'residual':
            if 'pre' in node.name:
                base_y = component_order['residual_pre'] * 3
            elif 'mid' in node.name:
                base_y = component_order['residual_mid'] * 3
            else:
                base_y = component_order['residual_post'] * 3
        else:
            base_y = component_order.get(node.component_type, 3) * 3

        # For attention heads, spread them horizontally within their slot
        if node.component_type == 'attention' and node.head_idx is not None:
            # Offset x position for each head
            head_offset = (node.head_idx - (n_heads - 1) / 2) * 0.15
            x_pos += head_offset

        node_positions[node] = (x_pos, base_y)

    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_hover_text = []

    for edge in graph.edges:
        x0, y0 = node_positions[edge.sender]
        x1, y1 = node_positions[edge.receiver]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover_text.append(f"{edge.sender.name} → {edge.receiver.name}")

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.8, color='#888'),
        hoverinfo='skip',
        showlegend=False,
        opacity=0.5
    )

    # Prepare node traces (one per component type for legend)
    node_traces = {}

    for comp_type in component_order.keys():
        node_traces[comp_type] = {
            'x': [],
            'y': [],
            'text': [],
            'hover_text': [],
            'marker_size': []
        }

    for node in graph.nodes:
        x, y = node_positions[node]
        if node.component_type == 'residual':
            if 'pre' in node.name:
                comp_type = 'residual_pre'
            elif 'mid' in node.name:
                comp_type = 'residual_mid'
            else:
                comp_type = 'residual_post'
        else:
            comp_type = node.component_type

        node_traces[comp_type]['x'].append(x)
        node_traces[comp_type]['y'].append(y)
        node_traces[comp_type]['text'].append(node.name)

        # Create hover text
        in_degree = len(graph.reverse_adjacency[node]) if node in graph.reverse_adjacency else 0
        out_degree = len(graph.adjacency[node]) if node in graph.adjacency else 0
        hover_text = (
            f"<b>{node.name}</b><br>"
            f"Type: {node.component_type}<br>"
            f"Layer: {node.layer}<br>"
            f"In-degree: {in_degree}<br>"
            f"Out-degree: {out_degree}"
        )
        if node.head_idx is not None:
            hover_text += f"<br>Head: {node.head_idx}"

        node_traces[comp_type]['hover_text'].append(hover_text)

        # Size based on connectivity
        size = 10 + (in_degree + out_degree) * 2
        node_traces[comp_type]['marker_size'].append(min(size, 30))

    # Create figure
    fig = go.Figure()

    # Add edge trace
    fig.add_trace(edge_trace)

    # Add node traces
    for comp_type, trace_data in node_traces.items():
        if len(trace_data['x']) > 0:  # Only add if there are nodes of this type
            fig.add_trace(go.Scatter(
                x=trace_data['x'],
                y=trace_data['y'],
                mode='markers+text',
                name=comp_type.replace('_', ' ').title(),
                text=trace_data['text'],
                hovertext=trace_data['hover_text'],
                hoverinfo='text',
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                marker=dict(
                    size=trace_data['marker_size'],
                    color=color_map.get(comp_type, '#95a5a6'),
                    line=dict(width=2, color='white'),
                    symbol='circle'
                )
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Model: {graph.model_name} | "
                 f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}</sub>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        width=width,
        height=height,
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            title="Layer",
            tickmode='array',
            tickvals=list(range(0, n_layers * 2, 2)),
            ticktext=[f"L{i}" for i in layers]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=""
        ),
        margin=dict(l=20, r=20, t=100, b=40)
    )

    # Add annotations for component types
    for comp_type, order_idx in component_order.items():
        if any(n.component_type == comp_type for n in graph.nodes):
            fig.add_annotation(
                x=-0.5,
                y=order_idx * 3,
                text=comp_type.replace('_', ' ').title(),
                showarrow=False,
                xanchor='right',
                font=dict(size=10, color=color_map.get(comp_type, '#95a5a6')),
                bgcolor='white',
                bordercolor=color_map.get(comp_type, '#95a5a6'),
                borderwidth=1,
                borderpad=4
            )
    fig.show()
    fig.write_html("computational_graph.html")

def visualize_computational_graph_hierarchical(
        graph,
        title: str = "Circuit Discovery: Hierarchical View",
        width: int = 1600,
        height: int = 1200
):
    """ Alternative visualization function to get a more hierarchical, top-down visualization."""
    # Use Plotly's built-in tree layout approximation
    color_map = {
        'embedding': '#FF6B6B',
        'residual': '#4ECDC4',
        'residual_mid': '#45B7D1',
        'residual_post': '#96CEB4',
        'attention': '#FFEAA7',
        'mlp': '#DDA0DD'
    }
    # Build position using layer-based vertical layout
    n_layers = graph.model.cfg.n_layers
    layers = list(range(n_layers + 1))
    node_positions = {}
    # Group nodes by layer and type
    layer_groups = defaultdict(lambda: defaultdict(list))
    for node in graph.nodes:
        layer_groups[node.layer][node.component_type].append(node)
    # Assign positions
    for layer_idx, layer in enumerate(layers):
        y_base = -layer_idx * 4  # Vertical spacing
        groups = layer_groups[layer]
        all_nodes_in_layer = []
        for comp_type in ['embedding', 'residual', 'attention' 'mlp']:
            if comp_type == 'residual':
                for sub_type in ['residual_pre', 'residual_mid', 'residual_post']:
                    all_nodes_in_layer.extend(groups[sub_type])
            else:
                all_nodes_in_layer.extend(groups[comp_type])
        n_nodes = len(all_nodes_in_layer)
        for i, node in enumerate(all_nodes_in_layer):
            x_pos = (i - (n_nodes - 1) / 2) * 1.5
            node_positions[node] = (x_pos, y_base)
    # Create visualization similar to first function but with vertical layout
    edge_x, edge_y = [], []
    for edge in graph.edges:
        x0, y0 = node_positions[edge.sender]
        x1, y1 = node_positions[edge.receiver]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='skip',
        showlegend=False,
        opacity=0.4
    ))
    # Add nodes
    for comp_type in color_map.keys():
        if comp_type in ['residual_pre', 'residual_mid', 'residual_post']:
            nodes_of_type = [n for n in graph.nodes if n.component_type == 'residual' and comp_type.split('_')[1] in n.name]
        else:
            nodes_of_type = [n for n in graph.nodes if n.component_type == comp_type]
        if not nodes_of_type:
            continue
        xs = [node_positions[n][0] for n in nodes_of_type]
        ys = [node_positions[n][1] for n in nodes_of_type]
        texts = [n.name for n in nodes_of_type]
        hovers = []
        for n in nodes_of_type:
            in_deg = len(graph.reverse_adjacency[n])
            out_deg = len(graph.adjacency[n])
            hover = f"<b>{n.name}</b><br>Layer: {n.layer}<br>In: {in_deg} | Out: {out_deg}"
            hovers.append(hover)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            name=comp_type.replace('_', ' ').title(),
            text=texts,
            hovertext=hovers,
            hoverinfo='text',
            textposition="middle center",
            textfont=dict(size=8),
            marker=dict(
                size=15,
                color=color_map[comp_type],
                line=dict(width=2, color='white')
            )
        ))
    fig.update_layout(
        title=f"{title}<br><sub>{graph.model_name}</sub>",
        showlegend=True,
        width=width,
        height=height,
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def plot_circuit_convergence(experiments):
    all_experiments: List[Tuple[str, List[str]]] = []
    for n_samples, components in experiments.items():
        run_id = f"N={n_samples}"
        all_experiments.append((run_id, components))

    # Calculate Jaccard Scores
    # Get the *final* set of removed components (from the last experiment)
    final_set = set(all_experiments[-1][1])

    plot_data = []

    for exp in all_experiments:
        current_set = set(exp[1])
        samples = exp[0]

        # Compare the current set to the final, stable set
        score = jaccard_similarity(current_set, final_set)

        plot_data.append({
            "samples": samples,
            "jaccard_index": score
        })

    # Convert to DataFrame for easy plotting
    plot_df = pd.DataFrame(plot_data)

    # Plot the Line Chart

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=plot_df,
        x="samples",
        y="jaccard_index",
        marker='o',
        markersize=8
    )

    plt.ylim(0, 1.1)  # Set Y-axis from 0.0 to 1.1 (for padding)
    plt.title('Convergence of Discovered Circuits', fontsize=16)
    plt.xlabel('Number of Samples Used for Circuit Discovery', fontsize=12)
    plt.ylabel('Jaccard Similarity to Final Set (N=100)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add a horizontal line at 1.0 for reference
    plt.axhline(y=1.0, color='red', linestyle='--', label='Perfect Convergence (1.0)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_circuit_discovery_heatmap(experiments: Dict[int, List[str]], visualization_mode):

    # Transform data into the format needed for plotting:
    # [ ("Run 1 (N=20)", ["L6-Head2", "L6-Head1"]), ... ]
    all_experiments: List[Tuple[str, List[str]]] = []
    if visualization_mode == "samples":
        for n_samples, components in experiments.items():
            run_id = f"N={n_samples}"
            all_experiments.append((run_id, components))
    else: # threshold visualization
        for threshold, components in experiments.items():
            run_id = f"T={threshold}"
            all_experiments.append((run_id, components))
    # Prepare the Data Grid
    # Get all experiment IDs in the new sorted order
    experiment_ids = [exp_id for exp_id, removals in all_experiments]

    # Find all unique components that were *ever* removed
    all_unique_removed: Set[str] = set()
    for _, removals in all_experiments:
        all_unique_removed.update(removals)

    # Sort the components for a stable Y-axis
    sorted_unique_components = sorted(list(all_unique_removed))

    if not sorted_unique_components:
        print("No removed components found in any experiment. Nothing to plot.")
        return

    # Create the empty 0-filled DataFrame
    heatmap_df = pd.DataFrame(
        0,
        index=sorted_unique_components,
        columns=experiment_ids
    )

    # Fill the Grid
    # Populate the DataFrame. 1 = Removed, 0 = Kept.
    for exp_id, removals in all_experiments:
        for component in removals:
            if component in heatmap_df.index:
                heatmap_df.loc[component, exp_id] = 1

    # Plot the Heatmap
    cmap = sns.color_palette(["#EAEAEB", "#B00000"])  # [Kept, Removed]

    fig_height = max(6, int(len(sorted_unique_components) * 0.4))
    fig_width = max(10, len(experiment_ids) * 1)

    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        heatmap_df,
        annot=False,
        cmap=cmap,
        linewidths=.5,
        linecolor='lightgrey',
        cbar=False,
        vmin=0,
        vmax=1
    )

    # Add custom legend
    legend_elements = [
        Patch(facecolor="#B00000", edgecolor='black', label='Component Removed'),
        Patch(facecolor="#EAEAEB", edgecolor='black', label='Component Kept')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))

    plt.title('Component Removal by Experiment', fontsize=16)
    if visualization_mode == "samples":
        plt.xlabel('Increasing n samples for circuit discovery', fontsize=12)
    else: # threshold
        plt.xlabel('Increasing threshold for circuit discovery', fontsize=12)
    plt.ylabel('Component', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_scores_by_threshold(scores_per_threshold):

    metric_names = ['Accuracy', 'ROC-AUC', 'NLL']

    # Convert dictionary to DataFrame in long format for seaborn
    data = []
    for threshold, scores in scores_per_threshold.items():
        for metric_name, score in zip(metric_names, scores):
            data.append({
                'Threshold': threshold,
                'Score': score,
                'Metric': metric_name
            })

    df = pd.DataFrame(data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x='Threshold', y='Score', hue='Metric',
                 marker='o', ax=ax)

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics vs Threshold', fontsize=14)
    ax.legend(title='Metric', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_scores_by_n_samples(df_results):

    # Convert from wide format (separate columns) to long format for Seaborn
    df_long = df_results.melt(
        id_vars=['n_samples'],
        value_vars=['accuracy', 'roc_auc'],
        var_name='Metric',
        value_name='Score'
    )

    # Rename metrics values for legend
    df_long['Metric'] = df_long['Metric'].replace({
        'accuracy': 'Accuracy',
        'roc_auc': 'ROC-AUC'
    })

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_long,
        x='n_samples',
        y='Score',
        hue='Metric',
        marker='o',
        ax=ax
    )
    ax.set_xlabel('N samples', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Circuit performance at different number of samples for circuit discovery', fontsize=14)
    ax.legend(title='Metric', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_results_multiple_samples(model_name, threshold):
    nodes_removed_per_samples = {}

    # Load file with experiments data
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to root, then into the save folder
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    path = os.path.join(ROOT_DIR, "removed_components", f"{model_name.replace('/', '-')}-t{threshold}.json")
    with open(path, "r") as f:
        params = json.load(f)
    experiments_data = params["experiments"]
    for experiment in experiments_data:
        n_samples = experiment["samples"]
        nodes = experiment["components"]
        nodes_removed_per_samples[n_samples] = nodes

    # Plot heatmap of removed components
    plot_circuit_discovery_heatmap(nodes_removed_per_samples, visualization_mode="samples")

    # Plot line-chart showing circuits convergence
    plot_circuit_convergence(nodes_removed_per_samples)


