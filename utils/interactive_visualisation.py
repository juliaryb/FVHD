import numpy as np
import torch
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from typing import Union
from knn.graph import Graph
import plotly.io as pio

pio.renderers.default = "browser"

def plot_embedding(
    embedding: np.ndarray,
    labels: Union[np.ndarray, torch.Tensor],
    graph: Graph,
    dataset_name: str = "mnist",
):
    """
    Interactive 2D embedding viewer using Plotly.

    Hover over a point to view its index, class label,
    and the label distribution of its high-dimensional neighbors.

    Parameters:
    - embedding: (n_samples, 2) array with 2D coordinates
    - labels: class labels (torch.Tensor or np.ndarray)
    - graph: Graph object with .indexes (k-NN connections in original space)
    - dataset_name: title for the plot
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    n_samples = embedding.shape[0]
    indexes = graph.indexes

    hover_text = []
    for i in range(n_samples):
        # exclude self from neighbors
        neigh_idxs = [j for j in indexes[i] if j != i]
        neigh_labels = labels[neigh_idxs]

        unique, counts = np.unique(neigh_labels, return_counts=True)
        sorted_labels = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)

        neighbor_info = "<br>".join(
            [f"Label {label} → {count} neighbour{'s' if count > 1 else ''}" for label, count in sorted_labels]
        )

        hover_text.append(
            f"<b>Index:</b> {i}<br><b>Label:</b> {labels[i]}<br><b>Neighbours:</b><br>{neighbor_info}"
        )

    # map class labels to qualitative colors
    unique_classes = np.unique(labels)
    class_colors = {cls: qualitative.Plotly[i % len(qualitative.Plotly)] for i, cls in enumerate(unique_classes)}
    point_colors = [class_colors[label] for label in labels]

    fig = go.Figure()

    # main scatter plot
    fig.add_trace(go.Scattergl(
        x=embedding[:, 0],
        y=embedding[:, 1],
        mode='markers',
        marker=dict(
            color=point_colors,
            size=4,
            opacity=0.75,
        ),
        text=hover_text,
        hoverinfo='text',
        name='All points',
    ))

    fig.update_layout(
        title=f"{dataset_name} – Interactive 2D Embedding",
        dragmode='pan',
        height=750,
        width=750,
    )

    fig.show()