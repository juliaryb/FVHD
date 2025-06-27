import ssl
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from imblearn.under_sampling import TomekLinks

from fvhd import FVHD
from knn import Graph, NeighborConfig, NeighborGenerator
from utils import metrics, interactive_visualisation


def setup_ssl():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def load_dataset(name: str, n_samples: Optional[int] = None):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    elif name == "emnist":
        dataset = torchvision.datasets.EMNIST(
            "emnist", split="balanced", train=True, download=True
        )
    elif name == "fmnist":
        dataset = torchvision.datasets.FashionMNIST(
            "fashionMNIST", train=True, download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X = dataset.data[:n_samples]
    N = len(X) if n_samples is None else n_samples
    X = X.reshape(N, -1) / 255.0

    from sklearn.decomposition import PCA

    pca = PCA(n_components=50)
    X = torch.tensor(pca.fit_transform(X), dtype=torch.float32)

    Y = dataset.targets[:n_samples]
    return X, Y


def create_or_load_graph(X: torch.Tensor, nn: int) -> tuple[Graph, Graph]:
    config = NeighborConfig(metric="euclidean")
    df = pd.DataFrame(X.numpy())
    generator = NeighborGenerator(df=df, config=config)
    return generator.run(nn=nn)


def visualize_embeddings(x: np.ndarray, y: torch.Tensor, dataset_name: str):
    # plt.switch_backend("TkAgg")
    plt.figure(figsize=(8, 8))
    plt.title(f"{dataset_name} 2d visualization")

    unique_labels = np.unique(y)
    for i in range(len(unique_labels)):
        points = x[y == i]
        plt.scatter(
            points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5
        )
    plt.legend()
    plt.show()


def prep_data_and_graphs(dataset_name: str = "mnist", NN: int = 5, connected: bool = False, undersample: bool = False) -> tuple[torch.Tensor, torch.Tensor, Graph, Graph]:
    X, Y = load_dataset(dataset_name)

    print(f"dataset size: {len(X)}")

    if connected:
        graph, mutual_graph = create_or_load_graph(X, NN)
        connected_indices = graph._get_connected_components()
        X, Y = X[connected_indices], Y[connected_indices]
        print(f"reduced dataset size: {len(X)}")

    if undersample:
        tl = TomekLinks()
        X, Y = tl.fit_resample(X, Y)
        X, Y = torch.tensor(X), torch.tensor(Y)
        print(f"reduced dataset size: {len(X)}")

    graph, mutual_graph = create_or_load_graph(X, NN)

    return X, Y, graph, mutual_graph

if __name__ == "__main__":
    setup_ssl()

    dataset_name: str = "mnist"
    connected: bool = True
    undersample: bool = False
    NN: int = 4

    fvhd = FVHD(
        n_components=2,
        nn=NN,
        rn=2,
        c=0.2,
        eta=0.2,
        optimizer=None,
        optimizer_kwargs={"lr": 0.1},
        epochs=2000,
        device="cpu",
        velocity_limit=True,
        autoadapt=True,
        mutual_neighbors_epochs=300
    )

    X, Y, graph, mutual_graph = prep_data_and_graphs(dataset_name, NN, connected, undersample)

    # embeddings = fvhd.fit_transform(X, [graph, mutual_graph])
    # visualize_embeddings(embeddings, Y, dataset_name)
    # cf_values, _ = metrics.visualise_cf_scores(embeddings, Y, dataset_name)
    # interactive_visualisation.plot_embedding(embeddings, Y, graph, dataset_name)

    cf_all_runs = []
    for i in range(5):  # 5 repeated runs
        embedding = fvhd.fit_transform(X, [graph, mutual_graph])
        cf_values, _ = metrics.compute_cf(embedding, Y)
        cf_all_runs.append(cf_values)

    visualize_embeddings(embedding, Y, dataset_name)

    metrics.visualise_cf_multiple_runs(cf_all_runs, method_name="MNIST NN=5")