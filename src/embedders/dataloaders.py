import torch
from torchtyping import TensorType as TT
from typing import Tuple
import networkx as nx
import pandas as pd
import numpy as np
import shlex
from pathlib import Path
from scipy.io import mmread
import anndata
import gzip
import pickle


def _top_cc_dists(G: nx.Graph, to_undirected: bool = True) -> Tuple[np.ndarray, list]:
    """Returns the distances between the top connected component of a graph"""
    if to_undirected:
        G = G.to_undirected()
    top_cc = max(nx.connected_components(G), key=len)
    print(f"Top CC has {len(top_cc)} nodes; original graph has {G.number_of_nodes()} nodes.")
    return nx.floyd_warshall_numpy(G.subgraph(top_cc)), list(top_cc)


def load_cities(
    cities_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "cities" / "cities.txt",
) -> Tuple[TT["n_points", "n_points"], None, None]:
    dists_flattened = []
    with open(cities_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            dists_flattened += [float(x) for x in line.split()]

    cities_dists = torch.tensor(dists_flattened).reshape(312, 312)

    return cities_dists, None, None


def load_cs_phds(
    cs_phds_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "cs_phds.txt",
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], TT["n_points", "n_points"]]:
    G = nx.Graph()

    with open(cs_phds_path, "r") as f:
        lines = f.readlines()

    # Add nodes
    for line in lines[2:1027]:
        num, name, v1, v2, v3 = shlex.split(line)
        num = int(num)
        v1, v2, v3 = float(v1), float(v2), float(v3)
        G.add_node(num, attr={"name": line, "val1": v1, "val2": v2, "val3": v3})

    # Add edges
    for line in lines[1028:2071]:
        n1, n2, weight = shlex.split(line)
        n1, n2 = int(n1), int(n2)
        weight = float(weight)
        G.add_edge(n1, n2, weight=weight)

    # Add years
    for i, line in enumerate(lines[2075:-1]):
        year = int(line.strip())
        G.nodes[i + 1]["year"] = year  # They're 1-indexed

    phd_dists, idx = _top_cc_dists(G)
    labels = [G.nodes[i]["year"] for i in idx]

    return torch.tensor(phd_dists), torch.tensor(labels), torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_facebook():
    raise NotImplementedError


def load_power():
    raise NotImplementedError


def load_polblogs(
    polblogs_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "polblogs" / "polblogs.mtx",
    polblogs_labels_path: str = Path(__file__).parent.parent.parent
    / "data"
    / "graphs"
    / "polblogs"
    / "polblogs_labels.tsv",
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], TT["n_points", "n_points"]]:
    # Load the graph
    G = nx.from_scipy_sparse_array(mmread(polblogs_path))
    dists, idx = _top_cc_dists(G)

    # Load the labels
    polblogs_labels = pd.read_table(polblogs_labels_path, header=None)[0]

    # Filter to match G
    polblogs_labels = polblogs_labels[idx].tolist()

    return torch.tensor(dists), torch.tensor(polblogs_labels), torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def _load_network_repository(
    edges_path, labels_path
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], TT["n_points", "n_points"]]:
    # Edges
    G = nx.read_edgelist(edges_path, delimiter=",", data=[("weight", int)], nodetype=int)

    # Node labels
    with open(labels_path) as f:
        for line in f:
            node, label = line.strip().split(",")
            G.nodes[int(node)]["label"] = int(label)

    dists, idx = _top_cc_dists(G)

    labels = [G.nodes[i]["label"] for i in idx]
    return torch.tensor(dists), torch.tensor(labels), torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_cora(
    cora_edges_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "cora" / "cora.edges",
    cora_labels_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "cora" / "cora.node_labels",
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], TT["n_points", "n_points"]]:
    return _load_network_repository(cora_edges_path, cora_labels_path)


def load_citeseer(
    citeseer_edges_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "citeseer" / "citeseer.edges",
    citeseer_labels_path: str = Path(__file__).parent.parent.parent
    / "data"
    / "graphs"
    / "citeseer"
    / "citeseer.node_labels",
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], TT["n_points", "n_points"]]:
    return _load_network_repository(citeseer_edges_path, citeseer_labels_path)


def load_pubmed(
    pubmed_edges_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "pubmed" / "pubmed.edges",
    pubmed_labels_path: str = Path(__file__).parent.parent.parent / "data" / "graphs" / "pubmed" / "pubmed.node_labels",
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], TT["n_points", "n_points"]]:
    return _load_network_repository(pubmed_edges_path, pubmed_labels_path)


def load_karate_club(
    karate_club_path=Path(__file__).parent.parent.parent / "data" / "graphs" / "karate" / "karate.gml",
) -> Tuple[TT["n_points", "n_points"], None, TT["n_points", "n_points"]]:
    G = nx.read_gml(karate_club_path, label="id")

    dists, idx = _top_cc_dists(G)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_lesmis(
    lesmis_path=Path(__file__).parent.parent.parent / "data" / "graphs" / "lesmis" / "lesmis.gml",
) -> Tuple[TT["n_points", "n_points"], None, TT["n_points", "n_points"]]:
    G = nx.read_gml(lesmis_path, label="id")

    dists, idx = _top_cc_dists(G)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_adjnoun(
    adjnoun_path=Path(__file__).parent.parent.parent / "data" / "graphs" / "adjnoun" / "adjnoun.gml",
) -> Tuple[TT["n_points", "n_points"], None, TT["n_points", "n_points"]]:
    G = nx.read_gml(adjnoun_path, label="id")

    dists, idx = _top_cc_dists(G)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_blood_cells(
    blood_cell_anndata_path: str = Path(__file__).parent.parent.parent / "data" / "blood_cell_scrna" / "adata.h5ad.gz",
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], None]:
    with gzip.open(blood_cell_anndata_path, "rb") as f:
        adata = anndata.read_h5ad(f)
    X = torch.tensor(adata.X.todense()).float()
    X = X / X.sum(dim=1, keepdim=True)
    y = torch.tensor([int(x) for x in adata.obs["cell_type"].values])

    return X, y, None


def load_lymphoma(
    lymphoma_anndata_path: str = Path(__file__).parent.parent.parent / "data" / "lymphoma" / "adata.h5ad.gz",
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], None]:
    """https://www.10xgenomics.com/resources/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0"""
    with gzip.open(lymphoma_anndata_path, "rb") as f:
        adata = anndata.read_h5ad(f)
    X = torch.tensor(adata.X.todense()).float()
    X = X / X.sum(dim=1, keepdim=True)
    y = torch.tensor([int(x) for x in adata.obs["cell_type"].values])

    return X, y, None


def load_cifar_100(
    cifar_data_path=Path(__file__).parent.parent.parent / "data" / "cifar_100" / "cifar-100-python",
    coarse: bool = True,
    train: bool = True,
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], None]:
    # Load data
    split = "train" if train else "test"
    with open(cifar_data_path / split, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    X = torch.tensor(data[b"data"]).float()
    X = X.reshape(-1, 3, 32, 32)  # .permute(0, 2, 3, 1)
    X = X / 255.0

    labels = data[b"coarse_labels"] if coarse else data[b"fine_labels"]

    return X, torch.tensor(labels), None


def load_mnist(
    mnist_data_path=Path(__file__).parent.parent.parent / "data" / "mnist",
    train: bool = True,
) -> Tuple[TT["n_points", "n_points"], TT["n_points"], None]:
    split = "train" if train else "t10k"

    # Load data
    digits = []
    with open(mnist_data_path / f"{split}-images-idx3-ubyte", "rb") as f:
        f.read(16)
        while True:
            digit = f.read(28 * 28)
            if not digit:
                break
            digits.append(list(digit))

    X = torch.tensor(digits).reshape(-1, 28, 28).float()
    X /= 255.0

    # Get labels
    labels = []
    with open(mnist_data_path / f"{split}-labels-idx1-ubyte", "rb") as f:
        f.read(8)
        while True:
            label = f.read(1)
            if not label:
                break
            labels.append(int.from_bytes(label, byteorder="big"))

    return X, torch.tensor(labels), None


def _month_to_unit_circle_point(month: str) -> Tuple[float, float]:
    """Convert month abbreviation to a point on the unit circle."""
    month_to_index = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
    }

    if month not in month_to_index:
        raise ValueError(f"Invalid month: {month}")

    index = month_to_index[month]
    angle = 2 * np.pi * index / 12

    # Return x and y coordinates on the unit circle
    return np.cos(angle), np.sin(angle)


def load_temperature(
    temperature_path: str = Path(__file__).parent.parent / "data" / "temperature" / "temperature.csv",
) -> Tuple[TT["n_points", "n_points", "n_points"], TT["n_points", "n_points"], TT["n_points"]]:
    temperature_dataset = pd.read_csv(temperature_path)
    temperature_dataset = temperature_dataset.drop(columns=["Latitude", "Longitude", "Country", "City", "Year"])
    temperature_dataset = pd.melt(
        temperature_dataset, id_vars=["X", "Y", "Z"], var_name="Month", value_name="Temperature"
    )

    # Apply month_to_unit_circle_point to the 'Month' column to get x and y for each month
    temperature_dataset[["Month_X", "Month_Y"]] = temperature_dataset["Month"].apply(
        lambda month: pd.Series(_month_to_unit_circle_point(month))
    )

    return (
        torch.tensor(temperature_dataset[["X", "Y", "Z"]]),
        torch.tensor(temperature_dataset[["Month_X", "Month_Y"]]),
        torch.tensor(temperature_dataset[["Temperature"]]),
    )


def load(name: str, **kwargs) -> Tuple[TT["n_points", "n_points"], TT["n_points"], TT["n_points", "n_points"]]:
    if name == "cities":
        return load_cities(**kwargs)
    elif name == "cs_phds":
        return load_cs_phds(**kwargs)
    elif name == "facebook":
        return load_facebook(**kwargs)
    elif name == "power":
        return load_power(**kwargs)
    elif name == "polblogs":
        return load_polblogs(**kwargs)
    elif name == "cora":
        return load_cora(**kwargs)
    elif name == "citeseer":
        return load_citeseer(**kwargs)
    elif name == "pubmed":
        return load_pubmed(**kwargs)
    elif name == "karate_club":
        return load_karate_club(**kwargs)
    elif name == "lesmis":
        return load_lesmis(**kwargs)
    elif name == "adjnoun":
        return load_adjnoun(**kwargs)
    elif name == "blood_cells":
        return load_blood_cells(**kwargs)
    elif name == "lymphoma":
        return load_lymphoma(**kwargs)
    elif name == "cifar_100":
        return load_cifar_100(**kwargs)
    elif name == "mnist":
        return load_mnist(**kwargs)
    elif name == "temperature":
        return load_temperature(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
