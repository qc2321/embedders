import torch
from torchtyping import TensorType as TT
import networkx as nx
import pandas as pd
import numpy as np
import shlex
from scipy.io import mmread


def _top_cc_dists(G: nx.Graph, to_undirected: bool = True) -> (np.ndarray, list):
    """Returns the distances between the top connected component of a graph"""
    if to_undirected:
        G = G.to_undirected()
    top_cc = max(nx.connected_components(G), key=len)
    print(f"Top CC has {len(top_cc)} nodes; original graph has {G.number_of_nodes()} nodes.")
    return nx.floyd_warshall_numpy(G.subgraph(top_cc)), list(top_cc)


def load_cities(cities_path: str = "../../data/cities.txt") -> TT["n_points", "n_points"]:
    dists_flattened = []
    with open(cities_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            dists_flattened += [float(x) for x in line.split()]

    cities_dists = torch.tensor(dists_flattened).reshape(312, 312)

    return cities_dists


def load_cs_phds(cs_phds_path="../../data/cs_phds.txt", labels: bool = False) -> torch.Tensor:
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

    if labels:
        labels = [G.nodes[i]["year"] for i in idx]
        return torch.tensor(phd_dists), labels
    else:
        return torch.tensor(phd_dists)


def load_facebook():
    raise NotImplementedError


def load_power():
    raise NotImplementedError


def load_polblogs(
    polblogs_path: str = "../../data/graphs/polblogs.mtx",
    polblogs_labels_path: str = "../../data/graphs/polblogs_labels.tsv",
    labels=False,
) -> TT["n_points", "n_points"]:
    # Load the graph
    G = nx.from_scipy_sparse_array(mmread(polblogs_path))
    dists, idx = _top_cc_dists(G)

    # Load the labels
    polblogs_labels = pd.read_table(polblogs_labels_path, header=None)[0]

    # Filter to match G
    polblogs_labels = polblogs_labels[idx]

    if labels:
        return torch.tensor(dists), polblogs_labels
    else:
        return torch.tensor(dists)


def load_blood_cells():
    raise NotImplementedError


def _load_lymphoma():
    """https://www.10xgenomics.com/resources/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0"""
    raise NotImplementedError


def _load_healthy_donors():
    """https://www.10xgenomics.com/resources/datasets/pbm-cs-from-a-healthy-donor-targeted-compare-immunology-panel-3-1-standard-4-0-0"""
    raise NotImplementedError


def load_lymphoma_and_healthy_donors():
    lymphoma = _load_lymphoma()
    healthy_donors = _load_healthy_donors()
    raise NotImplementedError


def load(name: str, **kwargs) -> TT["n_points", "n_points"]:
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
    elif name == "blood_cells":
        return load_blood_cells(**kwargs)
    elif name == "lymphoma_and_healthy_donors":
        return load_lymphoma_and_healthy_donors(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
