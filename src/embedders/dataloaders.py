import torch
from torchtyping import TensorType
import networkx as nx
from scipy.io import mmread


def load_cities(cities_path: str = "/home/phil/productDT/data/cities.txt") -> TensorType["n_points", "n_points"]:
    dists_flattened = []
    with open(cities_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            dists_flattened += [float(x) for x in line.split()]

    cities_dists = torch.tensor(dists_flattened).reshape(312, 312)

    return cities_dists


def load_cs_phds(cs_phds_path="/home/phil/productDT/data/cs_phds.txt") -> TensorType["n_points", "n_points"]:
    G = nx.read_pajek(cs_phds_path).to_undirected()
    phd_dists = nx.floyd_warshall_numpy(G.subgraph(max(nx.connected_components(G), key=len)))

    return torch.tensor(phd_dists)


def load_facebook():
    raise NotImplementedError


def load_power():
    raise NotImplementedError


def load_polblogs(
    polblogs_path: str = "/home/phil/productDT/data/polblogs.mtx",
    polblogs_labels_path: str = "/home/phil/productDT/data/polblogs_labels.tsv",
) -> TensorType["n_points", "n_points"]:

    # Load the graph
    dists = nx.floyd_warshall_numpy(nx.from_scipy_sparse_array(mmread(polblogs_path)))

    # Load the labels
    polblogs_labels = pd.read_table(polblogs_labels_path, header=None)[0].values

    return torch.tensor(dists)


def load_blood_cells():
    raise NotImplementedError


def _load_lymphoma():
    """https://www.10xgenomics.com/resources/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0"""
    raise NotImplementedError


def _load_healthy_donors():
    """https://www.10xgenomics.com/resources/datasets/pbm-cs-from-a-healthy-donor-targeted-compare-immunology-panel-3-1-standard-4-0-0"""


def load_lymphoma_and_healthy_donors():
    lymphoma = _load_lymphoma()
    healthy_donors = _load_healthy_donors()
    raise NotImplementedError
