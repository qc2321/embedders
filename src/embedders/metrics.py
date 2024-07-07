import torch
import networkx as nx

from torchtyping import TensorType


def distortion_loss(
    estimated_distances: TensorType["n_points", "n_points"], true_distances: TensorType["n_points", "n_points"]
) -> float:
    """Compute the distortion loss between estimated SQUARED distances and true SQUARED distances."""
    n = true_distances.shape[0]
    idx = torch.triu_indices(n, n, offset=1)

    pdist_true = true_distances[idx[0], idx[1]]
    pdist_est = estimated_distances[idx[0], idx[1]]

    return torch.sum(torch.abs((pdist_est / pdist_true) ** 2 - 1))


def d_avg(
    estimated_distances: TensorType["n_points", "n_points"], true_distances: TensorType["n_points", "n_points"]
) -> float:
    """Average distance error D_avg"""
    n = true_distances.shape[0]
    idx = torch.triu_indices(n, n, offset=1)

    pdist_true = true_distances[idx[0], idx[1]]
    pdist_est = estimated_distances[idx[0], idx[1]]

    # Note that D_avg uses nonsquared distances:
    return torch.mean(torch.abs(pdist_est - pdist_true) / pdist_true)


def mean_average_precision(x_embed: TensorType["n_points", "n_dim"], graph: nx.Graph) -> float:
    """Mean averae precision (mAP) from the Gu et al paper."""
    raise NotImplementedError


def dist_component_by_manifold(pm, x_embed):
    # """How much of the variance in distance is explained by each manifold?"""
    sq_dists_by_manifold = [M.pdist2(x_embed[:, pm.man2dim[i]]) for i, M in enumerate(pm.P)]
    total_sq_dist = pm.pdist2(x_embed)

    return [
        torch.sum(D.triu(diagonal=1) / torch.sum(total_sq_dist.triu(diagonal=1))).item() for D in sq_dists_by_manifold
    ]
