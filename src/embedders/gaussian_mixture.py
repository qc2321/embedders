import torch

from torchtyping import TensorType

from .manifolds import ProductManifold

from typing import Optional


def gaussian_mixture(
    pm: ProductManifold,
    num_points: int = 1_000,
    num_classes: int = 2,
    seed: Optional[int] = None,
    cov_scale_means: float = 1.0,
    cov_scale_points: float = 1.0,
) -> (TensorType["n_points", "ambient_dim"], TensorType["n_points"]):
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)

    # Generate cluster means
    class_means = pm.sample(
        z_mean=torch.stack([pm.mu0] * num_classes),
        # sigma=torch.stack([torch.eye(pm.dim)] * num_classes) * cov_scale_means,
        sigma_factorized=[torch.stack([torch.eye(M.dim)] * num_classes) * cov_scale_means for M in pm.P]
    )
    assert class_means.shape == (num_classes, pm.ambient_dim)

    # Generate class assignments
    class_probs = torch.rand(num_classes)
    class_probs /= class_probs.sum()
    class_assignments = torch.multinomial(input=class_probs, num_samples=num_points, replacement=True)
    assert class_assignments.shape == (num_points,)

    # Generate covariance matrices for each class - Wishart distribution
    # cov_matrices = torch.distributions.Wishart(
        # df=pm.dim + 1, covariance_matrix=torch.eye(pm.dim) * cov_scale_points
    # ).sample(sample_shape=(num_classes,))
    # assert cov_matrices.shape == (num_classes, pm.dim, pm.dim)
    # assert torch.all(cov_matrices == cov_matrices.transpose(-1, -2))
    cov_matrices = [
        torch.distributions.Wishart(
            df=M.dim + 1, covariance_matrix=torch.eye(M.dim) * cov_scale_points
        ).sample(sample_shape=(num_classes,)) for M in pm.P
    ]

    # Generate random samples for each cluster
    sample_means = torch.stack([class_means[c] for c in class_assignments])
    assert sample_means.shape == (num_points, pm.ambient_dim)
    # sample_covs = torch.stack([cov_matrices[c] for c in class_assignments])
    sample_covs = [torch.stack([cov_matrix[c] for c in class_assignments]) for cov_matrix in cov_matrices]
    # assert sample_covs.shape == (num_points, pm.dim, pm.dim)
    # samples = pm.sample(z_mean=sample_means, sigma=sample_covs)
    samples = pm.sample(z_mean=sample_means, sigma_factorized=sample_covs)
    assert samples.shape == (num_points, pm.ambient_dim)

    return samples, class_assignments
