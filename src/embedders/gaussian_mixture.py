import torch

from torchtyping import TensorType as TT

from .manifolds import ProductManifold

from typing import Optional, Tuple


def gaussian_mixture(
    pm: ProductManifold,
    num_points: int = 1_000,
    num_classes: int = 2,
    seed: Optional[int] = None,
    cov_scale_means: float = 1.0,
    cov_scale_points: float = 1.0,
) -> Tuple[TT["n_points", "ambient_dim"], TT["n_points"]]:
    """
    Given a product manifold, generate a set of labeled samples from a Gaussian mixture model.

    Args:
        pm: A ProductManifold instance.
        num_points: The number of points to generate.
        num_classes: The number of classes to generate.
        seed: An optional seed for the random number generator.
        cov_scale_means: The scale of the covariance matrix for the means.
        cov_scale_points: The scale of the covariance matrix for the points.

    Returns:
        samples: A tensor of shape (num_points, ambient_dim) containing the generated samples.
        class_assignments: A tensor of shape (num_points,) containing the class assignments of the samples.
    """
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)

    # Generate cluster means
    class_means = pm.sample(
        z_mean=torch.stack([pm.mu0] * num_classes),
        sigma_factorized=[torch.stack([torch.eye(M.dim)] * num_classes) * cov_scale_means for M in pm.P],
    )
    assert class_means.shape == (num_classes, pm.ambient_dim)

    # Generate class assignments
    class_probs = torch.rand(num_classes)
    class_probs /= class_probs.sum()
    class_assignments = torch.multinomial(input=class_probs, num_samples=num_points, replacement=True)
    assert class_assignments.shape == (num_points,)

    # Generate covariance matrices for each class - Wishart distribution
    cov_matrices = [
        torch.distributions.Wishart(df=M.dim + 1, covariance_matrix=torch.eye(M.dim) * cov_scale_points).sample(
            sample_shape=(num_classes,)
        )
        for M in pm.P
    ]

    # Generate random samples for each cluster
    sample_means = torch.stack([class_means[c] for c in class_assignments])
    assert sample_means.shape == (num_points, pm.ambient_dim)
    sample_covs = [torch.stack([cov_matrix[c] for c in class_assignments]) for cov_matrix in cov_matrices]
    samples = pm.sample(z_mean=sample_means, sigma_factorized=sample_covs)
    assert samples.shape == (num_points, pm.ambient_dim)

    return samples, class_assignments
