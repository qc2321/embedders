import torch

from torchtyping import TensorType as TT

from .manifolds import ProductManifold

from typing import Optional, Tuple, Literal


@torch.no_grad()
def gaussian_mixture(
    pm: ProductManifold,
    num_points: int = 1_000,
    num_classes: int = 2,
    num_clusters: Optional[int] = None,
    seed: Optional[int] = None,
    cov_scale_means: float = 1.0,
    cov_scale_points: float = 1.0,
    regression_noise_std: float = 0.1,
    task: Literal["classification", "regression"] = "classification",
) -> Tuple[TT["n_points", "ambient_dim"], TT["n_points"]]:
    """
    Given a product manifold, generate a set of labeled samples from a Gaussian mixture model.

    Args:
        pm: A ProductManifold instance.
        num_points: The number of points to generate.
        num_classes: The number of classes to generate.
        num_clusters: The number of clusters to generate. If None, defaults to num_classes.
        seed: An optional seed for the random number generator.
        cov_scale_means: The scale of the covariance matrix for the means.
        cov_scale_points: The scale of the covariance matrix for the points.
        regression_noise_std: The standard deviation of the noise for regression labels.
        task: The type of labels to generate. Either "classification" or "regression".

    Returns:
        samples: A tensor of shape (num_points, ambient_dim) containing the generated samples.
        class_assignments: A tensor of shape (num_points,) containing the class assignments of the samples.
    """
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)

    # Deal with clusters
    if num_clusters is None:
        num_clusters = num_classes
    else:
        assert num_clusters >= num_classes

    # Generate cluster means
    cluster_means = pm.sample(
        z_mean=torch.stack([pm.mu0] * num_clusters),
        sigma_factorized=[torch.stack([torch.eye(M.dim)] * num_clusters) * cov_scale_means for M in pm.P],
    )
    assert cluster_means.shape == (num_clusters, pm.ambient_dim)

    # Generate class assignments
    cluster_probs = torch.rand(num_clusters)
    cluster_probs /= cluster_probs.sum()
    cluster_assignments = torch.multinomial(input=cluster_probs, num_samples=num_points, replacement=True)
    assert cluster_assignments.shape == (num_points,)

    # Generate covariance matrices for each class - Wishart distribution
    cov_matrices = [
        torch.distributions.Wishart(df=M.dim + 1, covariance_matrix=torch.eye(M.dim) * cov_scale_points).sample(
            sample_shape=(num_clusters,)
        )
        for M in pm.P
    ]

    # Generate random samples for each cluster
    sample_means = torch.stack([cluster_means[c] for c in cluster_assignments])
    assert sample_means.shape == (num_points, pm.ambient_dim)
    sample_covs = [torch.stack([cov_matrix[c] for c in cluster_assignments]) for cov_matrix in cov_matrices]
    samples, tangent_vals = pm.sample(z_mean=sample_means, sigma_factorized=sample_covs, return_tangent=True)
    assert samples.shape == (num_points, pm.ambient_dim)

    # Map clusters to classes
    cluster_to_class = list(range(num_classes))
    for i in range(num_clusters - num_classes):
        cluster_to_class.append(torch.randint(0, num_classes, (1,)).item())
    cluster_to_class = torch.tensor(cluster_to_class)
    assert cluster_to_class.shape == (num_clusters,)
    assert torch.unique(cluster_to_class).shape == (num_classes,)

    # Generate outputs
    if task == "classification":
        labels = cluster_to_class[cluster_assignments]
    elif task == "regression":
        slopes = (0.5 - torch.randn(num_clusters, pm.dim)) * 2
        intercepts = (0.5 - torch.randn(num_clusters)) * 20
        labels = torch.einsum("ij,ij->i", slopes[cluster_assignments], tangent_vals) + intercepts[cluster_assignments]

        # Noise component
        N = torch.distributions.Normal(0, regression_noise_std)
        v = N.sample((num_points,))
        labels += v

        # Normalize regression labels to range [0, 1] so that RMSE can be more easily interpreted
        labels = (labels - labels.min()) / (labels.max() - labels.min())

    return samples, labels
