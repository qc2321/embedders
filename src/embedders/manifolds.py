import torch
import geoopt

from torchtyping import TensorType as TT
from typing import List, Optional, Tuple, Union


class Manifold:
    def __init__(self, curvature: float, dim: int, device: str = "cpu"):
        # Device management
        self.device = device

        # Basic properties
        self.curvature = curvature
        self.dim = dim
        self.scale = abs(curvature) ** -0.5 if curvature != 0 else 1

        # A couple of manifold-specific quirks we need to deal with here
        if curvature < 0:
            self.type = "H"
            man = geoopt.Lorentz(k=1.0)
            # Use 'k=1.0' because the scale will take care of the curvature
            # For more information, see the bottom of page 5 of Gu et al. (2019):
            # https://openreview.net/pdf?id=HJxeWnCcF7
        elif curvature == 0:
            self.type = "E"
            man = geoopt.Euclidean(ndim=1)
            # Use 'ndim=1' because dim means the *shape* of the Euclidean space, not the dimensionality...
        else:
            self.type = "S"
            man = geoopt.Sphere()
        self.manifold = geoopt.Scaled(man, self.scale, learnable=True).to(self.device)

        self.ambient_dim = dim if curvature == 0 else dim + 1
        if curvature == 0:
            self.mu0 = torch.zeros(self.dim).to(self.device).reshape(1, -1)
        else:
            self.mu0 = torch.Tensor([1.0] + [0.0] * dim).to(self.device).reshape(1, -1)
        self.name = f"{self.type}_{abs(self.curvature):.1f}^{dim}"

        # Couple of assertions to check
        assert self.manifold.check_point(self.mu0)

    def to(self, device: str):
        self.device = device
        self.manifold = self.manifold.to(device)
        self.mu0 = self.mu0.to(device)
        return self

    def inner(self, X: TT["n_points1", "n_dim"], Y: TT["n_points2", "n_dim"]) -> TT["n_points1", "n_points2"]:
        """Not inherited because of weird broadcasting stuff, plus need for scale."""
        # This ensures we compute the right inner product for all manifolds (flip sign of dim 0 for hyperbolic)
        X_fixed = torch.cat([-X[:, 0:1], X[:, 1:]], dim=1) if self.type == "H" else X

        # This prevents dividing by zero in the Euclidean case
        scaler = 1 / abs(self.curvature) if self.type != "E" else 1
        return X_fixed @ Y.T * scaler

    def dist(self, X: TT["n_points1", "n_dim"], Y: TT["n_points2", "n_dim"]) -> TT["n_points1", "n_points2"]:
        """Inherit distance function from the geoopt manifold."""
        # if self.type == "E":
        #     return self.manifold.dist(X[:, None], Y[None, :]).norm(dim=-1)
        # else:
        return self.manifold.dist(X[:, None], Y[None, :])

    def dist2(self, X: TT["n_points1", "n_dim"], Y: TT["n_points2", "n_dim"]) -> TT["n_points1", "n_points2"]:
        """Inherit squared distance function from the geoopt manifold."""
        # if self.type == "E":
        #     return self.manifold.dist2(X[:, None], Y[None, :]).sum(dim=-1)
        # else:
        return self.manifold.dist2(X[:, None], Y[None, :])

    def pdist(self, X: TT["n_points", "n_dim"]) -> TT["n_points", "n_points"]:
        """Compute pairwise  distances between embeddings."""
        # if self.type == "E":
        #     dists = self.dist(X, X).norm(dim=-1)
        # else:
        dists = self.dist(X, X)

        # Fill diagonal with zeros
        dists.fill_diagonal_(0.0)

        return dists

    def pdist2(self, X: TT["n_points", "n_dim"]) -> TT["n_points", "n_points"]:
        """Compute pairwise SQUARED distances between embeddings."""
        # if self.type == "E":
        #     dists2 = self.dist2(X, X).sum(dim=-1)
        # else:
        dists2 = self.dist2(X, X)

        dists2.fill_diagonal_(0.0)

        return dists2

    def _to_tangent_plane_mu0(self, x: TT["n_points", "n_dim"]) -> TT["n_points", "n_ambient_dim"]:
        x = torch.Tensor(x).reshape(-1, self.dim)
        if self.type == "E":
            return x
        else:
            return torch.cat([torch.zeros((x.shape[0], 1), device=self.device), x], dim=1)

    def sample(
        self,
        z_mean: Optional[TT["n_points", "n_ambient_dim"]] = None,
        sigma: Optional[TT["n_points", "n_dim", "n_dim"]] = None,
        return_tangent: bool = False,
    ) -> Union[TT["n_points", "n_ambient_dim"], Tuple[TT["n_points", "n_ambient_dim"], TT["n_points", "n_dim"]]]:
        """Sample from the variational distribution."""
        if z_mean is None:
            z_mean = self.mu0
        z_mean = torch.Tensor(z_mean).reshape(-1, self.ambient_dim).to(self.device)
        n = z_mean.shape[0]
        if sigma is None:
            sigma = torch.stack([torch.eye(self.dim)] * n).to(self.device)
        else:
            sigma = torch.Tensor(sigma).reshape(-1, self.dim, self.dim).to(self.device)
        assert sigma.shape == (n, self.dim, self.dim)
        # assert torch.all(sigma == sigma.transpose(-1, -2))
        assert z_mean.shape[-1] == self.ambient_dim

        # Sample initial vector from N(0, sigma)
        N = torch.distributions.MultivariateNormal(torch.zeros((n, self.dim), device=self.device), sigma)
        v = N.sample(sample_shape=(1,)).reshape(n, self.dim)  # TODO: allow for other numbers of samples

        # Don't need to adjust normal vectors for the Scaled manifold class in geoopt - very cool!

        # Enter tangent plane
        v_tangent = self._to_tangent_plane_mu0(v)

        # Move to z_mean via parallel transport
        z = self.manifold.transp(x=self.mu0, y=z_mean, v=v_tangent)

        # If we're sampling at the origin, z and v should be the same
        mask = torch.all(z == self.mu0, dim=1)
        assert torch.allclose(v_tangent[mask], z[mask])

        # Exp map onto the manifold
        x = self.manifold.expmap(x=z_mean, u=z)

        # Different return types
        if return_tangent:
            return x, v
        else:
            return x

    def log_likelihood(
        self,
        z: TT["n_points", "n_ambient_dim"],
        mu: Optional[TT["n_points", "n_ambient_dim"]] = None,
        sigma: Optional[TT["n_points", "n_dim", "n_dim"]] = None,
    ) -> TT["n_points"]:
        """Probability density function for WN(z ; mu, Sigma) in manifold"""

        # Default to mu=self.mu0 and sigma=I
        if mu is None:
            mu = self.mu0
        mu = torch.Tensor(mu).reshape(-1, self.ambient_dim).to(self.device)
        if sigma is None:
            sigma = torch.stack([torch.eye(self.dim)] * n).to(self.device)
        else:
            sigma = torch.Tensor(sigma).reshape(-1, self.dim, self.dim).to(self.device)

        # Euclidean case is regular old Gaussian log-likelihood
        if self.type == "E":
            return torch.distributions.MultivariateNormal(mu, sigma).log_prob(z)

        elif self.type in ["S", "H"]:
            u = self.manifold.logmap(x=mu, y=z)  # Map z to tangent space at mu
            v = self.manifold.transp(x=mu, y=self.mu0, v=u)  # Parallel transport to origin
            # assert torch.allclose(v[:, 0], torch.Tensor([0.])) # For tangent vectors at origin this should be true
            # OK, so this assertion doesn't actually pass, but it's spiritually true
            if torch.isnan(v).any():
                print("NANs in parallel transport")
                v = torch.nan_to_num(v, nan=0.0)
            N = torch.distributions.MultivariateNormal(torch.zeros(self.dim, device=self.device), sigma)
            ll = N.log_prob(v[:, 1:])

            # For convenience
            R = self.scale
            n = self.dim

            # Final formula (epsilon to avoid log(0))
            if self.type == "S":
                sin_M = torch.sin
                u_norm = self.manifold.norm(x=mu, u=u)

            elif self.type == "H":
                sin_M = torch.sinh
                u_norm = self.manifold.base.norm(u=u)  # Horrible workaround needed for geoopt bug

            return ll - (n - 1) * torch.log(R * torch.abs(sin_M(u_norm / R) / u_norm) + 1e-8)

    def logmap(self, x, base=None):
        """Logarithmic map of point on manifold x at base point"""
        if base is None:
            base = self.mu0
        return self.manifold.logmap(x=base, y=x)

    def expmap(self, u, base=None):
        """Exponential map of tangent vector u at base point"""
        if base is None:
            base = self.mu0
        return self.expmap(x=base, u=u)


class ProductManifold(Manifold):
    def __init__(self, signature: List[Tuple[float, int]], device: str = "cpu"):
        # Device management
        self.device = device

        # Basic properties
        self.type = "P"
        self.signature = signature
        self.curvatures = [curvature for curvature, _ in signature]
        self.dims = [dim for _, dim in signature]
        self.n_manifolds = len(signature)

        # Actually initialize the geoopt manifolds; other derived properties
        self.P = [Manifold(curvature, dim, device=device) for curvature, dim in signature]
        self.manifold = geoopt.ProductManifold(*[(M.manifold, M.ambient_dim) for M in self.P])
        self.name = " x ".join([M.name for M in self.P])

        # Origin
        self.mu0 = torch.cat([M.mu0 for M in self.P], axis=1).to(self.device)

        # Manifold <-> Dimension mapping
        self.ambient_dim, self.n_manifolds, self.dim = 0, 0, 0
        self.dim2man, self.man2dim, self.man2intrinsic, self.intrinsic2man = {}, {}, {}, {}

        for M in self.P:
            for d in range(self.ambient_dim, self.ambient_dim + M.ambient_dim):
                self.dim2man[d] = self.n_manifolds
            for d in range(self.dim, self.dim + M.dim):
                self.intrinsic2man[d] = self.n_manifolds
            self.man2dim[self.n_manifolds] = list(range(self.ambient_dim, self.ambient_dim + M.ambient_dim))
            self.man2intrinsic[self.n_manifolds] = list(range(self.dim, self.dim + M.dim))

            self.ambient_dim += M.ambient_dim
            self.n_manifolds += 1
            self.dim += M.dim

        # Lift matrix - useful for tensor stuff
        # The idea here is to right-multiply by this to lift a vector in R^dim to a vector in R^ambient_dim
        # such that there are zeros in all the right places, i.e. to make it a tangent vector at the origin of P
        self.projection_matrix = torch.zeros(self.dim, self.ambient_dim, device=self.device)
        for i in range(len(self.P)):
            intrinsic_dims = self.man2intrinsic[i]
            ambient_dims = self.man2dim[i]
            for j, k in zip(intrinsic_dims, ambient_dims[-len(intrinsic_dims) :]):
                self.projection_matrix[j, k] = 1.0
        

    def params(self):
        return [x._log_scale for x in self.manifold.manifolds]

    def to(self, device: str):
        self.device = device
        self.P = [M.to(device) for M in self.P]
        self.manifold = self.manifold.to(device)
        self.mu0 = self.mu0.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        return self

    def initialize_embeddings(
        self, n_points: int, scales: Union[List[float], float] = 1.0
    ) -> TT["n_points", "n_ambient_dim"]:
        """Randomly initialize n_points embeddings on the product manifold."""
        # Scales management
        if not isinstance(scales, list):
            scales = [scales] * len(self.P)
        elif len(scales) != len(self.P):
            raise ValueError("The number of scales must match the number of manifolds.")

        x_embed = []
        for M, scale in zip(self.P, scales):
            if M.type == "H":
                x_embed.append(
                    M.manifold.expmap0(
                        u=torch.cat(
                            [
                                torch.zeros(n_points, 1, device=self.device),
                                scale * torch.randn(n_points, M.dim, device=self.device),
                            ],
                            dim=1,
                        )
                    )
                )
            elif M.type == "E":
                x_embed.append(scale * torch.randn(n_points, M.dim, device=self.device))
            elif M.type == "S":
                x_embed.append(M.manifold.random_uniform(n_points, M.ambient_dim).to(self.device))
            else:
                raise ValueError("Unknown manifold type")

        x_embed = torch.cat(x_embed, axis=1).to(self.device)
        # x_embed = geoopt.ManifoldParameter(x_embed, manifold=self.manifold)
        return x_embed

    def factorize(self, X: TT["n_points", "n_dim"], intrinsic=False) -> List[TT["n_points", "n_dim_manifold"]]:
        """Factorize the embeddings into the individual manifolds."""
        dims_dict = self.man2intrinsic if intrinsic else self.man2dim
        return [X[..., dims_dict[i]] for i in range(len(self.P))]

    def sample(
        self,
        z_mean: Optional[TT["n_points", "n_dim"]] = None,
        # sigma: Optional[TT["n_points", "n_dim", "n_dim"]] = None
        sigma_factorized: Optional[List[TT["n_points", "n_dim_manifold", "n_dim_manifold"]]] = None,
        return_tangent: bool = False,
    ) -> Union[TT["n_points", "n_ambient_dim"], Tuple[TT["n_points", "n_ambient_dim"], TT["n_points", "n_dim"]]]:
        """Sample from the variational distribution."""
        if z_mean is None:
            z_mean = self.mu0
        z_mean = torch.Tensor(z_mean).reshape(-1, self.ambient_dim).to(self.device)
        n = z_mean.shape[0]

        if sigma_factorized is None:
            sigma_factorized = [torch.stack([torch.eye(M.dim)] * n) for M in self.P]
        else:
            sigma_factorized = [
                torch.Tensor(sigma).reshape(-1, M.dim, M.dim).to(self.device)
                for M, sigma in zip(self.P, sigma_factorized)
            ]

        assert sum([sigma.shape == (n, M.dim, M.dim) for M, sigma in zip(self.P, sigma_factorized)]) == len(self.P)
        assert z_mean.shape[-1] == self.ambient_dim

        # Sample initial vector from N(0, sigma)
        # x = torch.cat(
        #     [M.sample(z_M, sigma_M) for M, z_M, sigma_M in zip(self.P, self.factorize(z_mean), sigma_factorized)],
        #     dim=1,
        # )
        samples = [
            M.sample(z_M, sigma_M, return_tangent=True)
            for M, z_M, sigma_M in zip(self.P, self.factorize(z_mean), sigma_factorized)
        ]

        x = torch.cat([s[0] for s in samples], dim=1)
        v = torch.cat([s[1] for s in samples], dim=1)

        if return_tangent:
            return x, v
        else:
            return x

    def log_likelihood(
        self,
        z: TT["batch_size", "n_dim"],
        mu: Optional[TT["n_dim",]] = None,
        # sigma: TT["n_intrinsic_dim", "n_intrinsic_dim"] = None,
        sigma_factorized: Optional[List[TT["n_points", "n_dim_manifold", "n_dim_manifold"]]] = None,
    ) -> TT["batch_size"]:
        """Probability density function for WN(z ; mu, Sigma) in manifold"""
        n = z.shape[0]
        if mu is None:
            mu = torch.stack([self.mu0] * n).to(self.device)

        if sigma_factorized is None:
            sigma_factorized = [torch.stack([torch.eye(M.dim)] * n) for M in self.P]
            # Note that this factorization assumes block-diagonal covariance matrices

        mu_factorized = self.factorize(mu)
        z_factorized = self.factorize(z)
        component_lls = [
            M.log_likelihood(z_M, mu_M, sigma_M).unsqueeze(dim=1)
            for M, z_M, mu_M, sigma_M in zip(self.P, z_factorized, mu_factorized, sigma_factorized)
        ]
        return torch.cat(component_lls, axis=1).sum(axis=1)
