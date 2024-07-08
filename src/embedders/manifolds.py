import torch
import geoopt

from torchtyping import TensorType


class Manifold:
    def __init__(self, curvature, dim):
        self.curvature = curvature
        self.dim = dim
        self.scale = abs(curvature) ** -0.5 if curvature != 0 else 1

        # A couple of manifold-specific quirks we need to deal with here
        if curvature < 0:
            self.type = "H"
            self.manifold = geoopt.Scaled(geoopt.Lorentz(), self.scale, learnable=True)
        elif curvature == 0:
            self.type = "E"
            self.manifold = geoopt.Scaled(geoopt.Euclidean(), self.scale, learnable=True)
        else:
            self.type = "S"
            self.manifold = geoopt.Scaled(geoopt.Sphere(), self.scale, learnable=True)

        self.ambient_dim = dim if curvature == 0 else dim + 1
        self.mu0 = torch.zeros(self.ambient_dim) if curvature == 0 else torch.Tensor([1.0] + [0.0] * dim)
        self.name = f"{self.type}_{abs(self.curvature):.1f}^{dim}"

        # Couple of assertions to check
        assert self.manifold.check_point(self.mu0)

    def dist(
        self, X: TensorType["n_points1", "n_dim"], Y: TensorType["n_points2", "n_dim"]
    ) -> TensorType["n_points1", "n_points2"]:
        """Inherit distance function from the geoopt manifold."""
        if self.type == "E":
            return self.manifold.dist(X, Y).norm(dim=-1)
        else:
            return self.manifold.dist(X, Y)

    def dist2(
        self, X: TensorType["n_points1", "n_dim"], Y: TensorType["n_points2", "n_dim"]
    ) -> TensorType["n_points1", "n_points2"]:
        """Inherit squared distance function from the geoopt manifold."""
        if self.type == "E":
            return self.manifold.dist2(X, Y).sum(dim=-1)
        else:
            return self.manifold.dist2(X, Y)

    def pdist(self, X: TensorType["n_points", "n_dim"]) -> TensorType["n_points", "n_points"]:
        """Compute pairwise  distances between embeddings."""
        if self.type == "E":
            return self.dist(X[:, None], X[None, :]).norm(dim=-1)
        else:
            return self.dist(X[:, None], X[None, :])

    def pdist2(self, X: TensorType["n_points", "n_dim"]) -> TensorType["n_points", "n_points"]:
        """Compute pairwise SQUARED distances between embeddings."""
        if self.type == "E":
            return self.dist2(X[:, None], X[None, :]).sum(dim=-1)
        else:
            return self.dist2(X[:, None], X[None, :])

    def kl_divergence(self, p, q, z, other_sample):
        """This doesn't come with geoopt, so we have to implement it ourselves.

        KL divergence is calculated as: KL(p || q) = E_p[log(p(z) / q(z))]
        where z is a point on the manifold.

        Args:
        p (Distribution): The "true" distribution which typically represents the prior.
        q (Distribution): The variational distribution, typically representing the posterior.
        """

        # q = WrappedNormal(z_mean, sigma, manifold=self.manifold)
        # p = WrappedNormal(origin, I, manifold=self.manifold)
        # For other_sample, look at this:
        # log(z) = log p(v) - log det [(\partial / \partial v) proj_{\mu}(v)]
        # v = data[1]
        # assert torch.isfinite(v).all()

        # n_logprob = self.normal.log_prob(v)
        # logdet = self.manifold.logdet(self.loc, self.scale, z, (*data, n_logprob))
        # assert n_logprob.shape == logdet.shape
        # log_prob = n_logprob - logdet
        # assert torch.isfinite(log_prob).all()
        # return log_prob

        log_q = q.log_prob(z, other_sample)
        log_p = p.log_prob(z)
        return log_q - log_p

    def _to_tangent_plane_mu0(self, x: TensorType["n_points", "n_dim"]) -> TensorType["n_points", "n_ambient_dim"]:
        x = torch.Tensor(x).reshape(-1, self.dim)
        if self.type == "E":
            return x
        else:
            return torch.cat([torch.zeros((x.shape[0], 1)), x], dim=1)

    def sample(self, z_mean, sigma=None):
        """Sample from the variational distribution."""
        z_mean = torch.Tensor(z_mean).reshape(-1, self.ambient_dim)
        if sigma is None:
            sigma = torch.eye(self.dim)
        else:
            sigma = torch.Tensor(sigma)
        assert sigma.shape == (self.dim, self.dim)
        assert torch.all(sigma == sigma.T)
        assert z_mean.shape[1] == self.ambient_dim

        # Sample initial vector from N(0, sigma)
        N = torch.distributions.MultivariateNormal(torch.zeros(self.dim), sigma)
        v = N.sample((z_mean.shape[0],))

        # Don't need to adjust normal vectors for the Scaled manifold class in geoopt - very cool!

        # Enter tangent plane
        v = self._to_tangent_plane_mu0(v)

        # Move to z_mean via parallel transport
        z = self.manifold.transp(x=self.mu0, y=z_mean, v=v)

        # Exp map onto the manifold
        return self.manifold.expmap(x=z_mean, u=z)
    
    def log_likelihood(self, z, mu, sigma):
        """ Probability density function for WN(z ; mu, Sigma) in manifold """

        # Euclidean case is regular old Gaussian log-likelihood
        if self.type == "E":
            return torch.distributions.MultivariateNormal(mu, sigma).log_prob(z)

        elif self.type in ["S", "H"]:
            u = self.manifold.logmap(x=mu, y=z) # Map z to tangent space at mu
            v = self.manifold.transp(x=mu, y=manifold.mu0, v=u) # Parallel transport to origin
            ll = torch.distributions.MultivariateNormal(torch.zeros(self.dim), sigma).log_prob(v)

            # For convenience
            R = self.scale
            n = self.dim
            u_norm = self.manifold.inner(u, u) # Works for H and S

            # Final formula (epsilon to avoid log(0))
            if self.type == "S":
                return ll - (n - 1) * torch.log(R * torch.abs(torch.sin(u_norm / R) / u_norm) + 1e-8)
            
            elif self.type == "H":
                return ll - (n - 1) * torch.log(R * torch.abs(torch.sinh(u_norm / R) / u_norm) + 1e-8)


class ProductManifold(Manifold):
    def __init__(self, signature):
        # Basic properties
        self.type = "P"
        self.signature = signature
        self.curvatures = [curvature for curvature, _ in signature]
        self.dims = [dim for _, dim in signature]
        self.n_manifolds = len(signature)

        # Actually initialize the geoopt manifolds; other derived properties
        self.P = [Manifold(curvature, dim) for curvature, dim in signature]
        self.manifold = geoopt.ProductManifold(*[(M.manifold, M.ambient_dim) for M in self.P])
        self.name = " x ".join([M.name for M in self.P])

        # Manifold <-> Dimension mapping
        curr_dim, curr_man, total_dims = 0, 0, 0
        dim2man, man2dim = {}, {}

        for M in self.P:
            for d in range(curr_dim, curr_dim + M.ambient_dim):
                dim2man[d] = curr_man
            man2dim[curr_man] = list(range(curr_dim, curr_dim + M.ambient_dim))

            curr_dim += M.ambient_dim
            curr_man += 1
            total_dims += M.dim

        self.dim2man = dim2man
        self.man2dim = man2dim
        self.ambient_dim = curr_dim
        self.dim = total_dims

    def initialize_embeddings(self, n_points, scales=1.0):
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
                        u=torch.cat([torch.zeros(n_points, 1), scale * torch.randn(n_points, M.dim)], dim=1)
                    )
                )
            elif M.type == "E":
                x_embed.append(scale * torch.randn(n_points, M.dim))
            elif M.type == "S":
                x_embed.append(M.manifold.random_uniform(n_points, M.ambient_dim))
            else:
                raise ValueError("Unknown manifold type")

        x_embed = torch.cat(x_embed, axis=1)
        # x_embed = geoopt.ManifoldParameter(x_embed, manifold=self.manifold)
        return x_embed

    def factorize(self, X: TensorType["n_points", "n_dim"]) -> list:
        """Factorize the embeddings into the individual manifolds."""
        return [X[:, self.man2dim[i]] for i in range(len(self.P))]

    def sample(self, z_mean: TensorType["n_points", "n_dim"], sigma=None) -> TensorType["n_points", "n_ambient_dim"]:
        """Sample from the variational distribution."""
        z_mean = torch.Tensor(z_mean).reshape(-1, self.ambient_dim)
        if sigma is None:
            sigma = torch.eye(self.dim)
        else:
            sigma = torch.Tensor(sigma)
        assert sigma.shape == (self.dim, self.dim)
        assert torch.all(sigma == sigma.T)
        assert z_mean.shape[1] == self.ambient_dim

        # Sample initial vector from N(0, sigma)
        z_means = self.factorize(z_mean)
        return torch.cat([M.sample(z_mean) for M, z_mean in zip(self.P, z_means)], axis=1)
