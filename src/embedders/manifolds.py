import torch
import geoopt

from torchtyping import TensorType


class Manifold:
    def __init__(self, curvature, dim):
        self.curvature = curvature
        self.dim = dim
        self.radius = 1 / abs(curvature) if curvature != 0 else 1
        if curvature < 0:
            self.type = "H"
            self.manifold = geoopt.Scaled(geoopt.Lorentz(), self.radius, learnable=True)
        elif curvature == 0:
            self.type = "E"
            self.manifold = geoopt.Scaled(geoopt.Euclidean(), self.radius, learnable=True)
        else:
            self.type = "S"
            self.manifold = geoopt.Scaled(geoopt.Sphere(), self.radius, learnable=True)
        self.ambient_dim = dim if curvature == 0 else dim + 1
        self.name = f"{self.type}_{abs(self.curvature):.1f}^{dim}"

    def dist(
        self, X: TensorType["n_points1", "n_dim"], Y: TensorType["n_points2", "n_dim"]
    ) -> TensorType["n_points1", "n_points2"]:
        """Inherit distance function from the geoopt manifold."""
        return self.manifold.dist(X, Y)

    def dist2(
        self, X: TensorType["n_points1", "n_dim"], Y: TensorType["n_points2", "n_dim"]
    ) -> TensorType["n_points1", "n_points2"]:
        """Inherit squared distance function from the geoopt manifold."""
        return self.manifold.dist2(X, Y)

    def pdist(self, X: TensorType["n_points", "n_dim"]) -> TensorType["n_points", "n_points"]:
        """Compute pairwise  distances between embeddings."""
        return self.dist(X[:, None], X[None, :])

    def pdist2(self, X: TensorType["n_points", "n_dim"]) -> TensorType["n_points", "n_points"]:
        """Compute pairwise SQUARED distances between embeddings."""
        return self.dist2(X[:, None], X[None, :])

    def kl_divergence(self, p, q):
        """This doesn't come with geoopt, so we have to implement it ourselves."""
        raise NotImplementedError


class ProductManifold(Manifold):
    def __init__(self, signature):
        # Basic properties
        self.signature = signature
        self.curvatures = [curvature for curvature, _ in signature]
        self.dims = [dim for _, dim in signature]
        self.n_manifolds = len(signature)

        # Actually initialize the geoopt manifolds; other derived properties
        self.P = [Manifold(curvature, dim) for curvature, dim in signature]
        self.manifold = geoopt.ProductManifold(*[(M.manifold, M.ambient_dim) for M in self.P])
        self.name = " x ".join([M.name for M in self.P])

        # Manifold <-> Dimension mapping
        curr_dim, curr_man = 0, 0
        dim2man, man2dim = {}, {}

        for M in self.P:
            for d in range(curr_dim, curr_dim + M.ambient_dim):
                dim2man[d] = curr_man
            man2dim[curr_man] = list(range(curr_dim, curr_dim + M.ambient_dim))

            curr_dim += M.ambient_dim
            curr_man += 1

        self.dim2man = dim2man
        self.man2dim = man2dim

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
