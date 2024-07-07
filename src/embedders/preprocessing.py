from torchtyping import TensorType


def knn_graph(x: TensorType["n_points", "n_dim"], k: int) -> TensorType["n_points", "n_points"]:
    """Compute the k-nearest neighbor graph from ambient coordinates."""
    raise NotImplementedError
