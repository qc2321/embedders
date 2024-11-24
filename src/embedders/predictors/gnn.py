import torch
from torch_geometric.nn import GCNConv
from torch import nn
from .mlp import MLP


# Create edges for a subset of nodes
def get_subset_edges(dist_matrix, node_indices):
    # Get submatrix of distances
    sub_dist = dist_matrix[node_indices][:, node_indices]

    # Create edges based on threshold
    threshold = sub_dist.mean()
    edges = (sub_dist < threshold).nonzero().t()

    return edges


def get_dense_edges(dist_matrix, node_indices):
    # Get submatrix of distances
    sub_dist = dist_matrix[node_indices][:, node_indices]

    # Create dense edges (all-to-all connections)
    n = len(node_indices)
    rows = torch.arange(n).repeat_interleave(n)
    cols = torch.arange(n).repeat(n)
    edge_index = torch.stack([rows, cols])

    # Get corresponding distances as edge weights
    edge_weights = sub_dist.flatten()

    # Convert distances to weights (you can modify this function)
    edge_weights = torch.exp(-edge_weights)  # Gaussian kernel
    # Alternative weightings:
    # edge_weights = 1 / (edge_weights + 1e-6)  # Inverse distance
    # edge_weights = torch.softmax(-edge_weights, dim=0)  # Softmax of negative distances

    return edge_index, edge_weights


class GNN(MLP):
    def __init__(
        self,
        pm,
        input_dim=0,
        hidden_dims=None,
        output_dim=0,
        tangent=True,
        task="classification",
        activation=nn.ReLU,
        edge_func=get_dense_edges,
    ):
        super().__init__(pm=pm)
        self.pm = pm
        self.task = task
        self.tangent = tangent
        self.edge_func = edge_func

        # Build architecture
        if hidden_dims is None:
            hidden_dims = []

        dimensions = [input_dim] + hidden_dims + [output_dim]
        layers = []

        # Create layers
        for i in range(len(dimensions) - 1):
            # bias = True if tangent or i > 0 else False  # First layer for non-tangent = no bias
            bias = not tangent and i == 0
            layers.append(GCNConv(dimensions[i], dimensions[i + 1], bias=bias))

            # Add activation function after all layers except the last one
            if i < len(dimensions) - 2:
                layers.append(activation())

        # Register layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x

    def fit(self, X, y, train_idx, epochs=1_000, lr=0.01):
        # Get dists and such
        dist_matrix = self.pm.pdist(X).detach()
        if self.tangent:
            X = self.pm.logmap(X).detach()

        # Get edges for training set
        train_edges, train_weights = self.edge_func(dist_matrix, train_idx)
        X_train = X[train_idx]
        y_train = y[train_idx]

        # This part is the same as MLP.fit()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        if self.task == "classification":
            loss_fn = nn.CrossEntropyLoss()
            y_train = y_train.long()
        else:
            loss_fn = nn.MSELoss()
            y_train = y_train.float()

        self.train()
        for i in range(epochs):
            opt.zero_grad()
            y_pred = self(X_train, train_edges, train_weights)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            opt.step()

    def predict(self, X, test_idx):
        """Make predictions."""
        # Get edges for test set
        self.eval()
        dist_matrix = self.pm.pdist(X).detach()
        test_edges, test_weights = self.edge_func(dist_matrix, test_idx)
        X_test = X[test_idx]
        y_pred = self(X_test, test_edges, test_weights)
        return y_pred.argmax(1).detach()
