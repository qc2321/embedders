import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from torch_geometric.nn import GCNConv  # If you want to use pytorch geometric


class ExpMapLayer(nn.Module):
    def __init__(self, pm):
        super().__init__()
        self.pm = pm

    def forward(self, x):
        return self.pm.expmap(x)


class TangentMLP(nn.Module):
    def __init__(self, pm, input_dim, hidden_dims, num_classes):
        super().__init__()
        self.pm = pm
        layers = [ExpMapLayer(pm)]
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TangentMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, pm, input_dim, hidden_dims=[64, 32], lr=1e-2):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.model = None
        self.pm = pm

    def fit(self, X, y, n_epochs=1000):
        # Convert to tensor, initialize model
        self.classes_ = torch.unique(y)
        self.model = TangentMLP(self.pm, self.input_dim, self.hidden_dims, len(self.classes_))

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        return self

    def predict(self, X):
        X = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            return self.classes_[output.argmax(dim=1)]


class TangentGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        return self.classifier(x)


class TangentGNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim=64, lr=1e-2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.model = None

    def fit(self, X, y, edge_index=None):
        # Convert to tensor
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        self.classes_ = torch.unique(y)

        # Create edge_index if not provided (e.g. kNN in tangent space)
        if edge_index is None:
            edge_index = self._get_knn_edges(X)

        # Initialize model
        self.model = TangentGNN(self.input_dim, self.hidden_dim, len(self.classes_))

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(1000):  # Add proper stopping criteria
            optimizer.zero_grad()
            output = self.model(X, edge_index)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X, edge_index=None):
        X = torch.FloatTensor(X)
        if edge_index is None:
            edge_index = self._get_knn_edges(X)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X, edge_index)
            return self.classes_[output.argmax(dim=1)]

    def _get_knn_edges(self, X, k=10):
        # Compute pairwise distances in tangent space
        # Return edge_index in PyG format
        # Can use torch_cluster.knn_graph if you have PyG installed
        pass
