"""EXPERIMENTAL: SPEED UP TREE WITH MASKING"""

import torch

# from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from hyperdt.torch.tree import DecisionNode, HyperbolicDecisionTreeClassifier
from hyperdt.torch.product_space_DT import ProductSpaceDT
from hyperdt.torch.hyperbolic_trig import _hyperbolic_midpoint
from .manifolds import ProductManifold


def circular_greater(angles, threshold):
    """
    Check if angles are in the half-circle (threshold, threshold + pi)
    """
    return ((angles - threshold + torch.pi) % (2 * torch.pi)) - torch.pi > 0


def get_mask(angles):
    # Apply circular_greater to get mask of shape (batch_size, batch_size, n_dim)
    mask = circular_greater(angles.unsqueeze(1), angles.unsqueeze(0))
    return mask


def calculate_info_gain(values, labels, mask):
    batch_size, n_dim = values.shape
    n_classes = labels.shape[1]

    # Calculate total Gini impurity without bincount
    class_probs = labels.mean(dim=0)
    total_gini = 1 - (class_probs**2).sum()

    # Initialize arrays for left and right counts
    left_counts = torch.zeros((batch_size, n_dim, n_classes), device=values.device)
    right_counts = torch.zeros((batch_size, n_dim, n_classes), device=values.device)

    # Calculate left and right counts for each potential split
    # mask = get_mask(values)
    for i in range(batch_size):
        for j in range(n_dim):
            left_mask = ~mask[i, :, j]
            right_mask = mask[i, :, j]
            left_counts[i, j] = labels[left_mask].sum(dim=0)
            right_counts[i, j] = labels[right_mask].sum(dim=0)

    # Calculate Gini impurities for left and right partitions
    left_total = left_counts.sum(dim=-1, keepdim=True).clamp(min=1)
    right_total = right_counts.sum(dim=-1, keepdim=True).clamp(min=1)
    left_gini = 1 - ((left_counts / left_total) ** 2).sum(dim=-1)
    right_gini = 1 - ((right_counts / right_total) ** 2).sum(dim=-1)

    # Calculate weighted Gini impurity
    left_weight = left_total.squeeze(-1) / batch_size
    right_weight = right_total.squeeze(-1) / batch_size
    weighted_gini = left_weight * left_gini + right_weight * right_gini

    # Calculate information gain
    info_gain = total_gini - weighted_gini

    return info_gain


def _get_angle_vals(X, pm):
    angle_vals = torch.zeros((X.shape[0], pm.dim), device=X.device, dtype=X.dtype)

    for i, M in enumerate(pm.P):
        dims = pm.man2dim[i]
        dims_target = pm.man2intrinsic[i]
        if M.type in ["H", "S"]:
            angle_vals[:, dims_target] = torch.atan2(X[:, dims[0]].view(-1, 1), X[:, dims[1:]])
        elif M.type == "E":
            angle_vals[:, dims_target] = torch.atan2(torch.tensor(1), X[:, dims])

    return angle_vals


class TorchProductSpaceDT(ProductSpaceDT):
    def __init__(self, signature, **kwargs):
        sig_r = [(x[1], x[0]) for x in signature]
        super().__init__(signature=sig_r, **kwargs)
        self.pm = ProductManifold(signature=signature)

    def fit(self, X, y, preprocess=True, mask=None):
        """Fit a decision tree to the data. Modified from HyperbolicDecisionTreeClassifier
        to remove multiple timelike dimensions in product space."""

        # Get array of classes
        self.classes_, y_relabeled = torch.unique(y, return_inverse=True)

        # First, we can compute the angles of all 2-d projections
        if preprocess:
            angle_vals = _get_angle_vals(X, pm=self.pm)
            labels = torch.nn.functional.one_hot(y_relabeled, num_classes=len(self.classes_)).float()
        else:
            angle_vals = X
            labels = y
        if mask is None:
            mask = get_mask(angle_vals)
        self.tree = self._fit_node(X=angle_vals, y=labels, depth=0, mask=mask)

    def _fit_node(self, X, y, depth, mask):
        # print(f"Depth {depth} with {X.shape} samples")
        # Base case
        if depth == self.max_depth or len(X) < self.min_samples_split or len(torch.unique(y)) == 1:
            value, probs = self._leaf_values(y)
            return DecisionNode(value=value, probs=probs)

        # Recursively find the best split:
        ig = calculate_info_gain(X, y, mask)
        best_idx = torch.argmax(ig)
        best_row, best_dim = best_idx // X.shape[1], best_idx % X.shape[1]
        best_ig = ig[best_row, best_dim]

        # Since we're evaluating greater than, we need to also find the next-largest value and take the midpoint
        next_largest = torch.max(X[~circular_greater(X[:, best_dim], X[best_row, best_dim]), best_dim])

        # Midpoint computation will depend on manifold; TODO: actually do this
        best_manifold = self.pm.P[self.pm.intrinsic2man[best_dim.item()]]
        if best_manifold.type == "H":
            best_theta = _hyperbolic_midpoint(X[best_row, best_dim], next_largest)
        elif best_manifold.type == "S":
            best_theta = (X[best_row, best_dim] + next_largest) / 2
        else:
            best_theta = torch.arctan2(
                torch.tensor([2.0], device=X.device), 1 / torch.tan(X[best_row, best_dim]) + 1 / torch.tan(next_largest)
            )

        # Fallback case:
        if best_ig <= 0:
            # print(f"Fallback triggered at depth {depth}")
            value, probs = self._leaf_values(y)
            return DecisionNode(value=value, probs=probs)

        # Populate:
        node = DecisionNode(feature=best_dim, theta=best_theta)
        node.score = best_ig
        # left, right = circular_greater(X[:, best_dim], best_theta), ~circular_greater(X[:, best_dim], best_theta)
        l, r = mask[best_row, :, best_dim], ~mask[best_row, :, best_dim]
        node.left = self._fit_node(X=X[l], y=y[l], depth=depth + 1, mask=mask[l][:, l])
        node.right = self._fit_node(X=X[r], y=y[r], depth=depth + 1, mask=mask[r][:, r])
        return node

    def _leaf_values(self, y):
        y_sum = y.sum(dim=0)
        value = torch.argmax(y_sum)
        probs = y_sum / y_sum.sum()
        return value, probs

    def predict(self, X):
        angle_vals = _get_angle_vals(X, pm=self.pm)
        return self.classes_[torch.tensor([self._traverse(x).value for x in angle_vals], device=X.device)]

    def _left(self, x, node):
        """Boolean: Go left?"""
        return circular_greater(x[node.feature], node.theta)


class TorchProductSpaceRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        signature,
        n_estimators=100,
        max_features="sqrt",
        max_samples=1.0,
        random_state=None,
        max_depth=3,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.trees = [TorchProductSpaceDT(signature, max_depth=max_depth) for _ in range(n_estimators)]
        self.pm = ProductManifold(signature=signature)

    def _generate_subsample(self, X, y, mask):
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.max_samples)
        # indices = np.random.choice(n_samples, size=sample_size, replace=True)
        indices = torch.randint(0, n_samples, (sample_size,))
        return X[indices], y[indices], mask[indices][:, indices]

    def fit(self, X, y):
        self.classes_, y_relabeled = torch.unique(y, return_inverse=True)

        angle_vals = _get_angle_vals(X, pm=self.pm)
        labels = torch.nn.functional.one_hot(y_relabeled, num_classes=len(self.classes_)).float()
        mask = get_mask(angle_vals)

        for tree in self.trees:
            X_sample, y_sample, mask_sample = self._generate_subsample(angle_vals, labels, mask)
            tree.fit(X_sample, y_sample, preprocess=False, mask=mask_sample)
        return self

    def predict(self, X):
        predictions = torch.stack([tree.predict(X) for tree in self.trees])
        return torch.mode(predictions, dim=0).values

    def predict_proba(self, X):
        predictions = torch.stack([tree.predict_proba(X) for tree in self.trees])
        return predictions.mean(dim=0)

    def score(self, X, y):
        return torch.mean(self.predict(X) == y)
