import torch

# from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin

# from hyperdt.torch.product_space_DT import ProductSpaceDT
from .manifolds import ProductManifold
from .midpoint import midpoint

# Typing stuff
from torchtyping import TensorType as TT
from typing import Tuple, List, Optional


def get_info_gains(
    comparisons: TT["query_batch dims key_batch"],
    labels: TT["query_batch n_classes"],
    eps: float = 1e-10,
) -> TT["query_batch dims"]:
    """
    Given comparisons matrix and labels, return information gain for each possible split.

    Args:
        comparisons: (query_batch, dims, key_batch) tensor of comparisons
        labels: (query_batch, n_classes) tensor of one-hot labels
        eps: small number to prevent division by zero

    Outputs:
        ig: (query_batch, dims) tensor of information gains
    """
    # Matrix-multiply to get counts of labels in left and right splits
    pos_labels = (comparisons @ labels).float()
    neg_labels = ((1 - comparisons) @ labels).float()

    # Total counts are sums of label counts
    n_pos = pos_labels.sum(dim=-1) + eps
    n_neg = neg_labels.sum(dim=-1) + eps
    n_total = n_pos + n_neg

    # Probabilities are label counts divided by total counts
    pos_probs = pos_labels / n_pos.unsqueeze(-1)
    neg_probs = neg_labels / n_neg.unsqueeze(-1)
    total_probs = (pos_labels + neg_labels) / n_total.unsqueeze(-1)

    # Gini impurity is 1 - sum(prob^2)
    gini_pos = 1 - (pos_probs**2).sum(dim=-1)
    gini_neg = 1 - (neg_probs**2).sum(dim=-1)
    gini_total = 1 - (total_probs**2).sum(dim=-1)

    # Information gain is the total gini impurity minus the weighted average of the new gini impurities
    ig = gini_total - (gini_pos * n_pos + gini_neg * n_neg) / n_total

    assert not ig.isnan().any()  # Ensure no NaNs

    return ig


def get_best_split(
    ig: TT["query_batch dims"],
    angles: TT["query_batch intrinsic_dim"],
    comparisons: TT["query_batch dims key_batch"],
    pm: ProductManifold,
) -> Tuple[int, int, float]:
    """
    All of the postprocessing for an information gain check

    Args:
        ig: (query_batch, dims) tensor of information gains
        angles: (query_batch, dims) tensor of angles
        comparisons: (query_batch, dims, key_batch) tensor of comparisons
        pm: ProductManifold object, for determining midpoint approach

    Returns:
        n: scalar index of best split (positive class)
        d: scalar dimension of best split
        theta: scalar angle of best split
    """
    # First, figure out the dimension (d) and sample (n)
    best_split = ig.argmax()
    nd = ig.shape[1]
    n, d = best_split // nd, best_split % nd

    # Get the corresponding angle
    theta_pos = angles[n, d]

    # We have the angle, but ideally we would like the *midpoint* angle.
    # So we need to grab the closest angle from the negative class:
    n_neg = (angles[comparisons[n, d] == 0.0, d] - theta_pos).abs().argmin()
    theta_neg = angles[comparisons[n, d] == 0.0, d][n_neg]

    # Get manifold
    manifold = pm.P[pm.intrinsic2man[d.item()]]

    # Print what you're doing
    m = midpoint(theta_pos, theta_neg, manifold)

    return n, d, m


def get_split(
    angles: TT["query_batch dims"],
    comparisons: TT["query_batch dims key_batch"],
    labels: TT["query_batch n_classes"],
    n: int,
    d: int,
) -> Tuple[
    TT["query_batch dims key_batch"],
    TT["query_batch n_classes"],
    TT["query_batch dims key_batch"],
    TT["query_batch n_classes"],
]:
    """
    Split comparisons and labels into negative and positive classes

    Args:
        angles: (query_batch, dims) tensor of angles
        comparisons: (query_batch, dims, key_batch) tensor of comparisons
        labels: (query_batch, n_classes) tensor of one-hot labels
        n: scalar index of split
        d: scalar dimension of split

    Returns:
        angle_neg: (query_batch_neg, dims) tensor of angles for negative class
        angles_pos: (query_batch_pos, dims) tensor of angles for positive class
        comparisons_neg: (query_batch_neg, dims, key_batch_neg) tensor of comparisons for negative class
        labels_neg: (query_batch_neg, n_classes) tensor of one-hot labels for negative class
        comparisons_pos: (query_batch_pos, dims, key_batch_pos) tensor of comparisons for positive class
        labels_pos: (query_batch_pos, n_classes) tensor of one-hot labels for positive class
    """
    # Get the split mask without creating an intermediate boolean tensor
    mask = comparisons[n, d].bool()

    # Use torch.where to avoid creating intermediate tensors
    # Bear in mind, "mask" is typically a float, so we have to be clever
    pos_indices = torch.where(mask)[0]
    neg_indices = torch.where(~mask)[0]

    # Split the comparisons and labels using advanced indexing
    angles_neg = angles[neg_indices]
    angles_pos = angles[pos_indices]
    comparisons_neg = comparisons[neg_indices][:, :, neg_indices]
    comparisons_pos = comparisons[pos_indices][:, :, pos_indices]
    labels_neg = labels[neg_indices]
    labels_pos = labels[pos_indices]

    return angles_neg, angles_pos, comparisons_neg, labels_neg, comparisons_pos, labels_pos


# Copied over from hyperdt.torch.tree
class DecisionNode:
    def __init__(self, value=None, probs=None, feature=None, theta=None, left=None, right=None):
        self.value = value
        self.probs = probs  # predicted class probabilities of all samples in the leaf
        self.feature = feature  # feature index
        self.theta = theta  # threshold
        self.left = None
        self.right = None


class ProductSpaceDT(BaseEstimator, ClassifierMixin):
    def __init__(self, pm, max_depth=3, min_samples_leaf=1, min_samples_split=2, **kwargs):
        # Store hyperparameters
        self.pm = pm
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def preprocess(
        self,
        X: TT["batch ambient_dim"],
        y: Optional[TT["batch"]] = None,
    ) -> Tuple[TT["batch intrinsic_dim"], TT["batch n_classes"], TT["n_classes"], TT["batch intrinsic_dim", "batch"]]:
        """
        Preprocessing function for the new version of ProductDT

        Args:
            X: (batch, ambient_dim) tensor of coordinates
            y: (batch,) tensor of labels

        Outputs:
            X: (batch, intrinsic_dim) tensor of angles
            y: (batch, n_classes) tensor of one-hot labels
            classes: (n_classes,) tensor of classes with original labels
            M: (batch, intrinsic_dim, batch) tensor of comparisons
        """
        # Ensure X and y are tensors
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        if y is not None and not torch.is_tensor(y):
            y = torch.tensor(y)

        # Assertions: input validation
        assert X.dim() == 2
        if y is not None:
            assert y.dim() == 1
            assert X.shape[0] == y.shape[0]

        # Process X-values into angles based on the signature
        angles = torch.zeros((X.shape[0], self.pm.dim), device=X.device)
        for i, M in enumerate(self.pm.P):
            if M.type in ["H", "S"]:
                idx0 = self.pm.man2intrinsic[i]
                idx1 = self.pm.man2dim[i][0]
                idx2 = self.pm.man2dim[i][1:]
                angles[:, idx0] = torch.atan2(X[:, idx1 : idx1 + 1], X[:, idx2])
            elif M.type == "E":
                idx0 = self.pm.man2intrinsic[i]
                idx1 = self.pm.man2dim[i]
                angles[:, idx0] = torch.atan2(torch.ones(1, device=X.device), X[:, idx1])

        # Create a tensor of comparisons
        comparisons = ((angles[:, None] - angles[None, :] + torch.pi) % (2 * torch.pi)) >= torch.pi

        # Reshape the comparisons tensor
        comparisons_reshaped = comparisons.permute(0, 2, 1)

        # One-hot encode labels
        # n_classes = y.unique().numel()
        if y is not None:
            classes, y_relabeled = y.unique(return_inverse=True)
            n_classes = len(classes)
            labels_onehot = torch.nn.functional.one_hot(y_relabeled, num_classes=n_classes)
        else:
            classes = torch.tensor([])
            labels_onehot = torch.tensor([])

        return angles, labels_onehot.float(), classes, comparisons_reshaped.float()

    @torch.no_grad()
    def fit(self, X, y):
        """Reworked fit function for new version of ProductDT"""

        # Preprocess data
        angles, labels_onehot, classes, comparisons_reshaped = self.preprocess(X=X, y=y)

        # Store classes for predictions
        self.classes_ = classes

        # Fit node
        self.tree = self.fit_node(angles, labels_onehot, comparisons_reshaped, self.max_depth)

    def fit_node(self, angles, labels, comparisons, depth):
        # Check halting conditions
        if depth == 0 or len(comparisons) < self.min_samples_split:
            value, probs = self._leaf_values(labels)
            return DecisionNode(value=value.item(), probs=probs)

        # The main loop is just the functions we've already defined
        ig = get_info_gains(comparisons=comparisons, labels=labels)
        n, d, theta = get_best_split(ig=ig, angles=angles, comparisons=comparisons, pm=self.pm)
        angles_neg, angles_pos, comparisons_neg, labels_neg, comparisons_pos, labels_pos = get_split(
            angles=angles, comparisons=comparisons, labels=labels, n=n, d=d
        )
        value, probs = self._leaf_values(labels)
        node = DecisionNode(
            value=value.item(),
            probs=probs,
            feature=d.item(),
            theta=theta.item(),
            left=self.fit_node(angles_neg, labels_neg, comparisons_neg, depth - 1),
            right=self.fit_node(angles_pos, labels_pos, comparisons_pos, depth - 1),
        )
        return node

    def _leaf_values(self, y):
        y_sum = y.sum(dim=0)
        value = torch.argmax(y_sum)
        probs = y_sum / y_sum.sum()
        return value, probs

    def _left(self, angle, node):
        """Boolean: Go left?"""
        return (angle[:, node.feature] - node.theta + torch.pi) % (2 * torch.pi) >= torch.pi

    def _traverse(self, x, node=None):
        """Traverse a decision tree for a single point"""
        # Root case
        if node is None:
            node = self.tree

        # Leaf case
        if node.value is not None:
            return node

        return self._traverse(x, node.left) if self._left(x, node) else self._traverse(x, node.right)

    def predict(self, X):
        # angle_vals = self._get_angle_vals(X)
        angles, labels_onehot, classes, comparisons_reshaped = self.preprocess(X=X)
        return self.classes_[torch.tensor([self._traverse(x).value for x in angles], device=X.device)]

    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        return torch.tensor([self._traverse(x).probs for x in X])

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return self.predict(X) == y


# class TorchProductSpaceRF(BaseEstimator, ClassifierMixin):
#     def __init__(
#         self,
#         signature,
#         n_estimators=100,
#         max_features="sqrt",
#         max_samples=1.0,
#         random_state=None,
#         max_depth=3,
#         n_jobs=-1,
#     ):
#         self.n_estimators = n_estimators
#         self.max_features = max_features
#         self.max_samples = max_samples
#         self.random_state = random_state
#         self.max_depth = max_depth
#         self.n_jobs = n_jobs
#         self.trees = [TorchProductSpaceDT(signature, max_depth=max_depth) for _ in range(n_estimators)]

#     def _generate_subsample(self, X, y):
#         n_samples = X.shape[0]
#         sample_size = int(n_samples * self.max_samples)
#         # indices = np.random.choice(n_samples, size=sample_size, replace=True)
#         indices = torch.randint(0, n_samples, (sample_size,))
#         return X[indices], y[indices]

#     def fit(self, X, y):
#         for tree in self.trees:
#             X_sample, y_sample = self._generate_subsample(X, y)
#             tree.fit(X_sample, y_sample)
#         return self

#     # def fit(self, X, y, use_tqdm=False, seed=None):
#     #     """Fit a decision tree to subsamples"""
#     #     self.classes_ = torch.unique(y)

#     #     if seed is not None:
#     #         self.random_state = seed
#     #     if self.random_state is not None:
#     #         # np.random.seed(self.random_state)
#     #         torch.manual_seed(self.random_state)

#     #     # Fit decision trees individually (parallelized):
#     #     trees = tqdm(self.trees) if use_tqdm else self.trees
#     #     if self.n_jobs != 1:
#     #         fitted_trees = Parallel(n_jobs=self.n_jobs)(
#     #             delayed(tree.fit)(*self._generate_subsample(X, y)) for tree in trees
#     #         )
#     #         self.trees = fitted_trees
#     #     else:
#     #         for tree in trees:
#     #             X_sample, y_sample = self._generate_subsample(X, y)
#     #             tree.fit(X_sample, y_sample)
#     #     return self

#     def predict(self, X):
#         predictions = torch.stack([tree.predict(X) for tree in self.trees])
#         return torch.mode(predictions, dim=0).values

#     def predict_proba(self, X):
#         predictions = torch.stack([tree.predict_proba(X) for tree in self.trees])
#         return predictions.mean(dim=0)

#     def score(self, X, y):
#         return torch.mean(self.predict(X) == y)
