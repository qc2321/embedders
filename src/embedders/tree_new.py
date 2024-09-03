import torch

# from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin

from .midpoint import midpoint

# Typing stuff
from torchtyping import TensorType as TT
from typing import Tuple, Optional, Literal


def _angular_greater(queries: TT["query_batch"], keys: TT["key_batch"]) -> TT["query_batch key_batch"]:
    """
    Given an angle theta, check whether a tensor of inputs is in [theta, theta + pi)

    Args:
        queries: (query_batch,) tensor of angles used to define a decision hyperplane
        keys: (key_batch,) tensor of angles to be compared to queries

    Outputs:
        comparisons: (query_batch, key_batch) tensor of Booleans checking whether each key is in range
        [query, query + pi)
    """
    # return ((keys[None, :] - queries[:, None] + torch.pi) % (2 * torch.pi)) >= torch.pi

    # Optimized version
    diff = keys.unsqueeze(0) - queries.unsqueeze(1)
    return (diff + torch.pi) % (2 * torch.pi) >= torch.pi


def _get_info_gains(
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


def _get_split(
    angles: TT["query_batch dims"],
    comparisons: TT["query_batch dims key_batch"],
    labels: TT["query_batch n_classes"],
    n: int,
    d: int,
) -> Tuple[
    Tuple[TT["query_batch_neg dims"], TT["query_batch_neg dims key_batch"], TT["query_batch_neg n_classes"]],
    Tuple[TT["query_batch_pos dims"], TT["query_batch_pos dims key_batch"], TT["query_batch_pos n_classes"]],
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
        neg = (
            angle_neg: (query_batch_neg, dims) tensor of angles for negative class
            comparisons_neg: (query_batch_neg, dims, key_batch_neg) tensor of comparisons for negative class
            labels_neg: (query_batch_neg, n_classes) tensor of one-hot labels for negative class
        ),
        pos = (
            angles_pos: (query_batch_pos, dims) tensor of angles for positive class
            comparisons_pos: (query_batch_pos, dims, key_batch_pos) tensor of comparisons for positive class
            labels_pos: (query_batch_pos, n_classes) tensor of one-hot labels for positive class
        )
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

    return (angles_neg, comparisons_neg, labels_neg), (angles_pos, comparisons_pos, labels_pos)


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

        # These will become important later
        self.nodes = []  # For fitted nodes
        self.permutations = None  # If used as part of a random forest

    def _preprocess(
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
            target_idx = self.pm.man2intrinsic[i]
            dims = self.pm.man2dim[i]
            if M.type in ["H", "S"]:
                num = X[:, dims[0] : dims[0] + 1]
                denom = X[:, dims[1:]]
            elif M.type == "E":
                num = torch.ones(1, device=X.device)
                denom = X[:, dims]
            angles[:, target_idx] = torch.atan2(num, denom)

        # Create a tensor of comparisons
        comparisons = _angular_greater(angles, angles)

        # Reshape the comparisons tensor
        # comparisons_reshaped = comparisons.permute(0, 2, 1)
        comparisons_reshaped = comparisons.permute(1, 2, 0)

        # One-hot encode labels
        if y is not None:
            classes, y_relabeled = y.unique(return_inverse=True)
            n_classes = len(classes)
            labels_onehot = torch.nn.functional.one_hot(y_relabeled, num_classes=n_classes)
        else:
            classes = torch.tensor([])
            labels_onehot = torch.tensor([])

        return angles, labels_onehot.float(), classes, comparisons_reshaped.float()

    def _get_best_split(
        self,
        ig: TT["query_batch dims"],
        angles: TT["query_batch intrinsic_dim"],
        comparisons: TT["query_batch dims key_batch"],
    ) -> Tuple[int, int, float]:
        """
        All of the postprocessing for an information gain check

        Args:
            ig: (query_batch, dims) tensor of information gains
            angles: (query_batch, dims) tensor of angles
            comparisons: (query_batch, dims, key_batch) tensor of comparisons

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
        if (comparisons[n, d] == 1.0).all():
            theta_neg = theta_pos
            # TODO: replace this with something better, e.g. make "circular_greater" strict and make the pos_class the
            # smallest theta where this is 1.
        else:
            n_neg = (angles[comparisons[n, d] == 0.0, d] - theta_pos).abs().argmin()
            theta_neg = angles[comparisons[n, d] == 0.0, d][n_neg]

        # Get manifold
        if self.permutations is not None:
            active_dim = self.permutations[d].item()
        else:
            active_dim = d.item()
        manifold = self.pm.P[self.pm.intrinsic2man[active_dim]]

        # Get midpoint
        m = midpoint(theta_pos, theta_neg, manifold)

        return n, d, m

    @torch.no_grad()
    def fit(self, X: TT["batch ambient_dim"], y: TT["batch"]) -> None:
        """
        Reworked fit function for new version of ProductDT

        Args:
            X: (batch, ambient_dim) tensor of trainind data (ambient coordinate representation)
            y: (batch,) tensor of labels (integer representation)

        Returns:
            None (fits tree in place)
        """

        # Preprocess data
        angles, labels_onehot, classes, comparisons_reshaped = self._preprocess(X=X, y=y)

        # Store classes for predictions
        self.classes_ = classes

        # Fit node
        self.tree = self._fit_node(
            angles=angles, labels=labels_onehot, comparisons=comparisons_reshaped, depth=self.max_depth
        )

    def _fit_node(
        self,
        angles: TT["batch intrinsic_dim"],
        labels: TT["batch n_classes"],
        comparisons: TT["query_batch dim key_batch"],
        depth: int,
    ) -> DecisionNode:
        """
        The recursive component of the product space decision tree fitting function

        Args:

        """
        # Check halting conditions
        if depth == 0 or len(comparisons) < self.min_samples_split:
            probs, value = self._leaf_values(labels)
            return DecisionNode(value=value.item(), probs=probs)

        # The main loop is just the functions we've already defined
        ig = _get_info_gains(comparisons=comparisons, labels=labels)
        n, d, theta = self._get_best_split(ig=ig, angles=angles, comparisons=comparisons)
        (angles_neg, comparisons_neg, labels_neg), (angles_pos, comparisons_pos, labels_pos) = _get_split(
            angles=angles, comparisons=comparisons, labels=labels, n=n, d=d
        )
        node = DecisionNode(feature=d.item(), theta=theta.item())
        self.nodes.append(node)

        # Do left and right recursion after appending node to self.nodes
        # This ensures that the order of self.nodes is correct
        node.left = self._fit_node(angles=angles_neg, labels=labels_neg, comparisons=comparisons_neg, depth=depth - 1)
        node.right = self._fit_node(angles=angles_pos, labels=labels_pos, comparisons=comparisons_pos, depth=depth - 1)
        return node

    def _leaf_values(self, y: TT["batch"]) -> Tuple[TT["batch", "n_classes"], TT["batch"]]:
        """Get majority class and class probabilities"""
        probs = y.sum(dim=0) / y.sum()
        return probs, probs.argmax()

    def _left(self, angles_row: TT["intrinsic_dim"], node: DecisionNode) -> bool:
        """Boolean: Go left? Works on a preprocessed input vector."""
        return _angular_greater(torch.tensor(node.theta).flatten(), angles_row[node.feature].flatten()).item()

    def _traverse(self, x: TT["intrinsic_dim"], node: DecisionNode) -> DecisionNode:
        """Traverse a decision tree for a single point"""
        # Leaf case
        if node.value is not None:
            return node

        return self._traverse(x, node.left) if self._left(x, node) else self._traverse(x, node.right)

    def predict_proba(self, X: TT["batch intrinsic_dim"]) -> TT["batch n_classes"]:
        """Predict class probabilities for samples in X"""
        angles, _, _, _ = self._preprocess(X=X)
        if self.permutations is not None:
            angles = angles[:, self.permutations]
        return torch.vstack([self._traverse(angles_row, self.tree).probs for angles_row in angles])

    def predict(self, X: TT["batch intrinsic_dim"]) -> TT["batch"]:
        """Predict class labels for samples in X"""
        return self.classes_[self.predict_proba(X).argmax(dim=1)]

    def score(self, X: TT["batch intrinsic_dim"], y: TT["batch"]) -> TT["batch"]:
        """Return the mean accuracy on the given test data and labels"""
        return self.predict(X) == y


class ProductSpaceRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pm,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        max_features="sqrt",
        max_samples=1.0,
        random_state=None,
        n_jobs=-1,
        **kwargs,
    ):
        # Store hyperparameters
        self.pm = pm
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.trees = [ProductSpaceDT(pm=pm, **kwargs) for _ in range(n_estimators)]

    def _generate_subsample(self, n_rows, n_cols, n_trees):
        # Get number of dimensions in our subsample
        if isinstance(self.max_features, int) and self.max_features <= n_cols:
            n_cols_sample = n_cols
        elif self.max_features == "sqrt":
            n_cols_sample = torch.ceil(torch.tensor(n_cols**0.5)).int()
        elif self.max_features == "log2":
            n_cols_sample = torch.ceil(torch.log2(torch.tensor(d))).int()
        elif self.max_features == "none":
            n_cols_sample = n_cols
        else:
            raise ValueError(f"Unknown max_features parameter: {self.max_features}")

        # Subsample - returns indices
        idx_sample = torch.randint(0, n_rows, (n_trees, n_rows))
        idx_dim = torch.stack([torch.randperm(n_cols)[:n_cols_sample] for _ in range(n_trees)])

        return idx_sample, idx_dim

    def fit(self, X: TT["batch ambient_dim"], y: TT["batch"]):
        """Preprocess and fit an ensemble of trees on subsampled data"""
        # Can use any tree to preprocess X and y
        angles, labels, classes, comparisons = self.trees[0]._preprocess(X=X, y=y)
        self.classes_ = classes

        # Subsample - just the indices
        n, d = angles.shape
        idx_sample_all, idx_dim_all = self._generate_subsample(n_rows=n, n_cols=d, n_trees=self.n_estimators)

        # Fit trees
        for tree, idx_sample, idx_dim in zip(self.trees, idx_sample_all, idx_dim_all):
            tree.permutations = idx_dim
            tree.classes_ = classes
            tree.tree = tree._fit_node(
                angles=angles[idx_sample][:, idx_dim],
                labels=labels[idx_sample],
                comparisons=comparisons[idx_sample][:, idx_dim][:, :, idx_sample],
                depth=self.max_depth,
            )
        return self

    def predict_proba(self, X: TT["batch intrinsic_dim"]) -> TT["batch n_classes"]:
        """Predict class probabilities for samples in X"""
        return torch.stack([tree.predict_proba(X) for tree in self.trees]).mean(dim=0)

    def predict(self, X: TT["batch intrinsic_dim"]) -> TT["batch"]:
        """Predict class labels for samples in X"""
        return self.classes_[self.predict_proba(X).argmax(dim=1)]

    def score(self, X: TT["batch intrinsic_dim"], y: TT["batch"]) -> TT["batch"]:
        """Return the mean accuracy on the given test data and labels"""
        return self.predict(X) == y
