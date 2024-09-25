import torch

# from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin

from tqdm.notebook import tqdm

from .midpoint import midpoint
from .manifolds import ProductManifold

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
    criterion: Literal["gini", "mse"] = "gini",
    min_values_leaf: int = 1,
    eps: float = 1e-10,
) -> TT["query_batch dims"]:
    """
    Given comparisons matrix and labels, return information gain for each possible split.

    Args:
        comparisons: (query_batch, dims, key_batch) tensor of comparisons
        labels: (query_batch, n_classes) tensor of one-hot labels
        eps: small number to prevent division by zero
        half: whether to use float16

    Outputs:
        ig: (query_batch, dims) tensor of information gains
    """
    # Matrix-multiply to get counts of labels in left and right splits
    if criterion == "gini":
        pos_labels = (comparisons @ labels).float()
        neg_labels = ((1 - comparisons) @ labels).float()

        # Total counts are sums of label counts
        n_pos = pos_labels.sum(dim=-1) + eps
        n_neg = neg_labels.sum(dim=-1) + eps
        n_total = n_pos + n_neg

        # Probabilities are label counts divided by total counts, when Gini is used
        pos_probs = pos_labels / n_pos.unsqueeze(-1)
        neg_probs = neg_labels / n_neg.unsqueeze(-1)
        total_probs = (pos_labels + neg_labels) / n_total.unsqueeze(-1)

        # Gini impurity is 1 - sum(prob^2)
        gini_pos = 1 - (pos_probs**2).sum(dim=-1)
        gini_neg = 1 - (neg_probs**2).sum(dim=-1)
        gini_total = 1 - (total_probs**2).sum(dim=-1)

    # For MSE, use the mean of the regression labels to compute MSE (i.e. look at variance)
    elif criterion == "mse":
        pos_sums = (comparisons @ labels).float()
        neg_sums = ((1 - comparisons) @ labels).float()

        # Total counts are sums of comparisons
        n_pos = comparisons.sum(dim=-1) + eps
        n_neg = (1 - comparisons).sum(dim=-1) + eps
        n_total = n_pos + n_neg

        # Means should be computed in a slightly odd way, since we want to use n_pos and n_neg
        pos_means = pos_sums / n_pos
        neg_means = neg_sums / n_neg
        all_means = labels.mean()  # Should be a scalar

        # Compute MSE using the comparisons and the means
        pos_se = ((labels[:, None, None] - pos_means) ** 2).permute(1, 2, 0)
        neg_se = ((labels[:, None, None] - neg_means) ** 2).permute(1, 2, 0)
        pos_mse = (comparisons * pos_se).sum(dim=-1) / n_pos
        neg_mse = ((1 - comparisons) * neg_se).sum(dim=-1) / n_neg
        total_mse = ((labels - all_means) ** 2).mean()

        # Just reuse these variable names for now
        gini_pos, gini_neg, gini_total = pos_mse, neg_mse, total_mse

    # Information gain is the total gini impurity minus the weighted average of the new gini impurities
    ig = gini_total - (gini_pos * n_pos + gini_neg * n_neg) / n_total

    assert not ig.isnan().any()  # Ensure no NaNs

    # Set information gain to zero if the split is invalid
    invalid_mask = torch.logical_or(n_pos < min_values_leaf, n_neg < min_values_leaf)
    ig[invalid_mask] = 0.0

    return ig


def _get_info_gains_nobatch(
    angles: TT["batch n_dims"],
    labels: TT["batch n_classes"],
    criterion: Literal["gini", "mse"] = "gini",
    min_values_leaf: int = 1,
    eps: float = 1e-10,
    batch_size=256,
) -> TT["batch dims"]:
    """
    Given angles matrix and labels, return information gain for each possible split.

    Args:
        angles: (batch, dims) tensor of angles
        labels: (query_batch, n_classes) tensor of one-hot labels
        eps: small number to prevent division by zero
        half: whether to use float16

    Outputs:
        ig: (query_batch, dims) tensor of information gains
    """
    # Matrix-multiply to get counts of labels in left and right splits
    if criterion == "gini":
        pos_labels = torch.zeros((angles.shape[0], angles.shape[1], labels.shape[1]), device=angles.device)
        neg_labels = torch.zeros((angles.shape[0], angles.shape[1], labels.shape[1]), device=angles.device)

        my_tqdm = tqdm(total=angles.shape[0] * angles.shape[1], desc="Calculating information gains", leave=False)
        for d in range(angles.shape[1]):
            for batch_start in range(0, angles.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, angles.shape[0])
                i_batch = torch.arange(batch_start, batch_end, device=angles.device)

                # Expanding angles for broadcasting over the batch
                angles_d = (
                    angles[:, d].unsqueeze(0).expand(batch_end - batch_start, -1)
                )  # [batch_size, angles.shape[0]]
                angles_i_d = angles[i_batch, d].unsqueeze(1)  # [batch_size, 1]

                mask = _angular_greater(angles_d, angles_i_d)  # [batch_size, angles.shape[0]]

                # Expanding the labels to match the broadcasting needs of the mask
                pos_labels_entry = torch.matmul(mask.float(), labels)  # [batch_size, labels.shape[1]]
                neg_labels_entry = torch.matmul((~mask).float(), labels)  # [batch_size, labels.shape[1]]

                # Assign the calculated values to the respective positions in the final tensors
                pos_labels[i_batch, d, :] = pos_labels_entry.sum(dim=0)
                neg_labels[i_batch, d, :] = neg_labels_entry.sum(dim=0)

                my_tqdm.update(batch_end - batch_start)

        # Total counts are sums of label counts
        n_pos = pos_labels.sum(dim=-1) + eps
        n_neg = neg_labels.sum(dim=-1) + eps
        n_total = n_pos + n_neg

        # Probabilities are label counts divided by total counts, when Gini is used
        pos_probs = pos_labels / n_pos.unsqueeze(-1)
        neg_probs = neg_labels / n_neg.unsqueeze(-1)
        total_probs = (pos_labels + neg_labels) / n_total.unsqueeze(-1)

        # Gini impurity is 1 - sum(prob^2)
        gini_pos = 1 - (pos_probs**2).sum(dim=-1)
        gini_neg = 1 - (neg_probs**2).sum(dim=-1)
        gini_total = 1 - (total_probs**2).sum(dim=-1)

    # For MSE, use the mean of the regression labels to compute MSE (i.e. look at variance)
    elif criterion == "mse":
        pos_sums = (comparisons @ labels).float()
        neg_sums = ((1 - comparisons) @ labels).float()

        # Total counts are sums of comparisons
        n_pos = comparisons.sum(dim=-1) + eps
        n_neg = (1 - comparisons).sum(dim=-1) + eps
        n_total = n_pos + n_neg

        # Means should be computed in a slightly odd way, since we want to use n_pos and n_neg
        pos_means = pos_sums / n_pos
        neg_means = neg_sums / n_neg
        all_means = labels.mean()  # Should be a scalar

        # Compute MSE using the comparisons and the means
        pos_se = ((labels[:, None, None] - pos_means) ** 2).permute(1, 2, 0)
        neg_se = ((labels[:, None, None] - neg_means) ** 2).permute(1, 2, 0)
        pos_mse = (comparisons * pos_se).sum(dim=-1) / n_pos
        neg_mse = ((1 - comparisons) * neg_se).sum(dim=-1) / n_neg
        total_mse = ((labels - all_means) ** 2).mean()

        # Just reuse these variable names for now
        gini_pos, gini_neg, gini_total = pos_mse, neg_mse, total_mse

    # Information gain is the total gini impurity minus the weighted average of the new gini impurities
    ig = gini_total - (gini_pos * n_pos + gini_neg * n_neg) / n_total

    assert not ig.isnan().any()  # Ensure no NaNs

    # Set information gain to zero if the split is invalid
    invalid_mask = torch.logical_or(n_pos < min_values_leaf, n_neg < min_values_leaf)
    ig[invalid_mask] = 0.0

    return ig


def _get_split(
    mask: TT["query_batch"],
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
        mask: (query_batch, key_batch) tensor of booleans
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
    # mask = comparisons[n, d].bool()

    # Use torch.where to avoid creating intermediate tensors
    # Bear in mind, "mask" is typically a float, so we have to be clever
    pos_indices = torch.where(mask)[0]
    neg_indices = torch.where(~mask)[0]

    # Split the comparisons and labels using advanced indexing
    angles_neg = angles[neg_indices]
    angles_pos = angles[pos_indices]
    if len(comparisons) != 0:  # Handle nonbatched case better
        comparisons_neg = comparisons[neg_indices][:, :, neg_indices]
        comparisons_pos = comparisons[pos_indices][:, :, pos_indices]
    else:
        comparisons_pos = comparisons_neg = comparisons
    labels_neg = labels[neg_indices]
    labels_pos = labels[pos_indices]

    return (angles_neg, comparisons_neg, labels_neg), (angles_pos, comparisons_pos, labels_pos)


# Copied over from hyperdt.torch.tree
class DecisionNode:
    def __init__(self, value=None, probs=None, feature=None, theta=None):
        self.value = value
        self.probs = probs  # predicted class probabilities of all samples in the leaf
        self.feature = feature  # feature index
        self.theta = theta  # threshold
        self.left = None
        self.right = None


class ProductSpaceDT(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pm,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        task: Literal["classification", "regression"] = "classification",
        use_special_dims=False,
        batch_size=None,
        n_features: Literal["d", "d_choose_2"] = "d",
    ):
        # Store hyperparameters
        self.pm = pm
        if max_depth is None:
            self.max_depth = -1  # This runs forever since the loop checks depth == 0
        else:
            self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.use_special_dims = use_special_dims
        self.n_features = n_features

        # I use "batched" to mean "all at once" and "batch_size" to mean "in chunks"
        self.batch_size = batch_size
        if batch_size is not None:
            self.batched = False
        else:
            self.batched = True

        # Task-specific stuff
        self.task = task
        self.criterion = "gini" if task == "classification" else "mse"

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
            include_special_dims: whether to include special dimensions as a new Euclidean component

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

            # If y is floats, warn about regression
            if self.task == "classification" and not torch.allclose(y, y.round()):
                print("Warning: y contains non-integer values. Try regression instead.")

        # Process X-values into angles based on the signature
        angles = []
        angle2man = []
        special_first = []
        for i, M in enumerate(self.pm.P):
            dims = self.pm.man2dim[i]

            # Non-Euclidean manifolds use angular projections
            if M.type in ["H", "S"]:
                if self.n_features == "d":
                    dim = dims[0]
                    num = X[:, dim : dim + 1]
                    denom = X[:, dims[1:]]
                    angles.append(torch.atan2(num, denom))
                    special_first += [True] * (len(dims) - 1)
                    angle2man += [i] * (len(dims) - 1)

                elif self.n_features == "d_choose_2":
                    for j, dim in enumerate(dims[:-1]):
                        num = X[:, dim : dim + 1]
                        denom = X[:, dims[j + 1 :]]
                        angles.append(torch.atan2(num, denom))
                        special_first += [j == 0] * (len(dims) - j - 1)
                        angle2man += [i] * (len(dims) - j - 1)

            # Euclidean manifolds use a dummy dimension to get an angle
            elif M.type == "E":
                num = torch.ones(1, device=X.device, dtype=X.dtype)
                denom = X[:, dims]
                angles.append(torch.atan2(num, denom))
                special_first += [True] * len(dims)
                angle2man += [i] * len(dims)

                if self.n_features == "d_choose_2":
                    # Note that we do the entire loop over dims here
                    # This is because we faked a dimension at the start
                    # That's also why we don't subtract 1 from len(dims)
                    for j, dim in enumerate(dims[:-1]):
                        num = X[:, dim : dim + 1]
                        denom = X[:, dims[j + 1 :]]
                        angles.append(torch.atan2(num, denom))
                        special_first += [False] * (len(dims) - j)
                        angle2man += [i] * (len(dims) - j)

        angles = torch.cat(angles, dim=1)
        self.angle2man = angle2man
        self.special_first = special_first

        # Create a tensor of comparisons
        if self.batched:
            comparisons = _angular_greater(angles, angles)
            comparisons_reshaped = comparisons.permute(1, 2, 0)
        else:
            comparisons_reshaped = torch.tensor([])

        # One-hot encode labels
        if y is not None and self.task == "classification":
            classes, y_relabeled = y.unique(return_inverse=True)
            n_classes = len(classes)
            labels = torch.nn.functional.one_hot(y_relabeled, num_classes=n_classes)
        elif y is not None and self.task == "regression":
            classes = torch.tensor([])
            labels = y
        else:
            classes = torch.tensor([])
            labels = torch.tensor([])
        labels = labels.to(dtype=X.dtype)
        # labels = labels.to_sparse()
        comparisons_reshaped = comparisons_reshaped.to(dtype=X.dtype)
        # comparisons_reshaped = comparisons_reshaped.to_sparse()

        return angles, labels, classes, comparisons_reshaped

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
        if self.batched:
            angle_comparisons = comparisons[n, d]
        else:
            angle_comparisons = _angular_greater(angles[:, d], theta_pos).flatten()
        if (angle_comparisons == 1.0).all():
            theta_neg = theta_pos
            # TODO: replace this with something better, e.g. make "circular_greater" strict and make the pos_class the
            # smallest theta where this is 1.
        else:
            n_neg = (angles[angle_comparisons == 0.0, d] - theta_pos).abs().argmin()
            theta_neg = angles[angle_comparisons == 0.0, d][n_neg]

        # Get manifold
        if self.permutations is not None:
            active_dim = self.permutations[d].item()
        else:
            active_dim = d.item()
        # manifold = self.pm.P[self.pm.intrinsic2man[active_dim]]
        manifold = self.pm.P[self.angle2man[active_dim]]
        special_first_bool = self.special_first[active_dim]

        # Get midpoint
        m = midpoint(theta_pos, theta_neg, manifold, special_first=special_first_bool)

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

        # Pre-preprocessing step: aggregate special dimensions into a new Euclidean component
        if self.use_special_dims:
            X, self.pm = self._aggregate_special_dims(X)

        # Preprocess data
        angles, labels, classes, comparisons_reshaped = self._preprocess(X=X, y=y)

        # Store classes for predictions
        self.classes_ = classes

        # Fit node
        self.tree = self._fit_node(angles=angles, labels=labels, comparisons=comparisons_reshaped, depth=self.max_depth)

    def _aggregate_special_dims(self, X: TT["batch ambient_dim"]) -> Tuple[TT["batch ambient_dim"], ProductManifold]:
        special_dims = []
        for i, M in enumerate(self.pm.P):
            if M.type in ["H", "S"]:
                dim = self.pm.man2dim[i][0]
                special_dims.append(X[:, dim : dim + 1])
        if len(special_dims) > 0:
            X = torch.cat([X] + special_dims, dim=1)
            self.signature = self.pm.signature + [(0, len(special_dims))]
        else:
            self.signature = self.pm.signature
        return X, ProductManifold(self.signature)

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

        def _halt(labels):
            probs, value = self._leaf_values(labels)
            node = DecisionNode(value=value.item(), probs=probs)
            self.nodes.append(node)
            return node

        # Check halting conditions
        if depth == 0 or labels.shape[0] < self.min_samples_split:
            return _halt(labels)

        # The main loop is just the functions we've already defined
        if self.batched:
            ig = _get_info_gains(comparisons=comparisons, labels=labels, criterion=self.criterion)
        else:
            ig = _get_info_gains_nobatch(
                angles=angles, labels=labels, criterion=self.criterion, batch_size=self.batch_size
            )

        # Check if we have a valid split
        if ig.max() <= self.min_impurity_decrease:
            return _halt(labels)

        # Get the best split
        n, d, theta = self._get_best_split(ig=ig, angles=angles, comparisons=comparisons)
        if self.batched:
            mask = comparisons[n, d].bool()
        else:
            mask = _angular_greater(angles[:, d], theta).flatten()
        (angles_neg, comparisons_neg, labels_neg), (angles_pos, comparisons_pos, labels_pos) = _get_split(
            mask=mask, angles=angles, comparisons=comparisons, labels=labels, n=n, d=d
        )
        node = DecisionNode(feature=d.item(), theta=theta.item())
        self.nodes.append(node)

        # Do left and right recursion after appending node to self.nodes (ensures order of self.nodes is correct)
        node.left = self._fit_node(angles=angles_neg, labels=labels_neg, comparisons=comparisons_neg, depth=depth - 1)
        node.right = self._fit_node(angles=angles_pos, labels=labels_pos, comparisons=comparisons_pos, depth=depth - 1)
        return node

    def _leaf_values(self, y: TT["batch"]) -> Tuple[TT["batch", "n_classes"], TT["batch"]]:
        """Get majority class and class probabilities"""
        if self.task == "regression":
            return y.mean(), y.mean()
        else:
            probs = y.sum(dim=0) / y.sum()
            return probs, probs.argmax()

    def _left(self, angles_row: TT["intrinsic_dim"], node: DecisionNode) -> bool:
        """Boolean: Go left? Works on a preprocessed input vector."""
        return _angular_greater(
            torch.tensor(node.theta, device=angles_row.device).flatten(), angles_row[node.feature].flatten()
        ).item()

    def _traverse(self, x: TT["intrinsic_dim"], node: DecisionNode) -> DecisionNode:
        """Traverse a decision tree for a single point"""
        # Leaf case
        if node.value is not None:
            return node

        return self._traverse(x, node.left) if self._left(x, node) else self._traverse(x, node.right)

    @torch.no_grad()
    def predict_proba(self, X: TT["batch intrinsic_dim"]) -> TT["batch n_classes"]:
        """Predict class probabilities for samples in X"""
        if self.use_special_dims:
            X, _ = self._aggregate_special_dims(X)
        angles, _, _, _ = self._preprocess(X=X)
        if self.permutations is not None:
            angles = angles[:, self.permutations]
        return torch.vstack([self._traverse(angles_row, self.tree).probs for angles_row in angles])

    def predict(self, X: TT["batch intrinsic_dim"]) -> TT["batch"]:
        """Predict class labels for samples in X"""
        if self.task == "classification":
            return self.classes_[self.predict_proba(X).argmax(dim=1)]
        else:
            return self.predict_proba(X).flatten()  # Simplified version

    def score(self, X: TT["batch intrinsic_dim"], y: TT["batch"]) -> TT["batch"]:
        """Return the mean accuracy on the given test data and labels"""
        if self.task == "classification":
            return self.predict(X) == y
        else:
            return ((self.predict(X) - y) ** 2).float().mean()


class ProductSpaceRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pm,
        task: Literal["classification", "regression"] = "classification",
        use_special_dims=False,
        n_features: Literal["d", "d_choose_2"] = "d",
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        n_estimators=100,
        max_features="sqrt",
        max_samples=1.0,
        batch_size=None,
        random_state=None,
        n_jobs=-1,
    ):
        # Tree hyperparameters
        tree_kwargs = {}
        self.pm = tree_kwargs["pm"] = pm
        self.task = tree_kwargs["task"] = task
        if max_depth is None:
            self.max_depth = tree_kwargs["max_depth"] = -1
        else:
            self.max_depth = tree_kwargs["max_depth"] = max_depth
        self.min_samples_leaf = tree_kwargs["min_samples_leaf"] = min_samples_leaf
        self.min_samples_split = tree_kwargs["min_samples_split"] = min_samples_split
        self.min_impurity_decrease = tree_kwargs["min_impurity_decrease"] = min_impurity_decrease
        self.use_special_dims = tree_kwargs["use_special_dims"] = use_special_dims
        self.n_features = tree_kwargs["n_features"] = n_features

        # I use "batched" to mean "all at once" and "batch_size" to mean "in chunks"
        self.batch_size = batch_size
        if batch_size is not None:
            self.batched = False
        else:
            self.batched = True

        # Random forest hyperparameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = [ProductSpaceDT(**tree_kwargs) for _ in range(n_estimators)]

    def _generate_subsample(self, n_rows, n_cols, n_trees):
        # Get number of dimensions in our subsample
        if isinstance(self.max_features, int) and self.max_features <= n_cols:
            n_cols_sample = n_cols
        elif self.max_features == "sqrt":
            n_cols_sample = torch.ceil(torch.tensor(n_cols**0.5)).int()
        elif self.max_features == "log2":
            n_cols_sample = torch.ceil(torch.log2(torch.tensor(n_cols))).int()
        elif self.max_features == "none" or self.max_features is None:
            n_cols_sample = n_cols
        else:
            raise ValueError(f"Unknown max_features parameter: {self.max_features}")

        # Subsample - returns indices
        idx_sample = torch.randint(0, n_rows, (n_trees, n_rows))
        idx_dim = torch.stack([torch.randperm(n_cols)[:n_cols_sample] for _ in range(n_trees)])

        return idx_sample, idx_dim

    @torch.no_grad()
    def fit(self, X: TT["batch ambient_dim"], y: TT["batch"]):
        """Preprocess and fit an ensemble of trees on subsampled data"""
        # Pre-preprocessing step: aggregate special dimensions
        if self.use_special_dims:
            X, self.pm = self.trees[0]._aggregate_special_dims(X)
            for tree in self.trees:
                tree.pm = self.pm

        # Can use any tree to preprocess X and y
        angles, labels, classes, comparisons = self.trees[0]._preprocess(X=X, y=y)
        self.classes_ = classes

        # Also update angle2man and special_first
        for tree in self.trees:
            tree.angle2man = self.trees[0].angle2man
            tree.special_first = self.trees[0].special_first

        # Use seed here
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Subsample - just the indices
        n, d = angles.shape
        idx_sample_all, idx_dim_all = self._generate_subsample(n_rows=n, n_cols=d, n_trees=self.n_estimators)

        # Fit trees
        for tree, idx_sample, idx_dim in zip(self.trees, idx_sample_all, idx_dim_all):
            tree.permutations = idx_dim
            tree.classes_ = classes
            if self.batched:
                comparisons_subsample = comparisons[idx_sample][:, idx_dim][:, :, idx_sample]
            else:
                comparisons_subsample = comparisons
            tree.tree = tree._fit_node(
                angles=angles[idx_sample][:, idx_dim],
                labels=labels[idx_sample],
                # comparisons=comparisons[idx_sample][:, idx_dim][:, :, idx_sample],
                comparisons=comparisons_subsample,
                depth=self.max_depth,
            )
        return self

    @torch.no_grad()
    def predict_proba(self, X: TT["batch intrinsic_dim"]) -> TT["batch n_classes"]:
        """Predict class probabilities for samples in X"""
        return torch.stack([tree.predict_proba(X) for tree in self.trees]).mean(dim=0)

    def predict(self, X: TT["batch intrinsic_dim"]) -> TT["batch"]:
        """Predict class labels for samples in X"""
        # return self.classes_[self.predict_proba(X).argmax(dim=1)]
        if self.task == "classification":
            return self.classes_[self.predict_proba(X).argmax(dim=1)]
        else:
            return self.predict_proba(X).flatten()  # Simplified version

    def score(self, X: TT["batch intrinsic_dim"], y: TT["batch"]) -> TT["batch"]:
        """Return the mean accuracy on the given test data and labels"""
        # return self.predict(X) == y
        if self.task == "classification":
            return self.predict(X) == y
        else:
            return ((self.predict(X) == y) ** 2).float().mean()
