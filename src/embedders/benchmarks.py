from torchtyping import TensorType as TT
from typing import List, Literal, Dict

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from .tree_new import ProductSpaceDT, ProductSpaceRF
from .manifolds import ProductManifold


def _fix_X(X, signature):
    # Use numpy since it's going to the legacy ProductDT anyway
    X_out = []
    i = 0
    for curvature, dimension in signature:
        if curvature == 0:
            X_out.append(np.ones((X.shape[0], 1)))
            X_out.append(X[:, i : i + dimension])
        else:
            X_out.append(X[:, i : i + dimension + 1])
        i += dimension
    return np.hstack(X_out)


def _restrict_X(X, signature):
    X_out = []
    i = 0
    for curvature, dimension in signature:
        if curvature == 0:
            X_out.append(X[:, i : i + dimension])
        else:
            X_out.append(X[:, i + 1 : i + dimension])
        i += dimension
    return np.hstack(X_out)


def benchmark(
    X: TT["batch", "dim"],
    y: TT["batch"],
    pm: ProductManifold,
    split: Literal["train_test", "cross_val"] = "train_test",
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    score: Literal["accuracy", "f1-micro"] = "f1-micro",
    classifiers: List[str] = [
        "sklearn_dt",
        "sklearn_rf",
        "product_dt",
        "product_rf",
        "tangent_dt",
        "tangent_rf",
        "restricted_dt",
        "restricted_rf",
        "knn",
    ],
    max_depth: int = 3,
    n_estimators: int = 12,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    **kwargs,
) -> Dict[str, float]:
    # Coerce to tensor as needed
    if not torch.is_tensor(X):
        X = torch.tensor(X).to(device)
    if not torch.is_tensor(y):
        y = torch.tensor(y).to(device)

    X = X.to(device)
    y = y.to(device)

    # # Fix nan and inf values
    # X = torch.nan_to_num(X, nan=0, posinf=3e38, neginf=-3e38)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_np, X_test_np = X_train.detach().cpu().numpy(), X_test.detach().cpu().numpy()
    y_train_np, y_test_np = y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()

    # Tangents
    X_train_tangent = pm.manifold.logmap(pm.mu0, X_train)
    X_test_tangent = pm.manifold.logmap(pm.mu0, X_test)
    X_train_tangent_np, X_test_tangent_np = (
        X_train_tangent.detach().cpu().numpy(),
        X_test_tangent.detach().cpu().numpy(),
    )

    # Restricted X:
    X_train_restricted = _restrict_X(X_train_np, pm.signature)
    X_test_restricted = _restrict_X(X_test_np, pm.signature)

    # TODO: Implement other splits
    # TODO: Implement other scoring metrics
    def _score(_X, _y, model, torch=False, score="f1-micro"):
        y_pred = model.predict(_X)
        if torch:
            y_pred = y_pred.detach().cpu().numpy()
        if score == "accuracy":
            return accuracy_score(_y, y_pred)
        elif score == "f1-micro":
            return f1_score(_y, y_pred, average="micro")
        else:
            raise ValueError(f"Unknown score: {score}")

    # Aggregate arguments
    tree_kwargs = {"max_depth": max_depth}
    rf_kwargs = {"n_estimators": n_estimators}

    # Evaluate sklearn
    accs = {}
    if "sklearn_dt" in classifiers:
        dt = DecisionTreeClassifier(**tree_kwargs)
        dt.fit(X_train_np, y_train_np)
        accs["sklearn_dt"] = _score(X_test_np, y_test_np, dt, torch=False)

    if "sklearn_rf" in classifiers:
        rf = RandomForestClassifier(**tree_kwargs, **rf_kwargs)
        rf.fit(X_train_np, y_train_np)
        accs["sklearn_rf"] = _score(X_test_np, y_test_np, rf, torch=False)

    if "product_dt" in classifiers:
        psdt = ProductSpaceDT(pm=pm, **tree_kwargs)
        psdt.fit(X_train, y_train)
        accs["product_dt"] = _score(X_test, y_test_np, psdt, torch=True)

    if "product_rf" in classifiers:
        psrf = ProductSpaceRF(pm=pm, **rf_kwargs)
        psrf.fit(X_train, y_train)
        accs["product_rf"] = _score(X_test, y_test_np, psrf, torch=True)

    if "tangent_dt" in classifiers:
        tdt = DecisionTreeClassifier(**tree_kwargs)
        tdt.fit(X_train_tangent_np, y_train_np)
        accs["tangent_dt"] = _score(X_test_tangent_np, y_test_np, tdt, torch=False)

    if "tangent_rf" in classifiers:
        trf = RandomForestClassifier(**tree_kwargs, **rf_kwargs)
        trf.fit(X_train_tangent_np, y_train_np)
        accs["tangent_rf"] = _score(X_test_tangent_np, y_test_np, trf, torch=False)

    if "restricted_dt" in classifiers:
        dt = DecisionTreeClassifier(**tree_kwargs)
        dt.fit(X_train_restricted, y_train_np)
        accs["restricted_dt"] = _score(X_test_restricted, y_test_np, dt, torch=False)

    if "restricted_rf" in classifiers:
        rf = RandomForestClassifier(**tree_kwargs, **rf_kwargs)
        rf.fit(X_train_restricted, y_train_np)
        accs["restricted_rf"] = _score(X_test_restricted, y_test_np, rf, torch=False)

    if "knn" in classifiers:
        # Get dists - max imputation is a workaround for some nan values we occasionally get
        train_dists = pm.pdist(X_train)
        train_dists = torch.nan_to_num(train_dists, nan=train_dists[~train_dists.isnan()].max())
        train_test_dists = pm.dist(X_test, X_train)
        train_test_dists = torch.nan_to_num(train_test_dists, nan=train_test_dists[~train_test_dists.isnan()].max())

        knn = KNeighborsClassifier(metric="precomputed")
        knn.fit(train_dists, y_train_np)
        accs["knn"] = _score(train_test_dists, y_test_np, knn, torch=False)

    return accs
