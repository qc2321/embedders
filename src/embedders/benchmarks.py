from torchtyping import TensorType as TT
from typing import List, Literal, Dict

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC, SVR

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
    score: Literal["accuracy", "f1-micro", "mse", "percent_rmse"] = "f1-micro",
    classifiers: List[str] = [
        "sklearn_dt",
        "sklearn_rf",
        "product_dt",
        "product_rf",
        "tangent_dt",
        "tangent_rf",
        "knn",
        "perceptron",
        "svm",
    ],
    max_depth: int = 3,
    n_estimators: int = 12,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    task: Literal["classification", "regression"] = "classification",
    **kwargs,
) -> Dict[str, float]:
    # Coerce to tensor as needed
    if not torch.is_tensor(X):
        X = torch.tensor(X).to(device)
    if not torch.is_tensor(y):
        y = torch.tensor(y).to(device)

    X = X.to(device)
    y = y.to(device)

    # Input validation on (task, score) pairing
    if task == "classification":
        assert score in ["accuracy", "f1-micro"]
    elif task == "regression":
        assert score in ["mse", "percent_rmse"]

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
    def _score(_X, _y, model, torch=False):
        y_pred = model.predict(_X)
        if torch:
            y_pred = y_pred.detach().cpu().numpy()
        if score == "accuracy":
            return accuracy_score(_y, y_pred)
        elif score == "f1-micro":
            return f1_score(_y, y_pred, average="micro")
        elif score == "mse":
            return ((y_pred - _y) ** 2).mean()
        elif score == "percent_rmse":
            return (np.sqrt((y_pred - _y) ** 2) / np.abs(_y)).mean()
        else:
            raise ValueError(f"Unknown score: {score}")

    # Aggregate arguments
    tree_kwargs = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "min_samples_split": min_samples_split}
    rf_kwargs = {"n_estimators": n_estimators} 

    # Define your models
    if task == "classification":
        dt_class = DecisionTreeClassifier
        rf_class = RandomForestClassifier
        knn_class = KNeighborsClassifier
        svm_class = SVC
    elif task == "regression":
        dt_class = DecisionTreeRegressor
        rf_class = RandomForestRegressor
        knn_class = KNeighborsRegressor
        svm_class = SVR

    # Evaluate sklearn
    accs = {}
    if "sklearn_dt" in classifiers:
        dt = dt_class(**tree_kwargs)
        dt.fit(X_train_np, y_train_np)
        accs["sklearn_dt"] = _score(X_test_np, y_test_np, dt, torch=False)

    if "sklearn_rf" in classifiers:
        rf = rf_class(**tree_kwargs, **rf_kwargs)
        rf.fit(X_train_np, y_train_np)
        accs["sklearn_rf"] = _score(X_test_np, y_test_np, rf, torch=False)

    if "product_dt" in classifiers:
        psdt = ProductSpaceDT(pm=pm, task=task, **tree_kwargs)
        psdt.fit(X_train, y_train)
        accs["product_dt"] = _score(X_test, y_test_np, psdt, torch=True)

    if "product_rf" in classifiers:
        psrf = ProductSpaceRF(pm=pm, task=task, **tree_kwargs, **rf_kwargs)
        psrf.fit(X_train, y_train)
        accs["product_rf"] = _score(X_test, y_test_np, psrf, torch=True)

    if "tangent_dt" in classifiers:
        tdt = dt_class(**tree_kwargs)
        tdt.fit(X_train_tangent_np, y_train_np)
        accs["tangent_dt"] = _score(X_test_tangent_np, y_test_np, tdt, torch=False)

    if "tangent_rf" in classifiers:
        trf = rf_class(**tree_kwargs, **rf_kwargs)
        trf.fit(X_train_tangent_np, y_train_np)
        accs["tangent_rf"] = _score(X_test_tangent_np, y_test_np, trf, torch=False)

    if "restricted_dt" in classifiers:
        dt = dt_class(**tree_kwargs)
        dt.fit(X_train_restricted, y_train_np)
        accs["restricted_dt"] = _score(X_test_restricted, y_test_np, dt, torch=False)

    if "restricted_rf" in classifiers:
        rf = rf_class(**tree_kwargs, **rf_kwargs)
        rf.fit(X_train_restricted, y_train_np)
        accs["restricted_rf"] = _score(X_test_restricted, y_test_np, rf, torch=False)

    if "knn" in classifiers:
        # Get dists - max imputation is a workaround for some nan values we occasionally get
        train_dists = pm.pdist(X_train)
        train_dists = torch.nan_to_num(train_dists, nan=train_dists[~train_dists.isnan()].max().item())
        train_test_dists = pm.dist(X_test, X_train)
        train_test_dists = torch.nan_to_num(
            train_test_dists, nan=train_test_dists[~train_test_dists.isnan()].max().item()
        )

        # Convert to numpy
        train_dists = train_dists.detach().cpu().numpy()
        train_test_dists = train_test_dists.detach().cpu().numpy()

        # Train classifier on distances
        knn = knn_class(metric="precomputed")
        knn.fit(train_dists, y_train_np)
        accs["knn"] = _score(train_test_dists, y_test_np, knn, torch=False)
    
    if "perceptron" in classifiers:
        # Perceptron doesn't work with regression
        if task == "regression":
            accs["perceptron"] = None
        else:
            ptron = Perceptron(fit_intercept=False) # Must be false for our data
            ptron.fit(X_train_np, y_train_np)
            accs["perceptron"] = _score(X_test_np, y_test_np, ptron, torch=False)
    
    if "svm" in classifiers:
        # Get inner products for precomputed kernel matrix
        train_ips = pm.manifold.component_inner(X_train[:, None], X_train[None, :]).sum(dim=-1)
        train_test_ips = pm.manifold.component_inner(X_test[:, None], X_train[None, :]).sum(dim=-1)

        # Convert to numpy
        train_ips = train_ips.detach().cpu().numpy()
        train_test_ips = train_test_ips.detach().cpu().numpy()

        # Train SVM on precomputed inner products
        svm = svm_class(kernel="precomputed", max_iter=10_000)
        # Need max_iter because it can hang. It can be large, since this doesn't happen often.
        svm.fit(train_ips, y_train_np)
        accs["svm"] = _score(train_test_ips, y_test_np, svm, torch=False)
    
    if "distance_dt" in classifiers:
        raise NotImplementedError("Distance decision tree not implemented yet")
    
    if "distance_rf" in classifiers:
        raise NotImplementedError("Distance random forest not implemented yet")

    return accs