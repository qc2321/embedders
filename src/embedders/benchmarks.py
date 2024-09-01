from torchtyping import TensorType
from typing import List, Literal, Dict

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# from hyperdt.product_space_DT import ProductSpaceDT
# from hyperdt.forest import ProductSpaceRF

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# from .tree import TorchProductSpaceDT, TorchProductSpaceRF
from .tree_new import ProductSpaceDT, ProductSpaceRF
from .manifolds import ProductManifold

def _fix_X(X, signature):
    # Use numpy since it's going to the legacy ProductDT anyway
    X_out = []
    i = 0
    for curvature, dimension in signature:
        if curvature == 0:
            X_out.append(np.ones((X.shape[0],1)))
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
    X: TensorType["batch", "dim"],
    y: TensorType["batch"],
    pm: ProductManifold,
    split: Literal["train_test", "cross_val"] = "train_test",
    device: Literal["cpu", "cuda"] = "cpu",
    score: Literal["accuracy", "f1-micro"] = "f1-micro",
    classifiers: List[str] = [
        "sklearn_dt",
        "sklearn_rf",
        "product_dt",
        "product_rf",
        "tangent_dt",
        "tangent_rf",
        # "product_dt_legacy",
        # "product_rf_legacy",
        "restricted_dt",
        "restricted_rf",
    ],
    max_depth: int = 3,
    n_estimators: int = 12,
    **kwargs
) -> Dict[str, float]:
    # Coerce to tensor as needed
    if not torch.is_tensor(X):
        X = torch.tensor(X).to(device)
    if not torch.is_tensor(y):
        y = torch.tensor(y).to(device)

    X = X.to(device)
    y = y.to(device)

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

    # Fixed X
    X_train_fixed = _fix_X(X_train_np, pm.signature)
    X_test_fixed = _fix_X(X_test_np, pm.signature)

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

    # Evaluate sklearn
    accs = {}
    if "sklearn_dt" in classifiers:
        dt = DecisionTreeClassifier(max_depth=max_depth)
        dt.fit(X_train_np, y_train_np)
        accs["sklearn_dt"] = _score(X_test_np, y_test_np, dt, torch=False)

    if "sklearn_rf" in classifiers:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(X_train_np, y_train_np)
        accs["sklearn_rf"] = _score(X_test_np, y_test_np, rf, torch=False)

    if "product_dt" in classifiers:
        # psdt = TorchProductSpaceDT(signature=pm.signature, max_depth=max_depth)
        psdt = ProductSpaceDT(pm=pm)
        psdt.fit(X_train, y_train)
        accs["product_dt"] = _score(X_test, y_test_np, psdt, torch=True)

    if "product_rf" in classifiers:
        # psrf = TorchProductSpaceRF(signature=pm.signature, n_estimators=n_estimators, max_depth=max_depth)
        psrf = ProductSpaceRF(pm=pm)
        psrf.fit(X_train, y_train)
        accs["product_rf"] = _score(X_test, y_test_np, psrf, torch=True)

    if "tangent_dt" in classifiers:
        tdt = DecisionTreeClassifier(max_depth=max_depth)
        tdt.fit(X_train_tangent_np, y_train_np)
        accs["tangent_dt"] = _score(X_test_tangent_np, y_test_np, tdt, torch=False)

    if "tangent_rf" in classifiers:
        trf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        trf.fit(X_train_tangent_np, y_train_np)
        accs["tangent_rf"] = _score(X_test_tangent_np, y_test_np, trf, torch=False)
    
    # if "product_dt_legacy" in classifiers:
    #     psdt = ProductSpaceDT([(x[1], x[0]) for x in pm.signature], max_depth=max_depth)
    #     psdt.fit(X_train_fixed, y_train_np)
    #     accs["product_dt_legacy"] = _score(X_test_fixed, y_test_np, psdt, torch=False)
    
    # if "product_rf_legacy" in classifiers:
    #     psrf = ProductSpaceRF([(x[1], x[0]) for x in pm.signature], n_estimators=n_estimators, max_depth=max_depth)
    #     psrf.fit(X_train_fixed, y_train_np)
    #     accs["product_rf_legacy"] = _score(X_test_fixed, y_test_np, psrf, torch=False)
    
    if "restricted_dt" in classifiers:
        dt = DecisionTreeClassifier(max_depth=max_depth)
        dt.fit(X_train_restricted, y_train_np)
        accs["restricted_dt"] = _score(X_test_restricted, y_test_np, dt, torch=False)
    
    if "restricted_rf" in classifiers:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(X_train_restricted, y_train_np)
        accs["restricted_rf"] = _score(X_test_restricted, y_test_np, rf, torch=False)

    return accs
