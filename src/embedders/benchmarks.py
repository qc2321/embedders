from torchtyping import TensorType as TT
from typing import List, Literal, Dict, Optional

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR

# Tabbaghi imports
# from hyperdt.product_space_svm import mix_curv_svm
# from hyperdt.product_space_perceptron import mix_curv_perceptron

import time

from .manifolds import ProductManifold

from .predictors.tree_new import ProductSpaceDT, ProductSpaceRF
from .predictors.perceptron import ProductSpacePerceptron
from .predictors.svm import ProductSpaceSVM
from .predictors.mlp import MLP
from .predictors.gnn import GNN, get_nonzero


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
    # score: Literal["accuracy", "f1-micro", "mse", "percent_rmse"] = "f1-micro",
    score: List[Literal["accuracy", "f1-micro", "mse", "percent_rmse"]] = ["accuracy", "f1-micro"],
    models: List[str] = [
        "sklearn_dt",
        "sklearn_rf",
        "product_dt",
        "product_rf",
        "tangent_dt",
        "tangent_rf",
        "knn",
        "ps_perceptron",
        # "ps_svm"
        # "svm",
        "tangent_mlp",
        "ambient_mlp",
        "tangent_gnn",
        "ambient_gnn",
    ],
    max_depth: int = 3,
    n_estimators: int = 12,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    task: Literal["classification", "regression"] = "classification",
    seed: Optional[int] = None,
    use_special_dims: bool = False,
    n_features: Literal["d", "d_choose_2"] = "d",
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    batch_size=None,
    adj=None,
) -> Dict[str, float]:
    # Input validation on (task, score) pairing
    if task == "classification":
        # assert score in ["accuracy", "f1-micro"]
        assert all(s in ["accuracy", "f1-micro", "time"] for s in score)
    elif task == "regression":
        # assert score in ["mse", "rmse", "percent_rmse"]
        assert all(s in ["mse", "rmse", "percent_rmse", "time"] for s in score)

    # # Fix nan and inf values
    # X = torch.nan_to_num(X, nan=0, posinf=3e38, neginf=-3e38)

    # Make sure classification labels are formatted correctly
    if task == "classification":
        y = torch.unique(y, return_inverse=True)[1]

    # Make sure we're on the right device
    pm = pm.to(device)

    # Get pdists
    pdists = pm.pdist(X).detach()

    # Get tangent plane
    X_tangent = pm.logmap(X).detach()

    # Split data
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        # Coerce to tensor as needed
        if not torch.is_tensor(X_train):
            X_train = torch.tensor(X_train)
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test)
        if not torch.is_tensor(y_train):
            y_train = torch.tensor(y_train)
        if not torch.is_tensor(y_test):
            y_test = torch.tensor(y_test)

        # Move to device
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)

    else:
        # Coerce to tensor as needed
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        if not torch.is_tensor(y):
            y = torch.tensor(y)

        X = X.to(device)
        y = y.to(device)

        X_train, X_test, y_train, y_test, X_train_tangent, X_test_tangent, train_idx, test_idx = train_test_split(
            X, y, X_tangent, np.arange(len(X)), test_size=0.2
        )

    # Get numpy versions
    X_train_np, X_test_np = X_train.detach().cpu().numpy(), X_test.detach().cpu().numpy()
    y_train_np, y_test_np = y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()

    # Tangents
    # X_train_tangent = pm.manifold.logmap(pm.mu0, X_train).detach()
    # X_test_tangent = pm.manifold.logmap(pm.mu0, X_test).detach()
    X_train_tangent_np, X_test_tangent_np = X_train_tangent.cpu().numpy(), X_test_tangent.cpu().numpy()

    # Restricted X:
    X_train_restricted = _restrict_X(X_train_np, pm.signature)
    X_test_restricted = _restrict_X(X_test_np, pm.signature)

    # TODO: Implement other splits
    # TODO: Implement other scoring metrics
    def _score(_X, _y, model, y_pred_override=None, torch=False):
        # Override y_pred
        if y_pred_override is not None:
            y_pred = y_pred_override
        else:
            y_pred = model.predict(_X)

        # Convert to numpy
        if torch:
            y_pred = y_pred.detach().cpu().numpy()

        # Score handling
        out = {}
        for s in score:
            if s == "accuracy":
                out[s] = accuracy_score(_y, y_pred)
            elif s == "f1-micro":
                out[s] = f1_score(_y, y_pred, average="micro")
            elif s == "mse":
                out[s] = mean_squared_error(_y, y_pred)
            elif s == "rmse":
                out[s] = root_mean_squared_error(_y, y_pred)
            elif s == "percent_rmse":
                out[s] = (root_mean_squared_error(_y, y_pred, multioutput="raw_values") / np.abs(_y)).mean()
            else:
                raise ValueError(f"Unknown score: {s}")
        return out

    # Aggregate arguments
    tree_kwargs = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "min_samples_split": min_samples_split}
    rf_kwargs = {"n_estimators": n_estimators, "n_jobs": -1, "random_state": seed}

    # Define your models
    if task == "classification":
        dt_class = DecisionTreeClassifier
        rf_class = RandomForestClassifier
        knn_class = KNeighborsClassifier
        svm_class = SVC

        perceptron_class = SGDClassifier
    elif task == "regression":
        dt_class = DecisionTreeRegressor
        rf_class = RandomForestRegressor
        knn_class = KNeighborsRegressor
        svm_class = SVR
        perceptron_class = SGDRegressor

    # Evaluate sklearn
    accs = {}
    if "sklearn_dt" in models:
        dt = dt_class(**tree_kwargs)
        t1 = time.time()
        dt.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["sklearn_dt"] = _score(X_test_np, y_test_np, dt, torch=False)
        accs["sklearn_dt"]["time"] = t2 - t1

    if "sklearn_rf" in models:
        rf = rf_class(**tree_kwargs, **rf_kwargs)
        t1 = time.time()
        rf.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["sklearn_rf"] = _score(X_test_np, y_test_np, rf, torch=False)
        accs["sklearn_rf"]["time"] = t2 - t1

    if "product_dt" in models:
        psdt = ProductSpaceDT(
            pm=pm,
            task=task,
            **tree_kwargs,
            use_special_dims=use_special_dims,
            n_features=n_features,
            batch_size=batch_size,
        )
        t1 = time.time()
        psdt.fit(X_train, y_train)
        t2 = time.time()
        accs["product_dt"] = _score(X_test, y_test_np, psdt, torch=True)
        accs["product_dt"]["time"] = t2 - t1

    if "product_rf" in models:
        psrf = ProductSpaceRF(
            pm=pm,
            task=task,
            **tree_kwargs,
            **rf_kwargs,
            use_special_dims=use_special_dims,
            n_features=n_features,
            batch_size=batch_size,
        )
        t1 = time.time()
        psrf.fit(X_train, y_train)
        t2 = time.time()
        accs["product_rf"] = _score(X_test, y_test_np, psrf, torch=True)
        accs["product_rf"]["time"] = t2 - t1

    if "tangent_dt" in models:
        tdt = dt_class(**tree_kwargs)
        t1 = time.time()
        tdt.fit(X_train_tangent_np, y_train_np)
        t2 = time.time()
        accs["tangent_dt"] = _score(X_test_tangent_np, y_test_np, tdt, torch=False)
        accs["tangent_dt"]["time"] = t2 - t1

    if "tangent_rf" in models:
        trf = rf_class(**tree_kwargs, **rf_kwargs)
        t1 = time.time()
        trf.fit(X_train_tangent_np, y_train_np)
        t2 = time.time()
        accs["tangent_rf"] = _score(X_test_tangent_np, y_test_np, trf, torch=False)
        accs["tangent_rf"]["time"] = t2 - t1

    if "restricted_dt" in models:
        dt = dt_class(**tree_kwargs)
        t1 = time.time()
        dt.fit(X_train_restricted, y_train_np)
        t2 = time.time()
        accs["restricted_dt"] = _score(X_test_restricted, y_test_np, dt, torch=False)
        accs["restricted_dt"]["time"] = t2 - t1

    if "restricted_rf" in models:
        rf = rf_class(**tree_kwargs, **rf_kwargs)
        t1 = time.time()
        rf.fit(X_train_restricted, y_train_np)
        t2 = time.time()
        accs["restricted_rf"] = _score(X_test_restricted, y_test_np, rf, torch=False)
        accs["restricted_rf"]["time"] = t2 - t1

    if "knn" in models:
        # Get dists - max imputation is a workaround for some nan values we occasionally get
        t1 = time.time()
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
        t2 = time.time()
        knn.fit(train_dists, y_train_np)
        t3 = time.time()
        accs["knn"] = _score(train_test_dists, y_test_np, knn, torch=False)
        accs["knn"]["time"] = t3 - t1

    if "perceptron" in models:
        loss = "perceptron" if task == "classification" else "squared_error"
        ptron = perceptron_class(
            loss=loss, learning_rate="constant", fit_intercept=False, eta0=1.0, max_iter=10_000
        )  # fit_intercept must be false for ambient coordinates
        t1 = time.time()
        ptron.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["perceptron"] = _score(X_test_np, y_test_np, ptron, torch=False)
        accs["perceptron"]["time"] = t2 - t1

    if "ps_perceptron" in models:
        if task == "classification":
            ps_per = ProductSpacePerceptron(pm=pm)
            t1 = time.time()
            ps_per.fit(X_train, y_train)
            t2 = time.time()
            accs["ps_perceptron"] = _score(X_test, y_test_np, ps_per, torch=True)
            accs["ps_perceptron"]["time"] = t2 - t1
        # TODO: regression

    if "svm" in models:
        # Get inner products for precomputed kernel matrix
        t1 = time.time()
        train_ips = pm.manifold.component_inner(X_train[:, None], X_train[None, :]).sum(dim=-1)
        train_test_ips = pm.manifold.component_inner(X_test[:, None], X_train[None, :]).sum(dim=-1)

        # Convert to numpy
        train_ips = train_ips.detach().cpu().numpy()
        train_test_ips = train_test_ips.detach().cpu().numpy()

        # Train SVM on precomputed inner products
        svm = svm_class(kernel="precomputed", max_iter=10_000)
        # Need max_iter because it can hang. It can be large, since this doesn't happen often.
        t2 = time.time()
        svm.fit(train_ips, y_train_np)
        t3 = time.time()
        accs["svm"] = _score(train_test_ips, y_test_np, svm, torch=False)
        accs["svm"]["time"] = t3 - t1

    if "ps_svm" in models:
        ps_svm = ProductSpaceSVM(pm=pm, task=task, h_constraints=False, e_constraints=False)
        t1 = time.time()
        ps_svm.fit(X_train, y_train)
        t2 = time.time()
        accs["ps_svm"] = _score(X_test, y_test_np, ps_svm, torch=False)
        accs["ps_svm"]["time"] = t2 - t1

    if "distance_dt" in models:
        raise NotImplementedError("Distance decision tree not implemented yet")

    if "distance_rf" in models:
        raise NotImplementedError("Distance random forest not implemented yet")

    # Need to get shapes for neural networks
    in_dim = X_train.shape[1]
    out_dim = 1 if task == "regression" else len(torch.unique(y))

    if "tangent_mlp" in models:
        tangent_mlp = MLP(pm=pm, tangent=True, task=task, input_dim=in_dim, hidden_dims=[in_dim], output_dim=out_dim)
        tangent_mlp = tangent_mlp.to(device)
        t1 = time.time()
        tangent_mlp.fit(X_train_tangent, y_train)
        t2 = time.time()
        accs["tangent_mlp"] = _score(X_test_tangent, y_test_np, tangent_mlp, torch=True)
        accs["tangent_mlp"]["time"] = t2 - t1

    if "ambient_mlp" in models:
        ambient_mlp = MLP(pm=pm, tangent=False, task=task, input_dim=in_dim, hidden_dims=[in_dim], output_dim=out_dim)
        ambient_mlp = ambient_mlp.to(device)
        t1 = time.time()
        ambient_mlp.fit(X_train, y_train)
        t2 = time.time()
        accs["ambient_mlp"] = _score(X_test, y_test_np, ambient_mlp, torch=True)
        accs["ambient_mlp"]["time"] = t2 - t1

    if adj is not None and "tangent_gnn" in models:
        tangent_gnn = GNN(pm=pm, tangent=True, input_dim=in_dim, hidden_dims=[in_dim], output_dim=out_dim, task=task, edge_func=get_nonzero)
        tangent_gnn = tangent_gnn.to(device)
        t1 = time.time()
        tangent_gnn.fit(X_tangent, y, adj=adj, train_idx=train_idx)
        y_pred = tangent_gnn.predict(X_tangent, adj=adj, test_idx=test_idx)
        t2 = time.time()
        accs["tangent_gnn_adj"] = _score(None, y_test_np, tangent_gnn, y_pred_override=y_pred, torch=True)
        accs["tangent_gnn_adj"]["time"] = t2 - t1

    if adj is not None and "ambient_gnn" in models:
        ambient_gnn = GNN(pm=pm, tangent=False, input_dim=in_dim, hidden_dims=[in_dim], output_dim=out_dim, task=task, edge_func=get_nonzero)
        ambient_gnn = ambient_gnn.to(device)
        t1 = time.time()
        ambient_gnn.fit(X, y, adj=adj, train_idx=train_idx)
        y_pred = ambient_gnn.predict(X, adj=adj, test_idx=test_idx)
        t2 = time.time()
        accs["ambient_gnn_adj"] = _score(None, y_test_np, ambient_gnn, y_pred_override=y_pred, torch=True)
        accs["ambient_gnn_adj"]["time"] = t2 - t1

    if "tangent_gnn" in models:
        tangent_gnn = GNN(pm=pm, tangent=True, input_dim=in_dim, hidden_dims=[in_dim], output_dim=out_dim, task=task)
        tangent_gnn = tangent_gnn.to(device)
        t1 = time.time()
        tangent_gnn.fit(X_tangent, y, adj=pdists, train_idx=train_idx)
        y_pred = tangent_gnn.predict(X_tangent, adj=pdists, test_idx=test_idx)
        t2 = time.time()
        accs["tangent_gnn"] = _score(None, y_test_np, tangent_gnn, y_pred_override=y_pred, torch=True)
        accs["tangent_gnn"]["time"] = t2 - t1

    if "ambient_gnn" in models:
        ambient_gnn = GNN(pm=pm, tangent=False, input_dim=in_dim, hidden_dims=[in_dim], output_dim=out_dim, task=task)
        ambient_gnn = ambient_gnn.to(device)
        t1 = time.time()
        ambient_gnn.fit(X, y, adj=pdists, train_idx=train_idx)
        y_pred = tangent_gnn.predict(X, adj=pdists, test_idx=test_idx)
        t2 = time.time()
        accs["ambient_gnn"] = _score(None, y_test_np, ambient_gnn, y_pred_override=y_pred, torch=True)
        accs["ambient_gnn"]["time"] = t2 - t1

    return accs
