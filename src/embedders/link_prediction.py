import torch
from torchtyping import TensorType as TT
from typing import Tuple
from .manifolds import ProductManifold


def make_link_prediction_dataset(
    X_embed: TT["batch", "n_dim"], pm: ProductManifold, adj: TT["batch", "batch"], add_dists: bool = True
) -> Tuple[TT["batch**2", "2*n_dim"], TT["batch**2"], ProductManifold]:
    # Stack embeddings
    # emb = []
    # for X_i in X_embed:
    #     for X_j in X_embed:
    #         joint_embed = torch.cat([X_embed[i], X_embed[j]])
    #         emb.append(joint_embed)
    # embed = [[torch.cat([X_i, X_j]) for X_j in X_embed] for X_i in X_embed]

    # X = torch.stack(emb)
    X = torch.stack([torch.cat([X_i, X_j]) for X_i in X_embed for X_j in X_embed])

    # Add distances
    if add_dists:
        dists = pm.pdist(X_embed)
        X = torch.cat([X, dists.flatten().unsqueeze(1)], dim=1)

    # y = torch.tensor(adj.flatten())
    y = adj.flatten()

    # Make a new signature
    new_sig = pm.signature + pm.signature
    if add_dists:
        new_sig.append((0, 1))
    new_pm = ProductManifold(signature=new_sig)

    return X, y, new_pm
