import numpy as np
import pandas as pd

from src.models.classifier_mlp import evaluate_classifier, fit_mlp_classifier


def train_classifier_on_split(
    adata,
    latent,
    label_key: str,
    split_key: str,
    train_split_name: str = "train_ref",
    hidden_dim: int = 128,
    dropout: float = 0.1,
    max_iter: int = 200,
    seed: int = 42,
):
    split_vals = adata.obs[split_key].astype(str).values
    train_mask = split_vals == train_split_name

    if train_mask.sum() == 0:
        raise ValueError(f"No cells found for split={train_split_name}")

    y = adata.obs[label_key].astype(str).values
    X_train = latent[train_mask]
    y_train = y[train_mask]

    clf = fit_mlp_classifier(
        X_train=X_train,
        y_train=y_train,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_iter=max_iter,
        seed=seed,
    )

    train_acc = evaluate_classifier(clf, X_train, y_train)

    meta = pd.DataFrame(
        [
            {
                "train_split": train_split_name,
                "n_train": int(train_mask.sum()),
                "train_accuracy": train_acc,
                "n_classes": int(np.unique(y_train).size),
            }
        ]
    )
    return clf, meta
