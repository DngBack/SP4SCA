from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def fit_mlp_classifier(
    X_train,
    y_train,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    max_iter: int = 200,
    seed: int = 42,
):
    # sklearn MLP has no dropout; dropout argument kept for config compatibility.
    _ = dropout
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=256,
        max_iter=max_iter,
        early_stopping=False,
        n_iter_no_change=15,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X, y) -> float:
    pred = clf.predict(X)
    return float(accuracy_score(y, pred))
