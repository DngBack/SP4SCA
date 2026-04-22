import numpy as np
import pandas as pd


def predict_all_cells(adata, latent, clf, label_key: str, split_key: str) -> pd.DataFrame:
    probs = clf.predict_proba(latent)
    classes = clf.classes_
    pred = classes[np.argmax(probs, axis=1)]
    logits = np.log(probs + 1e-12)

    out = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "split": adata.obs[split_key].astype(str).values,
            "y_true": adata.obs[label_key].astype(str).values,
            "y_pred": pred.astype(str),
        }
    )
    return out, probs, logits, classes
