from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
