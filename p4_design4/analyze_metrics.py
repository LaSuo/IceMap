"""Optional CSV error comparison plots and event export, without retraining."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seaborn", action="store_true")
    p.add_argument("--tensorboard-dir")
    args = p.parse_args()
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [
        "LER_percent",
        "LE_h_percent",
        "LE_rho_percent",
        "AE_h_percent",
        "AE_rho_percent",
    ]
    data = pd.read_csv(args.csv)
    if not {"dataset", *columns}.issubset(data.columns):
        raise ValueError("CSV must contain dataset and all five error columns.")
    if data.empty or data[columns].isna().any().any():
        raise ValueError(
            "Empty or incomplete metrics cannot be plotted as valid results."
        )
    if (data[columns] < 0).any().any():
        raise ValueError("Error metrics cannot be negative.")
    if args.seaborn:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
    axes = data.set_index("dataset")[columns].plot.barh(
        figsize=(10, max(4, 0.35 * len(data)))
    )
    axes.set_xlabel("Error (%)")
    axes.set_ylabel("Dataset")
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(args.tensorboard_dir) as writer:
            for _, row in data.iterrows():
                for column in columns:
                    writer.add_scalar(
                        f'evaluation/{row["dataset"]}/{column}', float(row[column]), 0
                    )


if __name__ == "__main__":
    main()
