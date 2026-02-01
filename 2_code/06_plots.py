import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "3_outputs" / "tables"
FIG_TS_DIR = PROJECT_ROOT / "3_outputs" / "figures" / "time_series"
FIG_TS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_XLSX = TABLES_DIR / "quarterly_statistics_4month.xlsx"

COMPANY_ORDER = ["HCL", "Infosys", "TCS", "Wipro"]

def _quarter_sort_key(q: str):
    # q format: 'YYYY_Q1', 'YYYY_Q2', 'YYYY_Q3'
    year_str, qpart = q.split("_")
    qnum = int(qpart.replace("Q", ""))
    return (int(year_str), qnum)

def _pretty_quarter_labels(q_list):
    # "2023_Q1" -> "2023 Q1"
    return [q.replace("_", " ") for q in q_list]

def plot_timeseries(pivot_df: pd.DataFrame, title: str, y_label: str, outpath: Path):
    # General styling (good-looking but still simple)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10
    })

    quarters = pivot_df.index.tolist()
    x_labels = _pretty_quarter_labels(quarters)

    fig, ax = plt.subplots(figsize=(11, 6))

    for col in pivot_df.columns:
        ax.plot(
            x_labels,
            pivot_df[col].values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=col
        )

    ax.set_title(title)
    ax.set_xlabel("Quarter (4-month)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    # Put legend outside right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)

def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Missing {INPUT_XLSX}. Run 04_quarterly_stats.py first.")

    df = pd.read_excel(INPUT_XLSX)
    df = df[["Quarter", "Company", "Mean", "SD"]].copy()

    quarters = sorted(df["Quarter"].unique().tolist(), key=_quarter_sort_key)

    mean_pivot = df.pivot(index="Quarter", columns="Company", values="Mean").loc[quarters]
    sd_pivot = df.pivot(index="Quarter", columns="Company", values="SD").loc[quarters]

    # enforce company order
    mean_cols = [c for c in COMPANY_ORDER if c in mean_pivot.columns] + [c for c in mean_pivot.columns if c not in COMPANY_ORDER]
    sd_cols = [c for c in COMPANY_ORDER if c in sd_pivot.columns] + [c for c in sd_pivot.columns if c not in COMPANY_ORDER]
    mean_pivot = mean_pivot[mean_cols]
    sd_pivot = sd_pivot[sd_cols]

    # Save pivots (still useful for report writing)
    mean_pivot.to_excel(TABLES_DIR / "mean_by_quarter_pivot.xlsx")
    sd_pivot.to_excel(TABLES_DIR / "sd_by_quarter_pivot.xlsx")

    out1 = FIG_TS_DIR / "avg_price_by_quarter.png"
    out2 = FIG_TS_DIR / "sd_by_quarter.png"

    plot_timeseries(
        mean_pivot,
        title="Average Daily Close Price by Quarter",
        y_label="Average Close Price (₹)",
        outpath=out1
    )

    plot_timeseries(
        sd_pivot,
        title="Standard Deviation of Daily Close Price by Quarter",
        y_label="SD of Close Price (₹)",
        outpath=out2
    )

    print("Saved improved plots:")
    print("-", out1)
    print("-", out2)

if __name__ == "__main__":
    main()