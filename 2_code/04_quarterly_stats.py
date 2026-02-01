import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "1_data" / "processed"
TABLES_DIR = PROJECT_ROOT / "3_outputs" / "tables"
FIG_Q_DIR = PROJECT_ROOT / "3_outputs" / "figures" / "quarterly"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_Q_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PROCESSED_DIR / "it_sector_balanced_close_2023_2024.csv"

START_DATE = "2023-01-03"
END_DATE_INCLUSIVE = "2024-06-28"


def safe_mode(s: pd.Series):
    m = s.mode(dropna=True)
    if len(m) == 0:
        return np.nan
    return float(m.min())


def slope_rupees_per_day(dates: pd.DatetimeIndex, prices: pd.Series) -> float:
    t = (dates - dates[0]).days.astype(float)
    y = prices.astype(float).values
    b, a = np.polyfit(t, y, 1)
    return float(b)


def make_4month_quarters(index: pd.DatetimeIndex) -> pd.Series:
    """
    Assign each date to a 4-month quarter label:
    Q1: Jan-Apr, Q2: May-Aug, Q3: Sep-Dec, etc., by year.
    Returns a Series of labels aligned to index.
    """
    m = index.month
    qnum = ((m - 1) // 4) + 1  # 1..3
    # label like 2023_Q1 (Jan-Apr), 2023_Q2 (May-Aug), 2023_Q3 (Sep-Dec)
    labels = [f"{y}_Q{qn}" for y, qn in zip(index.year, qnum)]
    return pd.Series(labels, index=index, name="Quarter")


def plot_quarter_scatter(df_q: pd.DataFrame, quarter_label: str):
    for company in df_q.columns:
        plt.figure()
        plt.scatter(df_q.index, df_q[company], s=6)
        plt.title(f"{company} — Daily Close Price ({quarter_label})")
        plt.xlabel("Date")
        plt.ylabel("Close Price (₹)")
        plt.tight_layout()
        out = FIG_Q_DIR / f"{company}_{quarter_label}_scatter.png"
        plt.savefig(out, dpi=200)
        plt.close()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Run 02 first.")

    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE_INCLUSIVE)]

    qlabels = make_4month_quarters(df.index)
    df = df.copy()
    df["Quarter"] = qlabels.values

    all_rows = []

    for q in df["Quarter"].unique():
        df_q = df[df["Quarter"] == q].drop(columns=["Quarter"])
        # scatterplots per quarter
        plot_quarter_scatter(df_q, q)

        for company in df_q.columns:
            s = df_q[company].dropna()
            q1 = float(s.quantile(0.25))
            q2 = float(s.quantile(0.50))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1

            mean = float(s.mean())
            sd = float(s.std(ddof=1))
            cv = float(sd / mean) if mean != 0 else np.nan
            sl = slope_rupees_per_day(df_q.index, s)

            all_rows.append({
                "Quarter": q,
                "Company": company,
                "N (days)": int(s.shape[0]),
                "Mean": mean,
                "SD": sd,
                "Median": float(s.median()),
                "Mode": safe_mode(s),
                "Min": float(s.min()),
                "Max": float(s.max()),
                "Range": float(s.max() - s.min()),
                "Q1": q1,
                "Q2 (Median)": q2,
                "Q3": q3,
                "IQR": float(iqr),
                "Coeff. of Variation (SD/Mean)": cv,
                "Slope (₹/day)": float(sl),
            })

    out = pd.DataFrame(all_rows).sort_values(["Quarter", "Company"])

    out_csv = TABLES_DIR / "quarterly_statistics_4month.csv"
    out_xlsx = TABLES_DIR / "quarterly_statistics_4month.xlsx"
    out.to_csv(out_csv, index=False)
    out.to_excel(out_xlsx, index=False)

    print("Saved quarterly stats:")
    print("-", out_csv)
    print("-", out_xlsx)
    print("Saved quarterly scatterplots to:", FIG_Q_DIR)


if __name__ == "__main__":
    main()