import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "1_data" / "processed"
TABLES_DIR = PROJECT_ROOT / "3_outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "3_outputs" / "figures" / "full_period"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PROCESSED_DIR / "it_sector_balanced_close_2023_2024.csv"


def safe_mode(s: pd.Series):
    """
    Prices often have no unique mode. pandas mode() can return many.
    We'll return the smallest mode if multiple; NaN if none.
    """
    m = s.mode(dropna=True)
    if len(m) == 0:
        return np.nan
    return float(m.min())


def average_slope_per_day(dates: pd.DatetimeIndex, prices: pd.Series) -> float:
    """
    Fits a line: price = a + b*t where t is days since first day.
    Returns b as average slope in ₹ per day.
    """
    t = (dates - dates[0]).days.astype(float)
    y = prices.astype(float).values
    b, a = np.polyfit(t, y, 1)
    return float(b)


def compute_full_period_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for company in df.columns:
        s = df[company].dropna()

        q1 = float(s.quantile(0.25))
        q2 = float(s.quantile(0.50))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1

        mean = float(s.mean())
        sd = float(s.std(ddof=1))
        cv = float(sd / mean) if mean != 0 else np.nan

        slope = average_slope_per_day(df.index, s)

        rows.append({
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
            "Avg Slope (₹/day)": slope
        })

    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame):
    for company in df.columns:
        plt.figure()
        plt.scatter(df.index, df[company], s=6)  # keep default color
        plt.title(f"{company} — Daily Close Price (Full Period)")
        plt.xlabel("Date")
        plt.ylabel("Close Price (₹)")
        plt.tight_layout()
        out = FIG_DIR / f"{company}_scatter_full_period.png"
        plt.savefig(out, dpi=200)
        plt.close()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing processed file: {INPUT_CSV}. Run 02_balance_and_splits_check.py first.")

    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    stats = compute_full_period_stats(df)

    # Save stats table
    out_csv = TABLES_DIR / "full_period_statistics.csv"
    out_xlsx = TABLES_DIR / "full_period_statistics.xlsx"

    stats.to_csv(out_csv, index=False)
    stats.to_excel(out_xlsx, index=False)

    # Scatter plots
    plot_scatter(df)

    print("Saved:")
    print("-", out_csv)
    print("-", out_xlsx)
    print("-", FIG_DIR, "(scatterplots)")


if __name__ == "__main__":
    main()