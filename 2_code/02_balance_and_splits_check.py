import pandas as pd
import yfinance as yf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "1_data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "1_data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "HCL": "HCLTECH.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
}

START_DATE = "2023-01-03"
END_DATE_INCLUSIVE = "2024-06-28"


def _strip_tz(dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Ensure index is timezone-naive (required for clean comparisons).
    If tz-aware, convert to naive local time (drops tz info).
    """
    if getattr(dt_index, "tz", None) is not None:
        return dt_index.tz_localize(None)
    return dt_index


def read_raw_close(company_name: str) -> pd.Series:
    """
    Reads raw CSV and returns a Series indexed by Date (timezone-naive).
    Robust to extra columns / junk rows like 'Ticker'.
    """
    path = RAW_DIR / f"{company_name}_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}. Run 01_data_collection.py first.")

    df = pd.read_csv(path)

    # Locate date column
    date_col = None
    for c in ["Date", "date", "Datetime", "datetime", "index"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # Locate close column
    close_col = None
    for c in ["Close", "close", "Adj Close", "AdjClose", "adj_close"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError(f"{path} missing a Close-like column. Columns: {list(df.columns)}")

    df = df[[date_col, close_col]].copy()
    df.columns = ["Date", "Close"]

    # Parse Date safely; drop invalid rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Convert to timezone-naive (drop tz info)
    df["Date"] = df["Date"].dt.tz_convert(None)

    # Force numeric close
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).set_index("Date")

    # Ensure index is tz-naive (extra safety)
    df.index = _strip_tz(df.index)

    return df["Close"].rename(company_name)


def check_splits() -> pd.DataFrame:
    """
    Checks split events within the window (inclusive).
    """
    rows = []
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE_INCLUSIVE)

    for name, tkr in TICKERS.items():
        tk = yf.Ticker(tkr)
        splits = tk.splits  # Series indexed by date (tz-naive typically)

        if splits is None or len(splits) == 0:
            continue

        # Make sure splits index is tz-naive too
        splits.index = _strip_tz(pd.to_datetime(splits.index, errors="coerce"))

        splits_in_window = splits[(splits.index >= start) & (splits.index <= end)]

        for dt, ratio in splits_in_window.items():
            rows.append({
                "Company": name,
                "Ticker": tkr,
                "SplitDate": pd.Timestamp(dt).date(),
                "SplitRatio": float(ratio),
            })

    return pd.DataFrame(rows)


def main():
    # 1) Read all 4 companies
    close_df = pd.concat([read_raw_close(name) for name in TICKERS.keys()], axis=1)

    # Ensure index tz-naive
    close_df.index = _strip_tz(close_df.index)

    # 2) Restrict to exact window (tz-naive comparisons)
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE_INCLUSIVE)
    close_df = close_df.loc[(close_df.index >= start) & (close_df.index <= end)]

    # 3) Balance: keep only dates where ALL companies have data
    n_raw = close_df.shape[0]
    balanced = close_df.dropna(how="any").copy()
    n_bal = balanced.shape[0]

    if balanced.empty:
        raise RuntimeError("Balanced dataset is empty. Raw files may not overlap in dates.")
    if balanced.isna().any().any():
        raise RuntimeError("Balanced dataset still has missing values.")
    if balanced.index.duplicated().any():
        raise RuntimeError("Duplicate dates found in balanced dataset.")

    balanced = balanced.sort_index()

    # 4) Split check
    splits_table = check_splits()

    # 5) Save processed with explicit Date column
    out_csv = PROCESSED_DIR / "it_sector_balanced_close_2023_2024.csv"
    out_xlsx = PROCESSED_DIR / "it_sector_balanced_close_2023_2024.xlsx"

    balanced.index.name = "Date"
    balanced.to_csv(out_csv, index=True, index_label="Date")
    balanced.to_excel(out_xlsx, index=True)

    if not splits_table.empty:
        splits_table.to_csv(PROCESSED_DIR / "stock_split_events_in_window.csv", index=False)

    # 6) Print summary
    print("=== BALANCING SUMMARY ===")
    print(f"Raw rows (within window):     {n_raw}")
    print(f"Balanced rows (common dates): {n_bal}")
    print(f"Dropped rows:                 {n_raw - n_bal}")
    print(f"Final date range:             {balanced.index.min().date()} to {balanced.index.max().date()}")

    print("\n=== STOCK SPLIT CHECK (within window) ===")
    if splits_table.empty:
        print("No split events found within the study window for these tickers.")
    else:
        print("WARNING: Split events found within window:")
        print(splits_table.to_string(index=False))

    print("\nSaved:")
    print("-", out_csv)
    print("-", out_xlsx)

    print("\nHead:")
    print(balanced.head(3))
    print("\nTail:")
    print(balanced.tail(3))


if __name__ == "__main__":
    main()
