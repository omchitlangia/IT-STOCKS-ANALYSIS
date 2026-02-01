import pandas as pd
import yfinance as yf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "1_data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "HCL": "HCLTECH.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
}

START_DATE = "2023-01-03"
END_DATE = "2024-06-29"  # end is exclusive in yfinance; use next day to include Jun 28

def main():
    for name, tkr in TICKERS.items():
        df = yf.download(tkr, start=START_DATE, end=END_DATE, interval="1d", progress=False)
        if df.empty:
            raise RuntimeError(f"No data returned for {name} ({tkr}).")

        # Keep Date index and Close only (still raw — not balanced yet)
        out = df[["Close"]].copy()
        out.index.name = "Date"
        out.to_csv(RAW_DIR / f"{name}_raw.csv")

    print("Saved raw CSVs to:", RAW_DIR)

if __name__ == "__main__":
    main()
