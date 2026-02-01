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
END_DATE_EXCLUSIVE = "2024-06-29"  # to include 2024-06-28

def main():
    for name, tkr in TICKERS.items():
        df = yf.download(
            tkr,
            start=START_DATE,
            end=END_DATE_EXCLUSIVE,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {name} ({tkr}).")

        # Make sure index is Date
        df = df.reset_index()

        # Keep only Date and Close
        if "Date" not in df.columns:
            # yfinance sometimes uses 'index' depending on pandas versions
            if "index" in df.columns:
                df.rename(columns={"index": "Date"}, inplace=True)
            else:
                raise RuntimeError(f"Could not find Date column for {name} download output.")

        if "Close" not in df.columns:
            raise RuntimeError(f"Could not find Close column for {name} download output.")

        out = df[["Date", "Close"]].copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.sort_values("Date")

        out.to_csv(RAW_DIR / f"{name}_raw.csv", index=False)
        print(f"Saved {name}_raw.csv ({len(out)} rows)")

if __name__ == "__main__":
    main()
