import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "1_data" / "processed"
TABLES_DIR = PROJECT_ROOT / "3_outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PROCESSED_DIR / "it_sector_balanced_close_2023_2024.csv"
START_DATE = "2023-01-03"
END_DATE_INCLUSIVE = "2024-06-28"


def make_4month_quarters(index: pd.DatetimeIndex) -> pd.Series:
    m = index.month
    qnum = ((m - 1) // 4) + 1
    labels = [f"{y}_Q{qn}" for y, qn in zip(index.year, qnum)]
    return pd.Series(labels, index=index, name="Quarter")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Run 02 first.")

    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE_INCLUSIVE)]

    quarters = make_4month_quarters(df.index)
    df_q = df.copy()
    df_q["Quarter"] = quarters.values

    out_xlsx = TABLES_DIR / "correlation_matrices.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # pooled correlation
        pooled_corr = df.corr(method="pearson")
        pooled_corr.to_excel(writer, sheet_name="Pooled")

        # per-quarter correlation matrices
        for q in df_q["Quarter"].unique():
            sub = df_q[df_q["Quarter"] == q].drop(columns=["Quarter"])
            corr = sub.corr(method="pearson")
            # sheet name limit: 31 chars
            sheet = q[:31]
            corr.to_excel(writer, sheet_name=sheet)

    print("Saved correlation matrices to:", out_xlsx)


if __name__ == "__main__":
    main()
