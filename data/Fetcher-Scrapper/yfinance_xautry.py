import os
import yfinance as yf
import pandas as pd

start = "2013-01-01"
end   = "2026-02-01"

def safe_download(ticker):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"Download failed for {ticker}")
    return df["Close"]

# --- Gold ---
try:
    xau = safe_download("XAUUSD=X")
    print("Using XAUUSD=X (spot)")
except Exception:
    print("XAUUSD=X failed, falling back to GC=F (gold futures)")
    xau = safe_download("GC=F")

# --- USD/TRY ---
usd_try = safe_download("USDTRY=X")

# --- Align & compute ---
df = pd.concat([xau, usd_try], axis=1)
df.columns = ["XAU_USD", "USD_TRY"]

df["XAU_TRY"] = df["XAU_USD"] * df["USD_TRY"]
df = df.dropna()

# --- Save next to script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "xau_try_2013_2026.csv")
df.to_csv(csv_path)

print(f"Saved to {csv_path}")
print(df.head())
