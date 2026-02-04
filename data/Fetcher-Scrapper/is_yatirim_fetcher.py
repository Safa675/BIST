"""
Batch price fetcher for BIST tickers via isyatirimhisse.

What it does:
- Pulls daily prices for each ticker between START_DATE and END_DATE.
- Resamples to quarter-end closes (last trading day of each quarter).
- Saves per-ticker Excel (daily + quarterly) and CSV (quarterly only).
- Builds a combined quarterly CSV across all tickers.

Run: python is_yatirim_fetcher.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from isyatirimhisse import fetch_stock_data


START_DATE = "01-01-2016"
END_DATE = "31-12-2026"
OUTPUT_DIR = Path("isyatirim_prices")

# Full ticker universe provided
TICKERS: List[str] = [
    "A1CAP", "A1YEN", "ACSEL", "ADEL", "ADESE", "ADGYO", "AEFES", "AFYON",
    "AGESA", "AGHOL", "AGROT", "AGYO", "AHGAZ", "AHSGY", "AKBNK", "AKCNS",
    "AKENR", "AKFGY", "AKFIS", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN",
    "AKSGY", "AKSUE", "AKYHO", "ALARK", "ALBRK", "ALCAR", "ALCTL", "ALFAS",
    "ALGYO", "ALKA", "ALKIM", "ALKLC", "ALMAD", "ALTNY", "ALVES", "ANELE",
    "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARCLK", "ARDYZ", "ARENA", "ARMGD",
    "ARSAN", "ARTMS", "ARZUM", "ASELS", "ASGYO", "ASTOR", "ASUZU", "ATAGY",
    "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATSYH", "AVGYO", "AVHOL", "AVOD",
    "AVPGY", "AVTUR", "AYCES", "AYDEM", "AYEN", "AYES", "AYGAZ", "AZTEK",
    "BAGFS", "BAHKM", "BAKAB", "BALAT", "BALSU", "BANVT", "BARMA", "BASCM",
    "BASGZ", "BAYRK", "BEGYO", "BERA", "BESLR", "BEYAZ", "BFREN", "BIENY",
    "BIGCH", "BIGEN", "BIGTK", "BIMAS", "BINBN", "BINHO", "BIOEN", "BIZIM",
    "BJKAS", "BLCYT", "BLUME", "BMSCH", "BMSTL", "BNTAS", "BOBET", "BORLS",
    "BORSK", "BOSSA", "BRISA", "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN",
    "BRSAN", "BRYAT", "BSOKE", "BTCIM", "BUCIM", "BULGS", "BURCE", "BURVA",
    "BVSAN", "BYDNR", "CANTE", "CASA", "CATES", "CCOLA", "CELHA", "CEMAS",
    "CEMTS", "CEMZY", "CEOEM", "CGCAM", "CIMSA", "CLEBI", "CMBTN", "CMENT",
    "CONSE", "COSMO", "CRDFA", "CRFSA", "CUSAN", "CVKMD", "CWENE", "DAGHL",
    "DAGI", "DAPGM", "DARDL", "DCTTR", "DENGE", "DERHL", "DERIM", "DESA",
    "DESPC", "DEVA", "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMRGD",
    "DMSAS", "DNISI", "DOAS", "DOBUR", "DOCO", "DOFER", "DOFRB", "DOGUB",
    "DOHOL", "DOKTA", "DSTKF", "DUNYH", "DURDO", "DURKN", "DYOBY", "DZGYO",
    "EBEBK", "ECILC", "ECOGR", "ECZYT", "EDATA", "EDIP", "EFOR", "EFORC",
    "EGEEN", "EGEGY", "EGEPO", "EGGUB", "EGPRO", "EGSER", "EKGYO", "EKIZ",
    "EKOS", "EKSUN", "ELITE", "EMKEL", "EMNIS", "ENDAE", "ENERY", "ENJSA",
    "ENKAI", "ENSRI", "ENTRA", "EPLAS", "ERBOS", "ERCB", "EREGL", "ERSU",
    "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUHOL", "EUKYO", "EUPWR",
    "EUREN", "EUYO", "EYGYO", "FADE", "FENER", "FLAP", "FMIZP", "FONET",
    "FORMT", "FORTE", "FRIGO", "FROTO", "FZLGY", "GARAN", "GARFA", "GEDIK",
    "GEDZA", "GENIL", "GENTS", "GEREL", "GESAN", "GIPTA", "GLBMD", "GLCVY",
    "GLRMK", "GLRYH", "GLYHO", "GMTAS", "GOKNR", "GOLTS", "GOODY", "GOZDE",
    "GRNYO", "GRSEL", "GRTHO", "GRTRK", "GSDDE", "GSDHO", "GSRAY", "GUBRF",
    "GUNDG", "GWIND", "GZNMI", "HALKB", "HATEK", "HATSN", "HDFGS", "HEDEF",
    "HEKTS", "HKTM", "HLGYO", "HOROZ", "HRKET", "HTTBT", "HUBVC", "HUNER",
    "HURGZ", "ICBCT", "ICUGS", "IDEAS", "IDGYO", "IEYHO", "IHAAS", "IHEVA",
    "IHGZT", "IHLAS", "IHLGM", "IHYAY", "IMASM", "INDES", "INFO", "INGRM",
    "INTEK", "INTEM", "INVEO", "INVES", "IPEKE", "ISATR", "ISBIR", "ISBTR",
    "ISCTR", "ISDMR", "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISKUR", "ISMEN",
    "ISSEN", "ISYAT", "ITTFH", "IZENR", "IZFAS", "IZINV", "IZMDC", "JANTS",
    "KAPLM", "KAREL", "KARSN", "KARTN", "KARYE", "KATMR", "KAYSE", "KBORU",
    "KCAER", "KCHOL", "KENT", "KERVN", "KERVT", "KFEIN", "KGYO", "KIMMR",
    "KLGYO", "KLKIM", "KLMSN", "KLNMA", "KLRHO", "KLSER", "KLSYN", "KLYPV",
    "KMPUR", "KNFRT", "KOCMT", "KONKA", "KONTR", "KONYA", "KOPOL", "KORDS",
    "KOTON", "KOZAA", "KOZAL", "KRDMA", "KRDMB", "KRDMD", "KRGYO", "KRONT",
    "KRPLS", "KRSTL", "KRTEK", "KRVGD", "KSTUR", "KTLEV", "KTSKR", "KUTPO",
    "KUVVA", "KUYAS", "KZBGY", "KZGYO", "LIDER", "LIDFA", "LILAK", "LINK",
    "LKMNH", "LMKDC", "LOGO", "LRSHO", "LUKSK", "LYDHO", "LYDYE", "MAALT",
    "MACKO", "MAGEN", "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA", "MARMR",
    "MARTI", "MAVI", "MEDTR", "MEGAP", "MEGMT", "MEKAG", "MEPET", "MERCN",
    "MERIT", "MERKO", "METRO", "METUR", "MGROS", "MHRGY", "MIATK", "MIPAZ",
    "MMCAS", "MNDRS", "MNDTR", "MOBTL", "MOGAN", "MOPAS", "MPARK", "MRGYO",
    "MRSHL", "MSGYO", "MTRKS", "MTRYO", "MZHLD", "NATEN", "NETAS", "NIBAS",
    "NTGAZ", "NTHOL", "NUGYO", "NUHCM", "OBAMS", "OBASE", "ODAS", "ODINE",
    "OFSYM", "ONCSM", "ONRYT", "ORCAY", "ORGE", "ORMA", "OSMEN", "OSTIM",
    "OTKAR", "OTTO", "OYAKC", "OYAYO", "OYLUM", "OYYAT", "OZATD", "OZGYO",
    "OZKGY", "OZRDN", "OZSUB", "OZYSR", "PAGYO", "PAHOL", "PAMEL", "PAPIL",
    "PARSN", "PASEU", "PATEK", "PCILT", "PEHOL", "PEKGY", "PENGD", "PENTA",
    "PETKM", "PETUN", "PGSUS", "PINSU", "PKART", "PKENT", "PLTUR", "PNLSN",
    "PNSUT", "POLHO", "POLTK", "PRDGS", "PRKAB", "PRKME", "PRZMA", "PSDTC",
    "PSGYO", "QNBFB", "QNBFK", "QNBFL", "QNBTR", "QUAGR", "RALYH", "RAYSG",
    "REEDR", "RGYAS", "RNPOL", "RODRG", "ROYAL", "RTALB", "RUBNS", "RUZYE",
    "RYGYO", "RYSAS", "SAFKR", "SAHOL", "SAMAT", "SANEL", "SANFM", "SANKO",
    "SARKY", "SASA", "SAYAS", "SDTTR", "SEGMN", "SEGYO", "SEKFK", "SEKUR",
    "SELEC", "SELGD", "SELVA", "SERNT", "SEYKM", "SILVR", "SISE", "SKBNK",
    "SKTAS", "SKYLP", "SKYMD", "SMART", "SMRTG", "SMRVA", "SNGYO", "SNICA",
    "SNKRN", "SNPAM", "SODSN", "SOKE", "SOKM", "SONME", "SRVGY", "SUMAS",
    "SUNTK", "SURGY", "SUWEN", "TABGD", "TARKM", "TATEN", "TATGD", "TAVHL",
    "TBORG", "TCELL", "TCKRC", "TDGYO", "TEHOL", "TEKTU", "TERA", "TETMT",
    "TEZOL", "TGSAS", "THYAO", "TKFEN", "TKNSA", "TLMAN", "TMPOL", "TMSN",
    "TNZTP", "TOASO", "TRALT", "TRCAS", "TRENJ", "TRGYO", "TRHOL", "TRILC",
    "TRMET", "TSGYO", "TSKB", "TSPOR", "TTKOM", "TTRAK", "TUCLK", "TUKAS",
    "TUPRS", "TUREX", "TURGG", "TURSG", "UFUK", "ULAS", "ULKER", "ULUFA",
    "ULUSE", "ULUUN", "UMPAS", "UNLU", "USAK", "UZERB", "VAKBN", "VAKFA",
    "VAKFN", "VAKKO", "VANGD", "VBTYZ", "VERTU", "VERUS", "VESBE", "VESTL",
    "VKFYO", "VKGYO", "VKING", "VRGYO", "VSNMD", "YAPRK", "YATAS", "YAYLA",
    "YBTAS", "YEOTK", "YESIL", "YGGYO", "YGYO", "YIGIT", "YKBNK", "YKSLN",
    "YONGA", "YUNSA", "YYAPI", "YYLGD", "ZEDUR", "ZERGY", "ZOREN", "ZRGYO",
]


def pick_date_column(df: pd.DataFrame) -> str:
    # Common possibilities (library/website can change column naming)
    exact_matches = {"date", "tarih"}
    contains_matches = {"date", "tarih"}
    lower_cols = {col: str(col).lower() for col in df.columns}

    for col, low in lower_cols.items():
        if low in exact_matches:
            return col

    for col, low in lower_cols.items():
        if any(token in low for token in contains_matches):
            return col

    raise ValueError(f"Could not find a date column. Columns: {list(df.columns)}")


def pick_close_column(df: pd.DataFrame) -> str:
    candidates = [
        "close", "Close", "CLOSE",
        "kapanis", "Kapanis", "KAPANIS",
        "kapanış", "Kapanış", "KAPANIŞ",
        "closing", "Closing", "CLOSING",
        "Last", "LAST", "last",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    lower_cols = {col: str(col).lower() for col in df.columns}
    for col, low in lower_cols.items():
        if any(token in low for token in ["kapan", "close", "last"]):
            return col

    ohlc_like = [c for c in df.columns if str(c).lower() in {"open", "high", "low", "close"}]
    if "close" in [str(c).lower() for c in ohlc_like]:
        return [c for c in ohlc_like if str(c).lower() == "close"][0]

    raise ValueError(f"Could not find a close/last price column. Columns: {list(df.columns)}")


def fetch_and_process(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """
    Returns (daily_df, quarterly_df, error_message)
    """
    try:
        df_daily = fetch_stock_data(
            symbols=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
            save_to_excel=False,
        )
    except Exception as exc:  # noqa: BLE001
        return None, None, f"fetch failed: {exc}"

    if df_daily is None or len(df_daily) == 0:
        return None, None, "no data returned"

    try:
        date_col = pick_date_column(df_daily)
        df_daily[date_col] = pd.to_datetime(df_daily[date_col], dayfirst=True, errors="coerce")
        df_daily = df_daily.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

        close_col = pick_close_column(df_daily)
        df_daily[close_col] = pd.to_numeric(df_daily[close_col], errors="coerce")

        s_q = df_daily[close_col].dropna().resample("Q").last()
        df_quarterly = s_q.to_frame(name=f"{ticker}_Close_QuarterEnd")
        df_quarterly.index.name = "QuarterEnd"
    except Exception as exc:  # noqa: BLE001
        return df_daily, None, f"processing failed: {exc}"

    return df_daily, df_quarterly, None


def save_outputs(ticker: str, daily: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{ticker}_2016_2026"
    out_xlsx = OUTPUT_DIR / f"{base_name}_daily_and_quarterly.xlsx"
    out_csv = OUTPUT_DIR / f"{base_name}_quarterly.csv"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="daily")
        quarterly.to_excel(writer, sheet_name="quarterly")

    quarterly.to_csv(out_csv, encoding="utf-8-sig")


def main() -> None:
    successes: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    print(f"Fetching {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")
    for i, ticker in enumerate(TICKERS, start=1):
        print(f"[{i}/{len(TICKERS)}] {ticker} ...", end="", flush=True)
        daily, quarterly, err = fetch_and_process(ticker)
        if err:
            errors[ticker] = err
            print(f" failed ({err})")
            continue

        save_outputs(ticker, daily, quarterly)
        successes[ticker] = quarterly
        print(f" ok (daily {len(daily):,} rows, quarterly {len(quarterly):,} rows)")

    if successes:
        combined = pd.concat(successes.values(), axis=1)
        combined.index.name = "QuarterEnd"
        combined.to_csv(OUTPUT_DIR / "all_tickers_quarterly_2016_2026.csv", encoding="utf-8-sig")
        print(f"\nCombined quarterly saved to {OUTPUT_DIR/'all_tickers_quarterly_2016_2026.csv'} "
              f"with shape {combined.shape}")

    if errors:
        print("\nCompleted with errors:")
        for t, msg in errors.items():
            print(f"- {t}: {msg}")
    else:
        print("\nCompleted without errors.")


if __name__ == "__main__":
    main()
