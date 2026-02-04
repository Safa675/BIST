#!/usr/bin/env python3
import argparse
import csv
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_CONFIG = {
    "headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    },
    # Fill in once you capture a working request in DevTools or via HAR.
    "company_list_request": {
        "method": "GET",
        "url": "https://fintables.com/radar/hisse-senetleri",
        "params": {},
        "json": None,
        "data": None,
    },
    # Template fields available: {ticker}, {year}, {quarter}
    "market_data_request": {
        "method": "GET",
        "url": "REPLACE_WITH_MARKET_DATA_URL_TEMPLATE",
        "params": {},
        "json": None,
        "data": None,
    },
    # Optional JSON paths (dot notation) to avoid heuristic matching.
    "market_cap_path": "",
    "shares_path": "",
    "price_path": "",
    # Optional hard-coded cookies (otherwise use --cookies).
    "cookies": {},
}


KEY_PATTERNS = {
    "market_cap": [
        "marketcap",
        "market_cap",
        "marketcapitalization",
        "piyasa",
        "piyasadegeri",
        "piyasa_degeri",
        "marketvalue",
    ],
    "shares": [
        "shares",
        "sharecount",
        "share_count",
        "hisse",
        "hisseadet",
        "hisse_adet",
        "hisselersayisi",
    ],
    "price": [
        "price",
        "last",
        "close",
        "closing",
        "fiyat",
    ],
}

NAME_KEYS = {
    "name",
    "company",
    "companyname",
    "stockname",
    "title",
    "unvan",
    "longname",
}

CODE_KEYS = {
    "ticker",
    "symbol",
    "code",
    "stockcode",
    "stock_code",
}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    merged = DEFAULT_CONFIG.copy()
    merged.update(data or {})
    return merged


def load_cookie_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Cookie file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    # JSON: Playwright storage state or cookie list
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        data = json.loads(text)
        if isinstance(data, dict) and "cookies" in data:
            cookies = data["cookies"]
        elif isinstance(data, list):
            cookies = data
        else:
            cookies = []
        return {c["name"]: c["value"] for c in cookies if "name" in c and "value" in c}

    # Netscape cookies.txt format
    cookies: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        name = parts[5]
        value = parts[6]
        cookies[name] = value
    return cookies


def autodetect_cookie_file() -> Optional[Path]:
    candidates = [
        "fintables.com_cookies.txt",
        "cookies.txt",
        "cookies.json",
    ]
    for name in candidates:
        path = Path(name)
        if path.exists():
            return path
    return None


def render_template(value: Any, **vars: str) -> Any:
    if isinstance(value, str):
        return value.format(**vars)
    if isinstance(value, dict):
        return {k: render_template(v, **vars) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, **vars) for v in value]
    return value


def request_json(session: requests.Session, req: Dict[str, Any]) -> Any:
    method = (req.get("method") or "GET").upper()
    url = req.get("url")
    if not url or "REPLACE_WITH" in url:
        raise ValueError("Request URL is missing. Update your config.")

    params = req.get("params") or None
    json_body = req.get("json") if req.get("json") is not None else None
    data = req.get("data") if req.get("data") is not None else None

    response = session.request(method, url, params=params, json=json_body, data=data, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type or response.text.strip().startswith("{"):
        return response.json()
    raise ValueError(f"Non-JSON response from {url}")


def request_text(session: requests.Session, url: str) -> str:
    if not url:
        raise ValueError("URL is missing.")
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def find_first_numeric(obj: Any, key_patterns: List[str]) -> Optional[float]:
    patterns = [re.compile(pat, re.IGNORECASE) for pat in key_patterns]
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for k, v in current.items():
                if any(p.search(str(k)) for p in patterns):
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        try:
                            return float(v.replace(",", "").strip())
                        except ValueError:
                            pass
                stack.append(v)
        elif isinstance(current, list):
            stack.extend(current)
    return None


def extract_next_data(text: str) -> Optional[Any]:
    match = re.search(
        r'<script[^>]+id="__NEXT_DATA__"[^>]*>(?P<data>.*?)</script>',
        text,
        re.DOTALL,
    )
    if not match:
        return None
    data_raw = match.group("data").strip()
    if not data_raw:
        return None
    try:
        return json.loads(data_raw)
    except json.JSONDecodeError:
        return None


def normalize_key(key: str) -> str:
    return str(key).replace("_", "").lower()


def pick_first_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def extract_stock_records(payload: Any) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    stack = [payload]
    seen = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            name = None
            code = None
            for key, value in current.items():
                norm = normalize_key(key)
                if norm in NAME_KEYS:
                    name = pick_first_str(value) or name
                if norm in CODE_KEYS:
                    code = pick_first_str(value) or code

            if name and code:
                dedupe_key = (code, name)
                if dedupe_key not in seen:
                    records.append({"ticker": code, "name": name})
                    seen.add(dedupe_key)

            for value in current.values():
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return records


def extract_stock_records_from_html(text: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen = set()
    pattern = re.compile(
        r'<a[^>]+href="[^"]*/(?:hisse|sirketler)/([A-Z0-9]+)[^"]*"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        code = match.group(1).strip().upper()
        raw_name = re.sub(r"<[^>]+>", "", match.group(2))
        name = html.unescape(raw_name).strip()
        if not name:
            continue
        dedupe_key = (code, name)
        if dedupe_key in seen:
            continue
        records.append({"ticker": code, "name": name})
        seen.add(dedupe_key)
    return records


def get_by_path(obj: Any, path: str) -> Optional[Any]:
    if not path:
        return None
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def extract_tickers(payload: Any) -> List[str]:
    # Heuristic: look for list of dicts with a symbol/ticker-like key.
    candidates = []
    stack = [payload]
    keys = {"ticker", "symbol", "code", "stockcode", "stock_code"}
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for k, v in current.items():
                if isinstance(v, list):
                    stack.append(v)
                else:
                    stack.append(v)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if str(k).replace("_", "").lower() in keys:
                            if isinstance(v, str) and v.strip():
                                candidates.append(v.strip())
                else:
                    stack.append(item)
    # De-duplicate while preserving order.
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fetch_stock_list(session: requests.Session, url: str) -> List[Dict[str, str]]:
    text = request_text(session, url)
    next_data = extract_next_data(text)
    records = []
    if next_data is not None:
        records = extract_stock_records(next_data)
    if not records:
        records = extract_stock_records_from_html(text)
    return records


def read_tickers_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for field in ("ticker", "symbol", "code", "stock_code"):
                if field in reader.fieldnames:
                    return [row[field].strip() for row in reader if row.get(field)]
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_quarters(start_year: int, end_year: int, quarters: Iterable[int]) -> Iterable[Tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        for quarter in quarters:
            yield year, quarter


def load_existing_rows(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add((row.get("ticker"), row.get("year"), row.get("quarter")))
    return seen


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_quarters(value: str) -> List[int]:
    parts = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts.append(int(raw))
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch market cap, shares, and price per quarter from Fintables."
    )
    parser.add_argument("--config", default="fintables_config.json", help="Path to JSON config.")
    parser.add_argument("--tickers", help="CSV or TXT of tickers; skips company list request.")
    parser.add_argument("--cookies", help="Cookie file (Playwright storage state or cookies.txt).")
    parser.add_argument("--out", default="fintables_market_data.csv", help="Output CSV path.")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--quarters", default="1,2,3,4")
    parser.add_argument("--rate-limit", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--strict-paths", action="store_true", help="Require JSON paths in config.")
    parser.add_argument(
        "--stock-list",
        action="store_true",
        help="Fetch stock list from the Fintables radar page.",
    )
    parser.add_argument(
        "--stock-list-url",
        default="https://fintables.com/radar/hisse-senetleri",
        help="URL of the stock list page.",
    )
    parser.add_argument(
        "--stock-list-out",
        default="fintables_stocks.csv",
        help="Output CSV path for the stock list.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    session = requests.Session()
    session.headers.update(config.get("headers") or {})

    cookies = {}
    cookie_path = Path(args.cookies) if args.cookies else autodetect_cookie_file()
    if cookie_path:
        cookies.update(load_cookie_file(cookie_path))
    cookies.update(config.get("cookies") or {})
    for name, value in cookies.items():
        session.cookies.set(name, value)

    if args.stock_list:
        records = fetch_stock_list(session, args.stock_list_url)
        if not records:
            raise RuntimeError(
                "No stock records found. The page may require cookies or a JS challenge."
            )
        out_path = Path(args.stock_list_out)
        ensure_parent(out_path)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "name"])
            writer.writeheader()
            writer.writerows(records)
        return 0

    if args.tickers:
        tickers = read_tickers_file(Path(args.tickers))
    else:
        company_req = config.get("company_list_request") or {}
        data = request_json(session, company_req)
        tickers = extract_tickers(data)

    if not tickers:
        raise RuntimeError(
            "No tickers found. Provide --tickers or fix company_list_request in config."
        )

    quarters = parse_quarters(args.quarters)
    out_path = Path(args.out)
    ensure_parent(out_path)
    existing = load_existing_rows(out_path)
    new_file = not out_path.exists()

    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ticker", "year", "quarter", "market_cap", "shares", "price"]
        )
        if new_file:
            writer.writeheader()

        for ticker in tickers:
            for year, quarter in iter_quarters(args.start_year, args.end_year, quarters):
                key = (ticker, str(year), str(quarter))
                if key in existing:
                    continue

                req = render_template(
                    config.get("market_data_request") or {},
                    ticker=ticker,
                    year=str(year),
                    quarter=str(quarter),
                )
                payload = request_json(session, req)

                market_cap = get_by_path(payload, config.get("market_cap_path", ""))
                shares = get_by_path(payload, config.get("shares_path", ""))
                price = get_by_path(payload, config.get("price_path", ""))

                if args.strict_paths:
                    if market_cap is None or shares is None or price is None:
                        raise RuntimeError(
                            "Strict mode enabled, but one or more JSON paths are missing."
                        )
                else:
                    if market_cap is None:
                        market_cap = find_first_numeric(payload, KEY_PATTERNS["market_cap"])
                    if shares is None:
                        shares = find_first_numeric(payload, KEY_PATTERNS["shares"])
                    if price is None:
                        price = find_first_numeric(payload, KEY_PATTERNS["price"])

                writer.writerow(
                    {
                        "ticker": ticker,
                        "year": year,
                        "quarter": quarter,
                        "market_cap": market_cap,
                        "shares": shares,
                        "price": price,
                    }
                )
                existing.add(key)
                if args.rate_limit > 0:
                    time.sleep(args.rate_limit)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return_code = 1
        raise SystemExit(return_code)
