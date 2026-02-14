#!/usr/bin/env python3
"""
Classify all BIST stocks by sector and industry using borsapy.
Handles rate-limiting with progressive backoff.
Saves progress incrementally.
"""
import logging
import os
import sys
import time

import borsapy as bp
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_CSV = 'data/bist_sector_classification.csv'
OUTPUT_PARQUET = 'data/bist_sector_classification.parquet'

def main():
    # Get all tickers
    prices = pd.read_parquet('data/bist_prices_full.parquet')
    all_tickers = sorted(set(t.replace('.IS','') for t in prices['Ticker'].unique()))
    
    # Load already-classified tickers (if any)
    already_done = {}
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        for _, row in existing.iterrows():
            if str(row.get('sector', '')) not in ('ERROR', '', 'nan'):
                already_done[row['ticker']] = {
                    'ticker': row['ticker'],
                    'sector': row['sector'],
                    'industry': row['industry'],
                }
    
    remaining = [t for t in all_tickers if t not in already_done]
    logger.info(f"Total: {len(all_tickers)} | Already done: {len(already_done)} | Remaining: {len(remaining)}")
    sys.stdout.flush()
    
    results = list(already_done.values())
    delay = 1.0
    consecutive_errors = 0
    
    for i, ticker in enumerate(remaining):
        try:
            t = bp.Ticker(ticker)
            info = t.info
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            results.append({'ticker': ticker, 'sector': sector, 'industry': industry})
            consecutive_errors = 0
            delay = max(0.5, delay - 0.1)  # Speed up on success
        except Exception as e:
            err_msg = str(e)
            if '429' in err_msg:
                consecutive_errors += 1
                delay = min(10, delay + 1)  # Back off on rate limit
                logger.warning(f"  ⚠️  Rate limited at {ticker}, backing off to {delay}s")
                sys.stdout.flush()
                time.sleep(delay * 2)  # Extra wait after rate limit
                # Retry once
                try:
                    t = bp.Ticker(ticker)
                    info = t.info
                    results.append({'ticker': ticker, 'sector': info.get('sector',''), 'industry': info.get('industry','')})
                    consecutive_errors = 0
                except Exception as retry_exc:
                    logger.warning(f"  ⚠️  Retry failed for {ticker}: {retry_exc}")
                    sys.stdout.flush()
                    results.append({'ticker': ticker, 'sector': '', 'industry': ''})
            else:
                results.append({'ticker': ticker, 'sector': '', 'industry': ''})
        
        if (i + 1) % 25 == 0:
            n_classified = sum(1 for r in results if r.get('sector'))
            logger.info(f"  {i+1}/{len(remaining)} done ({n_classified} classified, delay={delay:.1f}s)")
            sys.stdout.flush()
            # Save progress
            df = pd.DataFrame(results).sort_values('ticker').reset_index(drop=True)
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        time.sleep(delay)
    
    # Final save
    df = pd.DataFrame(results).sort_values('ticker').reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    df.to_parquet(OUTPUT_PARQUET)
    
    classified = df[df['sector'].notna() & (df['sector'] != '')]
    unclassified = df[df['sector'].isna() | (df['sector'] == '')]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {len(classified)} classified, {len(unclassified)} unclassified")
    logger.info("\nSECTOR DISTRIBUTION:")
    logger.info(classified['sector'].value_counts().to_string())
    logger.info("\nINDUSTRY DISTRIBUTION:")
    logger.info(classified['industry'].value_counts().to_string())
    logger.info(f"\nSaved to {OUTPUT_CSV} and {OUTPUT_PARQUET}")

if __name__ == '__main__':
    main()
