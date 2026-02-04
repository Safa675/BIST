import csv
import time
from playwright.sync_api import sync_playwright

URL = "https://fintables.com/radar/hisse-senetleri"

def scrape_fintables_robust():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        print(f"1. Navigating to {URL}...")
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        
        print("2. Please complete the Cloudflare challenge manually if it appears...")
        print("   Once the table is visible, press Enter in this terminal to start scraping.")
        input() # WAIT FOR USER
        
        print("3. Focusing on the table and starting extraction...")
        
        # CLICK the first row to ensure the table has "Focus" so keyboard keys work
        try:
            page.click("table tbody tr:first-child", timeout=5000)
        except:
            print("Could not click table row automatically. Please click a row in the browser manually, then wait.")
            time.sleep(3)

        unique_stocks = {}
        no_new_data_counter = 0
        max_retries = 15  # How many times to press PageDown without new data before stopping
        
        while True:
            # --- SCRAPE VISIBLE ROWS ---
            # We look for links that contain stock codes
            links = page.query_selector_all("table tbody tr")
            
            new_data_found = False
            
            for row in links:
                try:
                    # Usually the stock code is in the first cell, Name in the second
                    cells = row.query_selector_all("td")
                    if len(cells) >= 2:
                        code = cells[0].inner_text().strip()
                        name = cells[1].inner_text().strip()
                        
                        if code and code not in unique_stocks:
                            unique_stocks[code] = name
                            new_data_found = True
                except:
                    continue
            
            print(f"Stocks Found: {len(unique_stocks)}", end="\r")

            # --- CHECK TERMINATION ---
            if not new_data_found:
                no_new_data_counter += 1
            else:
                no_new_data_counter = 0 # Reset counter if we found fresh data
            
            # If we pressed PageDown 15 times (approx 15 seconds) and found NOTHING new, we are done.
            if no_new_data_counter >= max_retries:
                print(f"\nNo new data found for {max_retries} attempts. Stopping.")
                break

            # --- SCROLL ACTION ---
            # Press PageDown to scroll the focused element
            page.keyboard.press("PageDown")
            time.sleep(0.5) # Wait for virtual scroll to render

        browser.close()

        # --- SAVE CSV ---
        print(f"\nSaving {len(unique_stocks)} stocks to 'fintables_stocks.csv'...")
        with open("fintables_stocks.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "name"])
            writer.writeheader()
            rows = [{"ticker": k, "name": v} for k, v in unique_stocks.items()]
            writer.writerows(rows)
            
        print("Done!")

if __name__ == "__main__":
    scrape_fintables_robust()