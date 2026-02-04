import os
import time
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
BASE_DIR = "Fintables_Data"
WAIT_TIME = 2 

# Full list of tickers
TICKERS = [
    "A1CAP", "A1YEN", "ACSEL", "ADEL", "ADESE", "ADGYO", "AEFES", "AFYON", "AGESA", 
    "AGHOL", "AGROT", "AGYO", "AHGAZ", "AHSGY", "AKBNK", "AKCNS", "AKENR", "AKFGY", 
    "AKFIS", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN", "AKSGY", "AKSUE", "AKYHO", 
    "ALARK", "ALBRK", "ALCAR", "ALCTL", "ALFAS", "ALGYO", "ALKA", "ALKIM", "ALKLC", 
    "ALMAD", "ALTNY", "ALVES", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARCLK", 
    "ARDYZ", "ARENA", "ARMGD", "ARSAN", "ARTMS", "ARZUM", "ASELS", "ASGYO", "ASTOR", 
    "ASUZU", "ATAGY", "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATSYH", "AVGYO", "AVHOL", 
    "AVOD", "AVPGY", "AVTUR", "AYCES", "AYDEM", "AYEN", "AYES", "AYGAZ", "AZTEK", 
    "BAGFS", "BAHKM", "BAKAB", "BALAT", "BALSU", "BANVT", "BARMA", "BASCM", "BASGZ", 
    "BAYRK", "BEGYO", "BERA", "BESLR", "BEYAZ", "BFREN", "BIENY", "BIGCH", "BIGEN", 
    "BIGTK", "BIMAS", "BINBN", "BINHO", "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BLUME", 
    "BMSCH", "BMSTL", "BNTAS", "BOBET", "BORLS", "BORSK", "BOSSA", "BRISA", "BRKO", 
    "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BRSAN", "BRYAT", "BSOKE", "BTCIM", "BUCIM", 
    "BULGS", "BURCE", "BURVA", "BVSAN", "BYDNR", "CANTE", "CASA", "CATES", "CCOLA", 
    "CELHA", "CEMAS", "CEMTS", "CEMZY", "CEOEM", "CGCAM", "CIMSA", "CLEBI", "CMBTN", 
    "CMENT", "CONSE", "COSMO", "CRDFA", "CRFSA", "CUSAN", "CVKMD", "CWENE", "DAGHL", 
    "DAGI", "DAPGM", "DARDL", "DCTTR", "DENGE", "DERHL", "DERIM", "DESA", "DESPC", 
    "DEVA", "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMRGD", "DMSAS", "DNISI", 
    "DOAS", "DOBUR", "DOCO", "DOFER", "DOFRB", "DOGUB", "DOHOL", "DOKTA", "DSTKF", 
    "DUNYH", "DURDO", "DURKN", "DYOBY", "DZGYO", "EBEBK", "ECILC", "ECOGR", "ECZYT", 
    "EDATA", "EDIP", "EFOR", "EFORC", "EGEEN", "EGEGY", "EGEPO", "EGGUB", "EGPRO", 
    "EGSER", "EKGYO", "EKIZ", "EKOS", "EKSUN", "ELITE", "EMKEL", "EMNIS", "ENDAE", 
    "ENERY", "ENJSA", "ENKAI", "ENSRI", "ENTRA", "EPLAS", "ERBOS", "ERCB", "EREGL", 
    "ERSU", "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUHOL", "EUKYO", "EUPWR", 
    "EUREN", "EUYO", "EYGYO", "FADE", "FENER", "FLAP", "FMIZP", "FONET", "FORMT", 
    "FORTE", "FRIGO", "FROTO", "FZLGY", "GARAN", "GARFA", "GEDIK", "GEDZA", "GENIL", 
    "GENTS", "GEREL", "GESAN", "GIPTA", "GLBMD", "GLCVY", "GLRMK", "GLRYH", "GLYHO", 
    "GMTAS", "GOKNR", "GOLTS", "GOODY", "GOZDE", "GRNYO", "GRSEL", "GRTHO", "GRTRK", 
    "GSDDE", "GSDHO", "GSRAY", "GUBRF", "GUNDG", "GWIND", "GZNMI", "HALKB", "HATEK", 
    "HATSN", "HDFGS", "HEDEF", "HEKTS", "HKTM", "HLGYO", "HOROZ", "HRKET", "HTTBT", 
    "HUBVC", "HUNER", "HURGZ", "ICBCT", "ICUGS", "IDEAS", "IDGYO", "IEYHO", "IHAAS", 
    "IHEVA", "IHGZT", "IHLAS", "IHLGM", "IHYAY", "IMASM", "INDES", "INFO", "INGRM", 
    "INTEK", "INTEM", "INVEO", "INVES", "IPEKE", "ISATR", "ISBIR", "ISBTR", "ISCTR", 
    "ISDMR", "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISKUR", "ISMEN", "ISSEN", "ISYAT", 
    "ITTFH", "IZENR", "IZFAS", "IZINV", "IZMDC", "JANTS", "KAPLM", "KAREL", "KARSN", 
    "KARTN", "KARYE", "KATMR", "KAYSE", "KBORU", "KCAER", "KCHOL", "KENT", "KERVN", 
    "KERVT", "KFEIN", "KGYO", "KIMMR", "KLGYO", "KLKIM", "KLMSN", "KLNMA", "KLRHO", 
    "KLSER", "KLSYN", "KLYPV", "KMPUR", "KNFRT", "KOCMT", "KONKA", "KONTR", "KONYA", 
    "KOPOL", "KORDS", "KOTON", "KOZAA", "KOZAL", "KRDMA", "KRDMB", "KRDMD", "KRGYO", 
    "KRONT", "KRPLS", "KRSTL", "KRTEK", "KRVGD", "KSTUR", "KTLEV", "KTSKR", "KUTPO", 
    "KUVVA", "KUYAS", "KZBGY", "KZGYO", "LIDER", "LIDFA", "LILAK", "LINK", "LKMNH", 
    "LMKDC", "LOGO", "LRSHO", "LUKSK", "LYDHO", "LYDYE", "MAALT", "MACKO", "MAGEN", 
    "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA", "MARMR", "MARTI", "MAVI", "MEDTR", 
    "MEGAP", "MEGMT", "MEKAG", "MEPET", "MERCN", "MERIT", "MERKO", "METRO", "METUR", 
    "MGROS", "MHRGY", "MIATK", "MIPAZ", "MMCAS", "MNDRS", "MNDTR", "MOBTL", "MOGAN", 
    "MOPAS", "MPARK", "MRGYO", "MRSHL", "MSGYO", "MTRKS", "MTRYO", "MZHLD", "NATEN", 
    "NETAS", "NIBAS", "NTGAZ", "NTHOL", "NUGYO", "NUHCM", "OBAMS", "OBASE", "ODAS", 
    "ODINE", "OFSYM", "ONCSM", "ONRYT", "ORCAY", "ORGE", "ORMA", "OSMEN", "OSTIM", 
    "OTKAR", "OTTO", "OYAKC", "OYAYO", "OYLUM", "OYYAT", "OZATD", "OZGYO", "OZKGY", 
    "OZRDN", "OZSUB", "OZYSR", "PAGYO", "PAHOL", "PAMEL", "PAPIL", "PARSN", "PASEU", 
    "PATEK", "PCILT", "PEHOL", "PEKGY", "PENGD", "PENTA", "PETKM", "PETUN", "PGSUS", 
    "PINSU", "PKART", "PKENT", "PLTUR", "PNLSN", "PNSUT", "POLHO", "POLTK", "PRDGS", 
    "PRKAB", "PRKME", "PRZMA", "PSDTC", "PSGYO", "QNBFB", "QNBFK", "QNBFL", "QNBTR", 
    "QUAGR", "RALYH", "RAYSG", "REEDR", "RGYAS", "RNPOL", "RODRG", "ROYAL", "RTALB", 
    "RUBNS", "RUZYE", "RYGYO", "RYSAS", "SAFKR", "SAHOL", "SAMAT", "SANEL", "SANFM", 
    "SANKO", "SARKY", "SASA", "SAYAS", "SDTTR", "SEGMN", "SEGYO", "SEKFK", "SEKUR", 
    "SELEC", "SELGD", "SELVA", "SERNT", "SEYKM", "SILVR", "SISE", "SKBNK", "SKTAS", 
    "SKYLP", "SKYMD", "SMART", "SMRTG", "SMRVA", "SNGYO", "SNICA", "SNKRN", "SNPAM", 
    "SODSN", "SOKE", "SOKM", "SONME", "SRVGY", "SUMAS", "SUNTK", "SURGY", "SUWEN", 
    "TABGD", "TARKM", "TATEN", "TATGD", "TAVHL", "TBORG", "TCELL", "TCKRC", "TDGYO", 
    "TEHOL", "TEKTU", "TERA", "TETMT", "TEZOL", "TGSAS", "THYAO", "TKFEN", "TKNSA", 
    "TLMAN", "TMPOL", "TMSN", "TNZTP", "TOASO", "TRALT", "TRCAS", "TRENJ", "TRGYO", 
    "TRHOL", "TRILC", "TRMET", "TSGYO", "TSKB", "TSPOR", "TTKOM", "TTRAK", "TUCLK", 
    "TUKAS", "TUPRS", "TUREX", "TURGG", "TURSG", "UFUK", "ULAS", "ULKER", "ULUFA", 
    "ULUSE", "ULUUN", "UMPAS", "UNLU", "USAK", "UZERB", "VAKBN", "VAKFA", "VAKFN", 
    "VAKKO", "VANGD", "VBTYZ", "VERTU", "VERUS", "VESBE", "VESTL", "VKFYO", "VKGYO", 
    "VKING", "VRGYO", "VSNMD", "YAPRK", "YATAS", "YAYLA", "YBTAS", "YEOTK", "YESIL", 
    "YGGYO", "YGYO", "YIGIT", "YKBNK", "YKSLN", "YONGA", "YUNSA", "YYAPI", "YYLGD", 
    "ZEDUR", "ZERGY", "ZOREN", "ZRGYO"
]

def download_file_robust(page, ticker, type_name, url):
    folder_path = os.path.join(BASE_DIR, ticker)
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{ticker}_{type_name}.xlsx"
    file_path = os.path.join(folder_path, filename)

    if os.path.exists(file_path):
        print(f"   -> [SKIP] {filename} exists.")
        return

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        
        # --- FIX STARTS HERE ---
        # We look for the button but use .first so Playwright doesn't panic if it finds two
        download_button = None

        # Strategy 1: Find text "Excel'e Aktar" (take the first one found)
        # We check count() first to avoid strict mode errors on is_visible()
        if page.get_by_text("Excel'e Aktar").count() > 0:
            download_button = page.get_by_text("Excel'e Aktar").first
        
        # Strategy 2: Fallback to Excel Icon
        elif page.locator(".fa-file-excel").count() > 0:
            download_button = page.locator(".fa-file-excel").first
            
        if not download_button:
            print(f"   -> [FAIL] Button not found.")
            return

        # Start Download
        with page.expect_download(timeout=45000) as download_info:
            # force=True helps if the button is slightly covered by an overlay
            download_button.click(force=True) 
        
        download = download_info.value
        download.save_as(file_path)
        print(f"   -> [SUCCESS] Saved {filename}")
        
        time.sleep(WAIT_TIME)

    except Exception as e:
        # Print a short error message so it doesn't flood the console
        error_msg = str(e).split("\n")[0]
        print(f"   -> [ERROR] {ticker} {type_name}: {error_msg}")

def main():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, accept_downloads=True)
        page = context.new_page()

        print("---------------------------------------------------------")
        print("STEP 1: LOGIN REQUIRED")
        print("1. Go to fintables.com/giris and LOG IN.")
        print("2. Make sure your Premium session is active.")
        print("3. Come back here and press ENTER.")
        print("---------------------------------------------------------")
        
        page.goto("https://fintables.com/giris")
        input("Press Enter AFTER you are fully logged in...")

        print(f"Starting downloads...")

        for index, ticker in enumerate(TICKERS):
            print(f"[{index+1}/{len(TICKERS)}] Processing {ticker}...")

            # Income Statement
            url_income = f"https://fintables.com/sirketler/{ticker}/finansal-tablolar/gelir-tablosu"
            download_file_robust(page, ticker, "Gelir_Tablosu", url_income)

            # Balance Sheet
            url_balance = f"https://fintables.com/sirketler/{ticker}/finansal-tablolar/bilanco"
            download_file_robust(page, ticker, "Bilanco", url_balance)

        print("\nAll tasks completed!")
        browser.close()

if __name__ == "__main__":
    main()