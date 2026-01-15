import pandas as pd
import os
import datetime
import pytz
import yfinance as yf
import numpy as np
from flask import Flask, render_template_string, request, redirect, session
import requests
import re
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# Load Credentials
load_dotenv()
KITE_API_KEY = os.getenv("KITE_API_KEY", "").strip().replace('"', '').replace("'", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "").strip().replace('"', '').replace("'", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "").strip().replace('"', '').replace("'", "")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print(f"DEBUG: KITE_API_KEY loaded: {'YES' if KITE_API_KEY else 'NO'} (Starts with: {KITE_API_KEY[:3] if KITE_API_KEY else 'N/A'})")

CSV_FILE = r"master_nse_data.csv"
WATCHLIST_FILE = "watchlist.json"

app = Flask(__name__)
app.secret_key = "price_ladder_pro_secret_key" # Required for session

# Global Kite Object
kite = None
VOL_BASELINE = {} # Cache for Avg Daily Volume
VOL_SAMPLES = {} # Store recent (time, total_volume) snapshots
NFO_INSTRUMENTS = None # Cache for F&O instruments
INSTRUMENT_MAP = {} # Cache for symbol -> token mapping

if KITE_API_KEY:
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        if KITE_ACCESS_TOKEN:
            kite.set_access_token(KITE_ACCESS_TOKEN)
    except Exception as e:
        print(f"DEBUG: Kite initialization error: {e}")

def get_instrument_token(symbol):
    global INSTRUMENT_MAP
    if not kite: return None
    
    if not INSTRUMENT_MAP:
        print("DEBUG: Loading Master Instrument Map from Zerodha...")
        try:
            # We fetch NSE and NFO
            inst = kite.instruments(["NSE", "NFO"])
            for i in inst:
                # Store primary NSE symbol or NFO tradingsymbol
                key = i['tradingsymbol']
                INSTRUMENT_MAP[key] = i['instrument_token']
            # Special Index Mapping
            INSTRUMENT_MAP['NIFTY'] = 256265 # Nifty 50
            INSTRUMENT_MAP['NIFTY 50'] = 256265
            INSTRUMENT_MAP['BANKNIFTY'] = 260105 # Nifty Bank
            INSTRUMENT_MAP['NIFTY BANK'] = 260105
            print(f"DEBUG: Map Loaded with {len(INSTRUMENT_MAP)} instruments.")
        except Exception as e:
            print(f"Error loading instruments: {e}")
            
    return INSTRUMENT_MAP.get(symbol)


def get_atm_options(index_name, spot_price):
    global NFO_INSTRUMENTS
    if not kite or not kite.access_token: return None
    
    try:
        if NFO_INSTRUMENTS is None:
            print(f"DEBUG: Fetching NFO Instruments for {index_name}...")
            NFO_INSTRUMENTS = pd.DataFrame(kite.instruments("NFO"))
            
        # Filter for index (e.g., NIFTY or BANKNIFTY)
        df_nfo = NFO_INSTRUMENTS[NFO_INSTRUMENTS['name'] == index_name]
        if df_nfo.empty: return None
        
        # Get nearest expiry
        df_nfo['expiry'] = pd.to_datetime(df_nfo['expiry'])
        nearest_expiry = df_nfo['expiry'].min()
        df_expiry = df_nfo[df_nfo['expiry'] == nearest_expiry]
        
        # Find ATM Strike
        strike_step = 50 if index_name == "NIFTY" else 100
        atm_strike = round(spot_price / strike_step) * strike_step
        
        # Get CE and PE symbols
        ce_row = df_expiry[(df_expiry['strike'] == atm_strike) & (df_expiry['instrument_type'] == 'CE')]
        pe_row = df_expiry[(df_expiry['strike'] == atm_strike) & (df_expiry['instrument_type'] == 'PE')]
        
        if ce_row.empty or pe_row.empty: return None
        
        return {
            'expiry': nearest_expiry.strftime('%d %b'),
            'strike': atm_strike,
            'ce_symbol': f"NFO:{ce_row.iloc[0]['tradingsymbol']}",
            'pe_symbol': f"NFO:{pe_row.iloc[0]['tradingsymbol']}"
        }
    except Exception as e:
        print(f"Error finding ATM options: {e}")
        return None


def detect_spurt(sym, current_total_vol):
    global VOL_SAMPLES
    now = datetime.datetime.now()
    
    if sym not in VOL_SAMPLES:
        VOL_SAMPLES[sym] = []
    
    # Add new sample
    VOL_SAMPLES[sym].append((now, current_total_vol))
    
    # Keep last 6 samples (approx 6-10 mins of history)
    if len(VOL_SAMPLES[sym]) > 6:
        VOL_SAMPLES[sym].pop(0)
        
    if len(VOL_SAMPLES[sym]) < 3:
        return 1.0 # Need at least 3 samples to calculate trend
    
    # Calculate Deltas
    deltas = []
    for i in range(1, len(VOL_SAMPLES[sym])):
        d_vol = VOL_SAMPLES[sym][i][1] - VOL_SAMPLES[sym][i-1][1]
        d_time = (VOL_SAMPLES[sym][i][0] - VOL_SAMPLES[sym][i-1][0]).total_seconds() / 60
        if d_time > 0:
            deltas.append(d_vol / d_time) # Volume per minute
            
    if not deltas: return 1.0
    
    current_velocity = deltas[-1]
    avg_velocity = sum(deltas[:-1]) / len(deltas[:-1]) if len(deltas) > 1 else deltas[0]
    
    if avg_velocity == 0: return 1.0
    return current_velocity / avg_velocity


def init_volume_baseline(symbols):
    global VOL_BASELINE
    print(f"DEBUG: Initializing Volume Baselines for {len(symbols)} symbols...")
    try:
        # We only do this once to avoid lag
        for i in range(0, len(symbols), 20):
            batch = symbols[i:i+20]
            tickers = [f"{s}.NS" for s in batch]
            data = yf.download(tickers, period="20d", interval="1d", progress=False, group_by='ticker')
            for s in batch:
                yf_s = f"{s}.NS"
                if yf_s in data.columns.levels[0]:
                    vols = data[yf_s]['Volume'].dropna()
                    if not vols.empty:
                        VOL_BASELINE[s] = vols.mean()
        print("DEBUG: Volume Baselines Initialized.")
    except Exception as e:
        print(f"Error initializing volume: {e}")

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, 'r') as f:
                return json.load(f)
        except: pass
    # Default top 20
    return ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'BHARTIARTL', 'SBIN', 'LICI', 'ITC', 'HINDUNILVR', 'LT', 'BAJFINANCE', 'ADANIENT', 'SUNPHARMA', 'MARUTI', 'TATASTEEL', 'KOTAKBANK', 'TITAN', 'AXISBANK', 'ADANIPORTS']

def save_watchlist(wl):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(wl, f)


def get_live_news(symbol=None):
    """
    Fetch live news from curated Indian financial sources.
    Sources: Moneycontrol, ET Markets, Livemint, Business Standard, NDTV Profit, Bloomberg Quint, Pulse by Zerodha
    """
    news_items = []
    
    # Curated RSS feeds from premium Indian financial sources
    feeds = [
        # Tier 1: Real-time market updates & Volatility Alerts
        {
            "url": "https://www.moneycontrol.com/rss/latestnews.xml",
            "name": "Moneycontrol",
            "priority": 1
        },
        {
            "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "name": "ET Markets",
            "priority": 1
        },
        {
            "url": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/stock-market.xml",
            "name": "CNBC TV18",
            "priority": 1
        },
        
        # Tier 2: Detailed stock analysis & Sector news
        {
            "url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
            "name": "ET Stocks",
            "priority": 2
        },
        {
            "url": "https://www.livemint.com/rss/markets",
            "name": "Livemint",
            "priority": 2
        },
        {
            "url": "https://www.business-standard.com/rss/markets-106.rss",
            "name": "Business Std",
            "priority": 2
        },

        # Tier 3: Broad economy, global ties & Hindi/Regional coverage coverage
        {
            "url": "https://in.investing.com/rss/news.rss",
            "name": "Investing.com",
            "priority": 3
        },
        {
            "url": "https://zeenews.india.com/rss/business.xml",
            "name": "Zee Business",
            "priority": 3
        },
        {
            "url": "https://feeds.feedburner.com/ndtvprofit-latest",
            "name": "NDTV Profit",
            "priority": 3
        },
        {
            "url": "https://www.bloombergquint.com/feed",
            "name": "BloombergQuint",
            "priority": 3
        }
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/xml, text/xml, */*'
    }
    
    # Sort by priority
    feeds.sort(key=lambda x: x['priority'])
    
    seen_titles = set()
    
    for feed_info in feeds:
        try:
            r = requests.get(feed_info['url'], headers=headers, timeout=5)
            if r.status_code == 200:
                # Extract titles from RSS items
                titles = re.findall(r'<item>.*?<title>(.*?)</title>', r.text, re.DOTALL | re.IGNORECASE)
                
                for t in titles:
                    # Clean CDATA and HTML entities
                    clean_t = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', t)
                    clean_t = re.sub(r'<[^>]+>', '', clean_t)  # Remove any HTML tags
                    clean_t = clean_t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                    clean_t = clean_t.replace("&quot;", '"').replace("&#39;", "'").strip()
                    
                    # Skip duplicates and very short titles
                    if not clean_t or len(clean_t) < 20 or clean_t in seen_titles:
                        continue
                    
                    # Symbol-specific filtering
                    if symbol:
                        keywords = [symbol.lower(), symbol[:4].lower()]  # Match full and partial
                        if any(kw in clean_t.lower() for kw in keywords):
                            news_items.append(f"[{feed_info['name']}] {clean_t}")
                            seen_titles.add(clean_t)
                            if len(news_items) >= 10:  # Cap at 10 symbol-specific news
                                break
                    else:
                        # General market news
                        news_items.append(f"[{feed_info['name']}] {clean_t}")
                        seen_titles.add(clean_t)
                        if len(news_items) >= 20:  # Cap at 20 general news
                            break
                            
        except Exception as e:
            print(f"News fetch error for {feed_info['name']}: {e}")
            continue
        
        # Stop if we have enough news
        if symbol and len(news_items) >= 10:
            break
        elif not symbol and len(news_items) >= 20:
            break
    
    # Fallback messages
    if not news_items:
        if symbol:
            news_items = [
                f"[System] Monitoring {symbol} for breaking news...",
                "[Market] NSE/BSE data streams active. No specific alerts at this time."
            ]
        else:
            news_items = [
                "[Market] NSE: Tracking derivative series with high open interest",
                "[Update] Live market data streaming from multiple sources"
            ]
    
    return news_items[:10] if symbol else news_items[:20]


def get_historical_news(symbol):
    """
    Fetch stock-specific news history (last 30 days).
    Strategy:
    1. Try yfinance .news first (High quality, structured)
    2. detailed Google News RSS Search if yfinance yields low results (< 5 items)
    """
    if not symbol: return []
    news_history = []
    seen_titles = set()
    current_ts = datetime.datetime.now().timestamp()
    day_30_ts = current_ts - (30 * 24 * 3600)

    # 1. YFINANCE FETCH
    try:
        ticker_sym = f"{symbol}.NS"
        if symbol in ['NIFTY', 'BANKNIFTY']: ticker_sym = "^NSEI" if symbol == 'NIFTY' else "^NSEBANK"
        elif symbol == 'SENSEX': ticker_sym = "^BSESN"
        
        t = yf.Ticker(ticker_sym)
        raw_news = t.news
        
        if raw_news:
            for item in raw_news:
                pub_time = item.get('providerPublishTime', 0)
                title = item.get('title', '')
                publisher = item.get('publisher', 'Yahoo')
                
                if pub_time >= day_30_ts and title not in seen_titles:
                    date_str = datetime.datetime.fromtimestamp(pub_time).strftime('%d %b')
                    news_history.append({'title': title, 'date': date_str, 'source': publisher})
                    seen_titles.add(title)
    except Exception as e:
        print(f"YF News Error: {e}")

    # 2. GOOGLE NEWS RSS FALLBACK (If news is sparse)
    if len(news_history) < 5:
        try:
            # Query: Symbol + "stock news india" to get relevant local news
            query = f"{symbol}+stock+news+india"
            gnews_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            r = requests.get(gnews_url, headers=headers, timeout=5)
            
            if r.status_code == 200:
                # Extract items
                items = re.findall(r'<item>.*?</item>', r.text, re.DOTALL)
                for item in items:
                    title_match = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
                    pubdate_match = re.search(r'<pubDate>(.*?)</pubDate>', item, re.DOTALL)
                    
                    if title_match and pubdate_match:
                        title = title_match.group(1).replace(" - Google News", "")
                        # Simple cleanup
                        title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title)
                        title = title.replace("&amp;", "&").replace("&#39;", "'")
                        
                        # Parse Date (RFC 822 format usually: Tue, 03 Jun 2025 13:00:00 GMT)
                        pub_str = pubdate_match.group(1)
                        try:
                            # Python 3.7+ can parse most RSS date formats automatically or manually
                            # We'll try a manual short parse for speed or fallback to today
                            dt_obj = datetime.datetime.strptime(pub_str[:16], '%a, %d %b %Y')
                        except:
                            dt_obj = datetime.datetime.now() # Fallback
                        
                        if dt_obj.timestamp() >= day_30_ts and title not in seen_titles:
                             news_history.append({
                                'title': title,
                                'date': dt_obj.strftime('%d %b'),
                                'source': 'Google News'
                            })
                             seen_titles.add(title)
                    
                    if len(news_history) >= 15: break
        except Exception as e:
            print(f"GNews Error: {e}")

    return news_history[:15] # Return top 15 most relevant



def find_levels(series, threshold=0.0025):
    """Efficiently find historical price clusters/levels for a given series."""
    prices = sorted(series.dropna().round(2).unique())
    if not prices: return []
    
    clusters = []
    current_cluster = [prices[0]]
    for i in range(1, len(prices)):
        if (prices[i] - prices[i-1]) / prices[i-1] <= threshold:
            current_cluster.append(prices[i])
        else:
            clusters.append(round(sum(current_cluster) / len(current_cluster), 2))
            current_cluster = [prices[i]]
    clusters.append(round(sum(current_cluster) / len(current_cluster), 2))
    return clusters


def get_stock_data_cloud(symbol, live_price=None):
    """Fetch historical prices from Supabase cloud and generate ladder."""
    try:
        # Fetch last 3000 days of data from Supabase (Increased from 500 to catch deep historical supports)
        response = supabase.table("market_data").select("price").eq("symbol", symbol).order("date", desc=True).limit(3000).execute()
        prices_raw = [round(float(r['price']), 2) for r in response.data]
        if not prices_raw: return None
        
        # STICK TO 2 DECIMALS
        prices = sorted(list(set(prices_raw)))
        
        if live_price:
            latest_price = round(live_price, 2)
        else:
            latest_price = prices_raw[0] # Most recent historical
        
        # ---------------------------------------------------------
        # SEGREGATED LOGIC:
        # 1. Trading Execution (VMF/Ladder): Uses PURE RAW Historical Closing Prices (HCP)
        # 2. Scanning/filtering (Runway): Uses CLUSTERS (Macro Structure)
        # ---------------------------------------------------------

        # A. TRADING DATA (Micro View - Actual Support/Resistance)
        # Deduplication Logic: Ensure levels are at least 0.15% or 1 unit apart 
        # to avoid showing 5 levels from a single tight consolidation (e.g. 1459.1, 1459.2...)
        
        def get_distinct_raw(candidates, direction='up', limit=5, min_gap_pct=0.0015):
            selected = []
            distinct_pool = sorted(list(set(candidates)), reverse=(direction=='down'))
            
            if not distinct_pool: return []
            
            last_p = distinct_pool[0]
            selected.append(last_p)
            
            for p in distinct_pool[1:]:
                if len(selected) >= limit: break
                gap = abs(p - last_p)
                # Use tick size 0.05 instead of 1.0 to support low-priced stocks
                if gap >= 0.05 and (gap / last_p) > min_gap_pct:
                    selected.append(p)
                    last_p = p
            return selected

        # Raw pools
        raw_above = [p for p in prices if p > latest_price]
        raw_below = [p for p in prices if p < latest_price]
        
        trading_above = sorted(get_distinct_raw(raw_above, 'up'))
        trading_below = sorted(get_distinct_raw(raw_below, 'down'), reverse=True) # Sort desc for standard viewing
        
        trading_levels_raw = sorted(trading_below + trading_above)
        
        # Prepare VMF Data (Strictly Raw)
        vmf_levels = []
        for p in trading_levels_raw:
             vmf_levels.append({
                'price': p,
                'strength': 1,
                'min': p, 'max': p,
                'is_pivot': False, 'is_raw': True
             })
        
        # Add LTP to VMF
        lower_vmf = sorted([c for c in vmf_levels if c['price'] < latest_price], key=lambda x: x['price'], reverse=True)
        higher_vmf = sorted([c for c in vmf_levels if c['price'] > latest_price], key=lambda x: x['price']) # Ascending
        
        # Combined VMF render list (High -> Low for Vertical Chart)
        render_levels = sorted(higher_vmf, key=lambda x: x['price'], reverse=True) + [{'price': latest_price, 'is_ltp': True}] + lower_vmf

        # Prepare Ladder Table (Strict 5+5)
        # Pad Low to length 5
        padded_low = [None] * (5 - len(lower_vmf)) + [c['price'] for c in reversed(lower_vmf)]
        # Pad High to length 5
        padded_high = [c['price'] for c in higher_vmf] + [None] * (5 - len(higher_vmf))
        
        ladder_prices = padded_low + [latest_price] + padded_high


        # B. SCANNING DATA (Macro View - Runway Potential)
        # Use Clustering Logic on the FULL relevant dataset to determine "Room to Move"
        # We use a larger pool for clustering to find major structural levels
        scan_above = sorted([p for p in prices if p > latest_price])[:50]
        scan_below = sorted([p for p in prices if p < latest_price], reverse=True)[:50]
        scan_pool = sorted(list(set(scan_below + scan_above)))
        
        clusters = []
        abs_min_gap = 10.0
        
        if scan_pool:
            current_cluster = [scan_pool[0]]
            for i in range(1, len(scan_pool)):
                price_diff = scan_pool[i] - scan_pool[i-1]
                pct_diff = price_diff / scan_pool[i-1]
                if pct_diff <= 0.0025 and price_diff < abs_min_gap:
                    current_cluster.append(scan_pool[i])
                else:
                    clusters.append(round(sum(current_cluster) / len(current_cluster), 2))
                    current_cluster = [scan_pool[i]]
            clusters.append(round(sum(current_cluster) / len(current_cluster), 2))

        # Calculate Runway based on MACRO CLUSTERS (Scanning Logic)
        # Find where LTP sits in these clusters
        c_above = sorted([c for c in clusters if c > latest_price])
        c_below = sorted([c for c in clusters if c < latest_price], reverse=True)
        
        # Calculate cluster gaps
        upside_runway = 0
        downside_runway = 0
        
        if len(c_above) >= 3:
            gaps_up = []
            levels_up = [latest_price] + c_above[:5]
            for i in range(len(levels_up)-1):
                gaps_up.append( ((levels_up[i+1] - levels_up[i])/levels_up[i])*100 )
            # Runway is sum of first 3 distinct cluster gaps (Headroom)
            upside_runway = sum(gaps_up[:3])

        if len(c_below) >= 3:
            gaps_down = []
            levels_down = [latest_price] + c_below[:5]
            for i in range(len(levels_down)-1):
                gaps_down.append( abs(((levels_down[i+1] - levels_down[i])/levels_down[i])*100) )
            downside_runway = sum(gaps_down[:3])

        # Ladder Gaps (For Table visual only)
        row_gaps = []
        for i in range(len(ladder_prices) - 1):
            curr, nxt = ladder_prices[i], ladder_prices[i+1]
            gap_pct = 0
            if curr and nxt: gap_pct = ((nxt - curr) / curr) * 100
            row_gaps.append(gap_pct)

        # ZOOM SCALE: Only use the range of the TRADING levels
        rendered_price_points = [c['price'] for c in render_levels]
        vmf_min = min(rendered_price_points)
        vmf_max = max(rendered_price_points)

        return {
            'symbol': symbol, 'prices': ladder_prices, 'clusters': render_levels, 
            'latest': latest_price, 'gaps': row_gaps, 'min_p': vmf_min, 'max_p': vmf_max,
            'upside_runway': round(upside_runway, 2), 'downside_runway': round(downside_runway, 2)
        }
    except Exception as e:
        print(f"Cloud fetch error for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Error clustering stock data for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Error getting stock data for {symbol}: {e}")
        return None


def get_live_prices_batch(symbols):
    live_prices = {}
    if not symbols:
        return live_prices
        
    try:
        # Prio 1: Zerodha/Kite
        if kite and kite.access_token:
            instruments = [f"NSE:{s}" for s in symbols]
            # Handle special index names for Zerodha
            idx_map = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK"}
            final_instruments = [idx_map.get(s, f"NSE:{s}") for s in symbols]
            
            ltp_data = kite.ltp(final_instruments)
            for k, v in ltp_data.items():
                sym = k.replace("NSE:", "").replace("NIFTY 50", "NIFTY").replace("NIFTY BANK", "BANKNIFTY")
                live_prices[sym] = v['last_price']
            return live_prices

        # Prio 2: yfinance (Fallback)
        ticker_map = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'CNXFINANCE': 'NIFTY_FIN_SERVICE.NS'
        }
        yf_tickers = [ticker_map.get(sym, f"{sym}.NS") for sym in symbols]
        
        # Download batch data
        # period="1d" gives today's data. interval="1m" gives minute data.
        # We just need the latest close.
        data = yf.download(yf_tickers, period="1d", interval="1m", group_by='ticker', threads=True)
        
        if not data.empty:
            print("DEBUG: Data received from yfinance")
            # Check if we have multiple tickers or just one
            if len(yf_tickers) > 1:
                for sym, yf_sym in zip(symbols, yf_tickers):
                    try:
                        # yfinance structure for multiple tickers: data[Ticker]['Close']
                        # But sometimes if data is missing it might not be there.
                        if yf_sym in data.columns.levels[0]:
                            closes = data[yf_sym]['Close'].dropna()
                            if not closes.empty:
                                live_prices[sym] = closes.iloc[-1]
                                print(f"DEBUG: Got price for {sym}: {closes.iloc[-1]}")
                            else:
                                print(f"DEBUG: No close data for {sym}")
                        else:
                            print(f"DEBUG: {yf_sym} not in columns")
                    except Exception as e:
                        print(f"Error extracting data for {sym}: {e}")
            else:
                # Single ticker structure: data['Close']
                sym = symbols[0]
                closes = data['Close'].dropna()
                if not closes.empty:
                    live_prices[sym] = closes.iloc[-1]
                    print(f"DEBUG: Got price for {sym}: {closes.iloc[-1]}")
        else:
            print("DEBUG: No data received from yfinance")
                    
    except Exception as e:
        print(f"Error fetching batch live prices: {e}")
        
    return live_prices

@app.route('/login')
def login():
    global kite
    request_token = request.args.get("request_token")
    
    if request_token:
        try:
            data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
            access_token = data["access_token"]
            kite.set_access_token(access_token)
            # You can also save this to .env here if you wish for persistence
            return redirect("/")
        except Exception as e:
            return f"Login failed: {e}", 400
            
    if not KITE_API_KEY or not KITE_API_SECRET:
        return "Please set KITE_API_KEY and KITE_API_SECRET in .env file", 400
    
    # Redirect to Kite login
    return redirect(kite.login_url())

@app.route('/', methods=['GET', 'POST'])
def index():
    display_symbols = []
    index_data = {}
    top_bullish = []
    top_bearish = []
    
    # Load Data
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, low_memory=False)
            df.columns = [col.strip() for col in df.columns]
            index_symbols = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'CNXFINANCE']
            symbol_columns = sorted([col for col in df.columns if col.lower() != 'date' and not col.lower().startswith('unnamed') and len(col.strip()) > 0])
            
            for col in symbol_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            for index_symbol in index_symbols:
                if index_symbol in df.columns:
                    latest_val = df[index_symbol].dropna().iloc[-1]
                    index_data[index_symbol] = round(latest_val, 2)

            display_symbols = symbol_columns
            
            # Opportunity Finder for Landing Page
            opportunities = []
            for symbol in display_symbols:
                if symbol in index_symbols: continue
                try:
                    sym_prices = df[symbol].dropna().round(2).unique()
                    if len(sym_prices) < 2: continue
                    latest_val = df[symbol].dropna().iloc[-1]
                    higher = sorted([p for p in sym_prices if p > latest_val])
                    p_plus_1 = higher[0] if higher else None
                    lower = sorted([p for p in sym_prices if p < latest_val], reverse=True)
                    p_minus_1 = lower[0] if lower else None
                    
                    gap_up = round(((p_plus_1 - latest_val) / latest_val * 100), 2) if p_plus_1 else 0
                    gap_down = round(((latest_val - p_minus_1) / latest_val * 100), 2) if p_minus_1 else 0
                    
                    opportunities.append({'symbol': symbol, 'gap_up': gap_up, 'gap_down': gap_down})
                except: continue
            
            top_bullish = sorted([o for o in opportunities if o['gap_up'] > 0], key=lambda x: x['gap_up'], reverse=True)[:10]
            top_bearish = sorted([o for o in opportunities if o['gap_down'] > 0], key=lambda x: x['gap_down'], reverse=True)[:10]
    except Exception as e:
        print(f"Index data loading error: {e}")

    # Fetch Live News & Watchlist
    news_list = get_live_news()
    base_watchlist = load_watchlist()
    
    # Enrich Watchlist with Live Data
    wl_prices = get_live_prices_batch(base_watchlist)
    watchlist_details = []
    
    try:
        if os.path.exists(CSV_FILE):
            # We already have df from the previous block if successful, else reload
            if 'df' not in locals():
                df = pd.read_csv(CSV_FILE, low_memory=False)
                df.columns = [col.strip() for col in df.columns]
                for col in base_watchlist:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

            for i, sym in enumerate(base_watchlist):
                live_p = wl_prices.get(sym)
                ref_p = None
                if sym in df.columns:
                    ref_p = df[sym].dropna().iloc[-1]
                
                price = round(live_p, 2) if live_p else (round(ref_p, 2) if ref_p else 0)
                change = 0
                if live_p and ref_p:
                    change = round(((live_p - ref_p) / ref_p) * 100, 2)
                
                watchlist_details.append({
                    'symbol': sym,
                    'price': price,
                    'change': change,
                    'orig_idx': i
                })
        
        # Sort by change descending (High gainers first)
        watchlist_details = sorted(watchlist_details, key=lambda x: x['change'], reverse=True)

    except Exception as e:
        print(f"Error enriching watchlist: {e}")
        # Fallback to simple list if error
        watchlist_details = [{'symbol': s, 'price': 0, 'change': 0, 'orig_idx': i} for i, s in enumerate(base_watchlist)]

    # Calculate max intensity for color scaling
    all_changes = [abs(d['change']) for d in watchlist_details]
    max_intensity = max(all_changes) if all_changes and max(all_changes) > 0 else 1

    # ============= WATCHTOWER SCANNER LOGIC (ALL 223 STOCKS) =============
    watchtower_alerts = []
    try:
        if os.path.exists(CSV_FILE):
            all_scan_symbols = [col for col in df.columns if col not in index_symbols and col.lower() != 'date' and not col.lower().startswith('unnamed')]
            
            # Initialize volume baseline once
            if not VOL_BASELINE and kite:
                init_volume_baseline(all_scan_symbols[:50]) # Limiting to 50 for speed in initialization

            # Fetch Live Quotes from Zerodha if possible
            scan_quotes = {}
            if kite and kite.access_token:
                try:
                    # Batch fetch for scanner (limit to top 100 for performance)
                    batch_instruments = [f"NSE:{s}" for s in all_scan_symbols[:100]]
                    scan_quotes = kite.quote(batch_instruments)
                except: pass

            for sym in all_scan_symbols:
                # Get live price and volume
                instr = f"NSE:{sym}"
                quote = scan_quotes.get(instr, {})
                
                lp = quote.get('last_price') or wl_prices.get(sym) or (df[sym].dropna().iloc[-1] if not df[sym].dropna().empty else None)
                if lp is None: continue
                
                vol_now = quote.get('volume', 0)
                change_pct = quote.get('change', 0)
                
                # Check for Volume Hunter (Intraday Spurt)
                spurt_ratio = detect_spurt(sym, vol_now) if vol_now > 0 else 1.0
                
                # Check for Daily Abnormal Volume (Historical)
                avg_vol = VOL_BASELINE.get(sym)
                vol_ratio = 1.0
                if avg_vol and vol_now > 0:
                    now_in = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
                    m_start = now_in.replace(hour=9, minute=15)
                    elapsed = max(1, (now_in - m_start).total_seconds() / 60)
                    expected_pct = elapsed / 375
                    vol_ratio = (vol_now / avg_vol) / expected_pct if expected_pct > 0 else 1.0

                if spurt_ratio > 3.5: # 3.5x faster than last 10 mins
                    watchtower_alerts.append({
                        'sym': sym, 'p': lp, 'lvl': 'Sudden Spurt', 
                        'type': 'Volume Hunter', 'target': 'Block Trade DETECTED', 
                        'severity': 'high', 'dist': round(spurt_ratio, 1)
                    })
                elif vol_ratio > 2.2: # Daily outlier
                    watchtower_alerts.append({
                        'sym': sym, 'p': lp, 'lvl': 'Volume Surge', 
                        'type': 'Volume Hunter', 'target': 'High Day Conviction', 
                        'severity': 'med', 'dist': round(vol_ratio, 1)
                    })

                # Dynamic Levels for this stock
                levels = find_levels(df[sym])
                if not levels: continue
                
                # Find nearest levels
                s1_list = [lvl for lvl in levels if lvl < lp]
                r1_list = [lvl for lvl in levels if lvl > lp]
                
                s1 = s1_list[-1] if s1_list else None
                r1 = r1_list[0] if r1_list else None
                s2 = s1_list[-2] if len(s1_list) >= 2 else None
                r2 = r1_list[1] if len(r1_list) >= 2 else None

                # Proximity (0.35% threshold)
                dist_s1 = abs((lp - s1) / lp * 100) if s1 else 100
                dist_r1 = abs((r1 - lp) / lp * 100) if r1 else 100
                
                # Signal Logic
                if dist_r1 < 0.35:
                    watchtower_alerts.append({'sym': sym, 'p': lp, 'lvl': r1, 'type': 'Approaching Resistance', 'target': r2 or 'Blue Sky', 'severity': 'high', 'dist': round(dist_r1, 2)})
                elif dist_s1 < 0.35:
                    watchtower_alerts.append({'sym': sym, 'p': lp, 'lvl': s1, 'type': 'Testing Support', 'target': s2 or 'Floorless', 'severity': 'high', 'dist': round(dist_s1, 2)})
                
                # Vacuum Detect (If distance between S1 and R1 is > 2.5%)
                if s1 and r1:
                    gap = (r1 - s1) / s1 * 100
                    if gap > 3.0:
                        pos = (lp - s1) / (r1 - s1)
                        if 0.3 < pos < 0.7:
                             watchtower_alerts.append({'sym': sym, 'p': lp, 'lvl': f"{s1} - {r1}", 'type': 'Vacuum Hunt', 'target': r1, 'severity': 'med', 'dist': round(gap, 2)})

    except Exception as e:
        print(f"Watchtower Scan Error: {e}")

    # Sort alerts by severity and then proximity
    watchtower_alerts = sorted(watchtower_alerts, key=lambda x: (x['severity'] == 'high', -x['dist']), reverse=True)[:15]

    return render_template_string(LANDING_TEMPLATE, 
                                  symbols=display_symbols, 
                                  index_data=index_data, 
                                  top_bullish=top_bullish, 
                                  top_bearish=top_bearish,
                                  news_list=news_list,
                                  watchlist=watchlist_details,
                                  max_intensity=max_intensity,
                                  watchtower=watchtower_alerts,
                                  kite_authenticated=True if (kite and kite.access_token) else False)

@app.route('/update_watchlist', methods=['POST'])
def update_watchlist():
    data = request.json
    index = data.get('index')
    symbol = data.get('symbol', '').strip().upper()
    
    if index is not None and symbol:
        wl = load_watchlist()
        if 0 <= index < len(wl):
            wl[index] = symbol
            save_watchlist(wl)
            return {"status": "success"}
    return {"status": "error"}, 400

@app.route('/analyze', methods=['POST'])
def analyze_redirect():
    symbol = request.form.get('symbol_search', '').strip().upper()
    if not symbol:
        symbol = request.form.get('symbols', '')
    if symbol:
        return f"<script>window.location.href='/analysis/{symbol}';</script>"
    return "<script>window.location.href='/';</script>"

@app.route('/analysis/<symbol>')
def analysis(symbol):
    try:
        # 0. Fuzzy match symbol if needed (using CSV as master list for now)
        if os.path.exists(CSV_FILE):
            df_temp = pd.read_csv(CSV_FILE, nrows=0) # Just headers
            df_temp.columns = [c.strip() for c in df_temp.columns]
            if symbol not in df_temp.columns:
                matches = [c for c in df_temp.columns if symbol[:3] in c]
                if matches: symbol = matches[0]

        # 1. Fetch live price
        lp_map = get_live_prices_batch([symbol])
        latest_price = lp_map.get(symbol)
        
        # 2. Try Supabase Cloud first
        s_data = None
        if supabase:
            s_data = get_stock_data_cloud(symbol, latest_price)
            
        # 3. Fallback to Local CSV if Cloud fails or is missing
        if not s_data and os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            df.columns = [c.strip() for c in df.columns]
            if symbol in df.columns:
                from price_ladder_app import find_levels # local ref
                # We'll use a simplified local generator
                s_data = get_stock_data_local_fallback(symbol, df, latest_price)

        if s_data:
            # History Fetch
            history_cloud = []
            if supabase:
                try:
                    resp = supabase.table("market_data").select("date, price").eq("symbol", symbol).order("date", desc=True).limit(50).execute()
                    history_cloud = [{"Date": r['date'], symbol: float(r['price'])} for r in reversed(resp.data)]
                except: pass
            
            if not history_cloud and os.path.exists(CSV_FILE):
                df_hist = pd.read_csv(CSV_FILE)
                df_hist.columns = [c.strip() for c in df_hist.columns]
                history_cloud = df_hist[['Date', symbol]].dropna().tail(50).to_dict('records')

            # Add comparative analytics
            gaps_only = [abs(g) for g in s_data['gaps'] if g is not None]
            s_data['row_max_gap'] = max(gaps_only) if gaps_only else 0.001
            
            # Fetch Intraday Data
            intraday_data = []
            try:
                ticker = f"{symbol}.NS"
                ticker_map = {'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'SENSEX': '^BSESN'}
                yf_ticker = ticker_map.get(symbol, ticker)
                idf = yf.download(yf_ticker, period="1d", interval="5m")
                if not idf.empty:
                    for idx, row_data in idf.iterrows():
                        intraday_data.append({
                            'time': int(idx.timestamp()), 'open': float(row_data['Open']),
                            'high': float(row_data['High']), 'low': float(row_data['Low']), 'close': float(row_data['Close'])
                        })
            except Exception: pass

            # Market Depth & Options
            market_depth = None
            options_data = None
            if kite and kite.access_token:
                try:
                    instr = f"NSE:{symbol}"
                    if symbol == "NIFTY": instr = "NSE:NIFTY 50"
                    elif symbol == "BANKNIFTY": instr = "NSE:NIFTY BANK"
                    quote = kite.quote(instr)
                    if instr in quote:
                        market_depth = quote[instr]['depth']
                        last_price = quote[instr]['last_price']
                        if symbol in ["NIFTY", "BANKNIFTY"]:
                            opt_info = get_atm_options(symbol, last_price)
                            if opt_info:
                                opt_quotes = kite.quote([opt_info['ce_symbol'], opt_info['pe_symbol']])
                                options_data = {'strike': opt_info['strike'], 'expiry': opt_info['expiry'], 'ce': opt_quotes.get(opt_info['ce_symbol']), 'pe': opt_quotes.get(opt_info['pe_symbol'])}
                except Exception: pass

            kite_token = get_instrument_token(symbol if symbol not in ["NIFTY", "BANKNIFTY"] else (f"NIFTY 50" if symbol == "NIFTY" else "NIFTY BANK"))
            news_list = get_live_news(symbol)
            hist_news = get_historical_news(symbol)
            
            return render_template_string(ANALYSIS_TEMPLATE, 
                                          row=s_data, 
                                          history=history_cloud, 
                                          intraday=intraday_data, 
                                          news_list=news_list,
                                          hist_news=hist_news,
                                          market_depth=market_depth,
                                          options_data=options_data,
                                          kite_token=kite_token,
                                          kite_authenticated=True if (kite and kite.access_token) else False)
        return f"Symbol {symbol} not found in Cloud or Local Database.", 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e), 500

def get_stock_data_local_fallback(symbol, df, live_price=None):
    """Local CSV fallback for get_stock_data."""
    try:
        # Strict 2 Decimal Rounding
        prices = sorted(df[symbol].dropna().round(2).unique())
        latest_price = round(live_price, 2) if live_price else round(df[symbol].dropna().iloc[-1], 2)
        
        # ---------------------------------------------------------
        # SEGREGATED LOGIC (Matching Cloud Implementation)
        # 1. Trading Execution (Micro View): Strict 5+5 Raw Levels
        # 2. Scanning (Macro View): Cluster-based Runway
        # ---------------------------------------------------------

        # A. TRADING DATA (Micro View)
        trading_above = sorted([p for p in prices if p > latest_price])[:5]
        trading_below = sorted([p for p in prices if p < latest_price], reverse=True)[:5]
        trading_levels_raw = sorted(trading_below + trading_above)
        
        # Prepare VMF Data
        vmf_levels = []
        for p in trading_levels_raw:
             vmf_levels.append({
                'price': p,
                'strength': 1,
                'min': p, 'max': p,
                'is_pivot': False, 'is_raw': True
             })
        
        lower_vmf = sorted([c for c in vmf_levels if c['price'] < latest_price], key=lambda x: x['price'], reverse=True)
        higher_vmf = sorted([c for c in vmf_levels if c['price'] > latest_price], key=lambda x: x['price']) # Ascending
        
        # Combined VMF render list
        render_levels = sorted(higher_vmf, key=lambda x: x['price'], reverse=True) + [{'price': latest_price, 'is_ltp': True}] + lower_vmf

        # Prepare Ladder Table (Strict 5+5)
        padded_low = [None] * (5 - len(lower_vmf)) + [c['price'] for c in reversed(lower_vmf)]
        padded_high = [c['price'] for c in higher_vmf] + [None] * (5 - len(higher_vmf))
        ladder_prices = padded_low + [latest_price] + padded_high

        # Ladder Gaps (For Table visual only)
        row_gaps = []
        for i in range(len(ladder_prices) - 1):
            curr, nxt = ladder_prices[i], ladder_prices[i+1]
            gap_pct = 0
            if curr and nxt: gap_pct = ((nxt - curr) / curr) * 100
            row_gaps.append(gap_pct)


        # B. SCANNING DATA (Macro View - Runway Potential)
        # Use Clustering Logic on larger pool
        scan_above = sorted([p for p in prices if p > latest_price])[:40]
        scan_below = sorted([p for p in prices if p < latest_price], reverse=True)[:40]
        scan_pool = sorted(list(set(scan_below + scan_above)))
        
        clusters = []
        abs_min_gap = 10.0
        
        if scan_pool:
            current_cluster = [scan_pool[0]]
            for i in range(1, len(scan_pool)):
                price_diff = scan_pool[i] - scan_pool[i-1]
                pct_diff = price_diff / scan_pool[i-1]
                if pct_diff <= 0.0025 and price_diff < abs_min_gap:
                    current_cluster.append(scan_pool[i])
                else:
                    clusters.append(round(sum(current_cluster) / len(current_cluster), 2))
                    current_cluster = [scan_pool[i]]
            clusters.append(round(sum(current_cluster) / len(current_cluster), 2))

        # Calculate Runway based on MACRO CLUSTERS
        c_above = sorted([c for c in clusters if c > latest_price])
        c_below = sorted([c for c in clusters if c < latest_price], reverse=True)
        
        upside_runway = 0
        downside_runway = 0
        
        if len(c_above) >= 3:
            gaps_up = []
            levels_up = [latest_price] + c_above[:5]
            for i in range(len(levels_up)-1):
                gaps_up.append( ((levels_up[i+1] - levels_up[i])/levels_up[i])*100 )
            upside_runway = sum(gaps_up[:3])

        if len(c_below) >= 3:
            gaps_down = []
            levels_down = [latest_price] + c_below[:5]
            for i in range(len(levels_down)-1):
                gaps_down.append( abs(((levels_down[i+1] - levels_down[i])/levels_down[i])*100) )
            downside_runway = sum(gaps_down[:3])
            
        # ZOOM SCALE
        rendered_price_points = [c['price'] for c in render_levels]
        vmf_min = min(rendered_price_points) if rendered_price_points else latest_price * 0.95
        vmf_max = max(rendered_price_points) if rendered_price_points else latest_price * 1.05

        return {
            'symbol': symbol, 'prices': ladder_prices, 'clusters': render_levels, 
            'latest': latest_price, 'gaps': row_gaps, 'min_p': vmf_min, 'max_p': vmf_max,
            'upside_runway': round(upside_runway, 2), 'downside_runway': round(downside_runway, 2)
        }
    except Exception as e:
        print(f"Local Fallback Error: {e}")
        return None


LANDING_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Price Ladder Pro - Market Dashboard</title>
    <style>
        :root {
            --bg-color: #0c0d10;
            --card-bg: #131722;
            --text-color: #d1d4dc;
            --accent-color: #bb86fc;
            --up-color: #03dac6;
            --down-color: #cf6679;
            --border-color: #2a2e39;
        }
        body { font-family: 'Segoe UI', sans-serif; background: var(--bg-color); color: var(--text-color); margin: 0; }
        
        /* Ticker Styling */
        .ticker-wrap { width: 100%; overflow: hidden; background: #1a1a1a; padding: 10px 0; border-bottom: 1px solid var(--border-color); }
        .ticker { display: flex; white-space: nowrap; animation: ticker 40s linear infinite; }
        .ticker-item { padding: 0 40px; font-size: 0.9em; font-weight: bold; }
        @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }

        .container { max-width: 1400px; margin: 0 auto; padding: 30px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; }
        
        .grid { display: grid; grid-template-columns: 300px 1fr; gap: 30px; }
        
        .sidebar { background: var(--card-bg); padding: 25px; border-radius: 12px; border: 1px solid var(--border-color); }
        .index-card { display: flex; flex-direction: column; gap: 15px; margin-bottom: 30px; }
        .index-item { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #222; }
        .index-val { font-weight: bold; color: var(--accent-color); }

        .search-box { display: flex; gap: 10px; margin-top: 20px; }
        input { flex: 1; padding: 12px; border-radius: 6px; border: 1px solid #333; background: #000; color: #fff; }
        button { background: var(--accent-color); color: #000; padding: 12px 20px; border-radius: 6px; border: none; font-weight: bold; cursor: pointer; }

        .opportunity-section { margin-bottom: 40px; }
        .op-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
        .op-card { background: var(--card-bg); padding: 15px; border-radius: 8px; border: 1px solid #333; transition: transform 0.2s; text-decoration: none; color: inherit; }
        .op-card:hover { transform: translateY(-5px); border-color: var(--accent-color); }
        
        .wl-card { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px; border-radius: 8px; border: 1px solid #333; transition: all 0.2s; text-decoration: none; color: #fff; text-align: center; }
        .wl-sym { font-weight: 900; font-size: 1.1em; margin-bottom: 2px; }
        .wl-data { font-size: 0.85em; opacity: 0.9; font-family: monospace; }
        .wl-change { font-weight: bold; }
    </style>
</head>
<body>
    <div class="ticker-wrap">
        <div class="ticker">
            {% for news in news_list %}
            <div class="ticker-item">💡 {{ news }}</div>
            {% endfor %}
            <div class="ticker-item">🗞️ NSE: Tracking Real-Time Filings & News Flashes...</div>
        </div>
    </div>

    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 2.5em; color: #fff;">Price Ladder <span style="color: var(--accent-color);">Pro</span></h1>
                <p style="color: #666; margin-top: 5px;">Institutional-Grade Support & Resistance Analysis</p>
            </div>
            <div style="text-align: right;">
                <p id="clock" style="font-family: monospace; font-size: 1.2em; color: var(--accent-color); margin-bottom: 5px;"></p>
                <script>setInterval(() => document.getElementById('clock').innerText = new Date().toLocaleTimeString(), 1000);</script>
                
                {% if kite_authenticated %}
                <div style="background: rgba(3, 218, 198, 0.1); border: 1px solid var(--up-color); color: var(--up-color); padding: 5px 12px; border-radius: 20px; font-size: 0.7em; font-weight: bold; display: inline-flex; align-items: center; gap: 5px;">
                    <span style="height: 8px; width: 8px; background: var(--up-color); border-radius: 50%;"></span>
                    ZERODHA KITE CONNECTED
                </div>
                {% else %}
                <a href="/login" style="background: var(--accent-color); color: #000; padding: 8px 15px; border-radius: 5px; text-decoration: none; font-weight: bold; font-size: 0.8em; display: inline-block;">
                    ⚡ LOGIN TO ZERODHA
                </a>
                {% endif %}
            </div>
        </div>

        <div class="grid">
            <div class="sidebar">
                <h3>Indexes</h3>
                <div class="index-card">
                    {% for k, v in index_data.items() %}
                    <div class="index-item"><span>{{ k }}</span> <span class="index-val">{{ v }}</span></div>
                    {% endfor %}
                </div>

                <h3>Quick Analysis</h3>
                <form action="/analyze" method="post">
                    <input type="text" name="symbol_search" list="symbol_list" placeholder="Enter Symbol (e.g. RELIANCE)">
                    <datalist id="symbol_list">
                        {% for sym in symbols %}<option value="{{ sym }}">{% endfor %}
                    </datalist>
                    <button type="submit" style="width: 100%; margin-top: 10px;">Deep Dive Analyze</button>
                </form>
            </div>

            <div class="main">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                    <div class="opportunity-section">
                        <h2 style="color: var(--up-color);">🚀 Top Bullish Breakouts</h2>
                        <div class="op-grid">
                            {% for opt in top_bullish %}
                            <a href="/analysis/{{ opt.symbol }}" class="op-card" style="border-left: 4px solid var(--up-color);">
                                <div style="font-weight: bold; margin-bottom: 5px;">{{ opt.symbol }}</div>
                                <div style="font-size: 0.8em; color: #aaa;">Target Gap: <span style="color: var(--up-color); font-weight: bold;">{{ opt.gap_up }}%</span></div>
                            </a>
                            {% endfor %}
                        </div>
                    </div>

                    <div class="opportunity-section">
                        <h2 style="color: var(--down-color);">📉 Top Bearish Vacuums</h2>
                        <div class="op-grid">
                            {% for opt in top_bearish %}
                             <a href="/analysis/{{ opt.symbol }}" class="op-card" style="border-left: 4px solid var(--down-color);">
                                <div style="font-weight: bold; margin-bottom: 5px;">{{ opt.symbol }}</div>
                                <div style="font-size: 0.8em; color: #aaa;">Support Gap: <span style="color: var(--down-color); font-weight: bold;">{{ opt.gap_down }}%</span></div>
                            </a>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                <div class="opportunity-section">
                    <h2 style="color: #ff9800; display: flex; align-items: center; gap: 10px;">
                        <span>🛡️ Live Watchtower Feed (L2L Scanner)</span>
                        <span style="font-size: 0.4em; background: #333; color: #ff9800; padding: 4px 8px; border-radius: 4px;">ACTIVE SCAN: 223 SYMBOLS</span>
                    </h2>
                    <div class="op-grid" style="grid-template-columns: repeat(3, 1fr);">
                        {% for alert in watchtower %}
                        <div class="op-card" onclick="window.location.href='/analysis/{{ alert.sym }}'" style="border-top: 3px solid {{ '#ff5722' if alert.type == 'Volume Hunter' else ('#f44336' if 'Support' in alert.type else ('#4caf50' if 'Resistance' in alert.type else '#2196f3')) }}; cursor: pointer; padding: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <span style="font-weight: 900; font-size: 1.2em;">{{ alert.sym }}</span>
                                <span style="font-size: 0.7em; background: {{ '#ff5722' if alert.type == 'Volume Hunter' else ('rgba(244, 67, 54, 0.2)' if alert.severity == 'high' else 'rgba(33, 150, 243, 0.2)') }}; color: #fff; padding: 2px 6px; border-radius: 4px;">{{ alert.type }}</span>
                            </div>
                            <div style="margin-top: 10px; font-size: 0.9em; color: #bbb;">
                                Price: <span style="color: #fff; font-weight: bold;">{{ alert.p }}</span> <br>
                                {% if alert.type == 'Volume Hunter' %}
                                    Metric: <span style="color: #ff5722; font-weight: bold;">{{ alert.dist }}x Vol Surge</span> <br>
                                {% else %}
                                    Level: <span style="color: var(--accent-color);">{{ alert.lvl }}</span> <br>
                                    Dist: <span style="color: {{ '#ffeb3b' if alert.dist < 0.2 else '#fff' }};">{{ alert.dist }}%</span> <br>
                                {% endif %}
                                Next Target: <span style="color: var(--up-color);">{{ alert.target }}</span>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>

                <div class="opportunity-section" style="margin-top: 20px;">
                    <h2 style="color: var(--accent-color);">💎 Institutional Watchlist (Top 20)</h2>
                    <p style="font-size: 0.8em; color: #666; margin-bottom: 15px;">Left-Click to Analyze | Right-Click to Alter Symbol</p>
                    <div class="op-grid" style="grid-template-columns: repeat(5, 1fr);">
                        {% for item in watchlist %}
                        {% set opacity = (item.change|abs / max_intensity * 0.6) + 0.1 %}
                        {% set bg = 'rgba(3, 218, 198, ' + opacity|string + ')' if item.change > 0 else 'rgba(207, 102, 121, ' + opacity|string + ')' %}
                        {% if item.change == 0 %}{% set bg = 'var(--card-bg)' %}{% endif %}
                        
                        <div class="wl-card" onclick="window.location.href='/analysis/{{ item.symbol }}'" oncontextmenu="editSymbol(event, this, {{ item.orig_idx }}); return false;" style="background: {{ bg }}; cursor: pointer;">
                            <div class="wl-sym">{{ item.symbol }}</div>
                            <div class="wl-data">
                                <span>{{ item.price }}</span> 
                                <span class="wl-change" style="color: {{ '#fff' if item.change == 0 else ('#5fffd7' if item.change > 0 else '#ff8a9a') }};">
                                    {{ '+' if item.change > 0 }}{{ item.change }}%
                                </span>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function editSymbol(e, el, index) {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            const currentSym = el.querySelector('.wl-sym').innerText.trim();
            const input = document.createElement('input');
            input.value = currentSym;
            input.style.width = '100%';
            input.style.textAlign = 'center';
            input.style.background = '#000';
            input.style.color = '#fff';
            input.style.border = '1px solid var(--accent-color)';
            
            el.innerHTML = '';
            el.appendChild(input);
            input.focus();

            input.onblur = () => window.location.reload(); // Simplest to reload to reset state
            input.onkeydown = async (e) => { 
                if(e.key === 'Enter') {
                    const newSym = input.value.toUpperCase().trim();
                    if (newSym && newSym !== currentSym) {
                        const response = await fetch('/update_watchlist', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({index: index, symbol: newSym})
                        });
                        if (response.ok) {
                            window.location.reload();
                            return;
                        }
                    }
                    window.location.reload();
                }
                if(e.key === 'Escape') window.location.reload();
            };
        }

        // --- NEWS MODAL ---
        function showNewsModal(text, source) {
            document.getElementById('modal-text').innerText = text;
            document.getElementById('modal-source').innerText = source;
            document.getElementById('news-modal').style.display = 'flex';
        }
        function closeNewsModal() {
            document.getElementById('news-modal').style.display = 'none';
        }
        // Close on escape key
        document.onkeydown = function(evt) {
            evt = evt || window.event;
            if (evt.keyCode == 27) {
                closeNewsModal();
            }
        };

    </script>

    <!-- Modal HTML -->
    <div id="news-modal" class="modal-overlay" onclick="if(event.target === this) closeNewsModal()">
        <div class="modal-content">
            <span class="modal-close" onclick="closeNewsModal()">&times;</span>
            <h3 style="margin-top: 0; color: var(--accent-color); font-size: 0.9em; text-transform: uppercase; border-bottom: 1px solid #333; padding-bottom: 10px;" id="modal-source">SOURCE</h3>
            <p id="modal-text" style="font-size: 1.1em; line-height: 1.6; margin-top: 15px;"></p>
            <div style="margin-top: 20px; text-align: right;">
                <button onclick="closeNewsModal()" style="background: transparent; border: 1px solid #333; color: #888; padding: 5px 15px; border-radius: 4px; cursor: pointer;">Close</button>
            </div>
        </div>
    </div>
</body>
</html>
"""

ANALYSIS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ row.symbol }} - Advanced Analysis</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        :root {
            --bg-color: #0c0d10;
            --card-bg: #131722;
            --accent-color: #bb86fc;
            --up-color: #03dac6;
            --down-color: #cf6679;
            --ltp-color: #ffd700;
        }
        body { font-family: 'Segoe UI', sans-serif; background: var(--bg-color); color: #d1d4dc; margin: 0; padding: 20px; }
        .back-btn { text-decoration: none; color: #fff; background: #1e222d; padding: 10px 20px; border-radius: 6px; border: 1px solid #363a45; display: inline-block; margin-bottom: 20px; }
        
        .analysis-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; background: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid #2a2e39; }
        .stat-group { display: flex; gap: 30px; }
        .stat-item { text-align: center; }
        .stat-val { font-size: 1.5em; font-weight: bold; display: block; }
        .stat-label { font-size: 0.75em; color: #666; text-transform: uppercase; }

        .ladder-wrap { overflow-x: auto; margin-bottom: 30px; }
        table { width: 100%; border-collapse: collapse; background: #000; border-radius: 8px; overflow: hidden; font-size: 0.85em; }
        th, td { padding: 12px; text-align: center; border: 1px solid #1a1a1a; }
        th { background: #131722; color: #888; }
        .ltp-cell { background: #1e222d; color: var(--ltp-color); font-weight: 900; font-size: 1.2em; border: 2px solid var(--ltp-color); }
        
        .target-tag { font-size: 9px; padding: 2px 5px; background: var(--accent-color); color: #000; border-radius: 3px; font-weight: 900; }
        
        
        .vmf-container { height: 600px; background: #050505; position: relative; border: 1px solid #222; border-radius: 12px; flex: 1; transition: all 0.3s ease; overflow: hidden; display: flex; }
        .vmf-container.maximized { 
            position: fixed !important; top: 0 !important; left: 0 !important; width: 100vw !important; height: 100vh !important; 
            z-index: 9999 !important; border-radius: 0 !important; 
        }
        .vmf-flow { flex: 1; position: relative; height: 100%; border-right: 1px solid #1a1a1a; cursor: crosshair; }
        .vmf-axis { width: 70px; height: 100%; position: relative; background: #080808; cursor: ns-resize; user-select: none; }
        .vmf-chart-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 5; pointer-events: none; }
        
        .analysis-visual-grid { display: grid; grid-template-columns: 350px 1fr 300px; gap: 20px; margin-top: 20px; }
        .chart-container { height: 600px; background: var(--card-bg); border-radius: 12px; border: 1px solid #222; overflow: hidden; }
        .news-sidebar { height: 600px; background: var(--card-bg); border-radius: 12px; border: 1px solid #222; padding: 15px; display: flex; flex-direction: column; gap: 20px; overflow: hidden; }
        .news-section { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; }
        .news-header { color: var(--accent-color); font-size: 0.8em; margin-bottom: 10px; text-transform: uppercase; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 5px; z-index: 2; background: var(--card-bg); }
        
        .news-scroll-box { flex: 1; overflow: hidden; position: relative; }
        .news-scroll-content { position: absolute; width: 100%; animation: scrollUp 25s linear infinite; }
        .news-scroll-content:hover { animation-play-state: paused; }
        
        @keyframes scrollUp {
            0% { top: 100%; }
            100% { top: -200%; } /* Scroll all the way up */
        }

        .news-item { margin-bottom: 15px; font-size: 0.8em; border-left: 2px solid #333; padding-left: 10px; transition: 0.2s; background: rgba(255,255,255,0.01); padding: 8px; border-radius: 0 4px 4px 0; cursor: pointer; }
        .news-item:hover { border-color: var(--accent-color); background: rgba(255,255,255,0.05); }

        /* Modal Styles */
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 10000; justify-content: center; align-items: center; }
        .modal-content { background: #1e222d; padding: 25px; border-radius: 12px; border: 1px solid var(--accent-color); max-width: 500px; width: 90%; color: #fff; box-shadow: 0 0 20px rgba(187, 134, 252, 0.2); }
        .modal-close { float: right; cursor: pointer; color: #888; font-size: 1.2em; }
        .modal-close:hover { color: #fff; }
    </style>
</head>
<body>
    <a href="/" class="back-btn">← Back to Dashboard</a>

    <div class="analysis-header">
        <div>
        <div style="display: flex; align-items: center; gap: 15px;">
            <h1 style="margin: 0; color: #fff;">{{ row.symbol }} <span style="font-size: 0.5em; vertical-align: middle; color: #666;">Analysis Deep Dive</span></h1>
            {% if kite_authenticated %}
            <span style="background: rgba(3, 218, 198, 0.1); border: 1px solid var(--up-color); color: var(--up-color); padding: 2px 8px; border-radius: 4px; font-size: 0.6em; font-weight: bold;">KITE PRO ACTIVE</span>
            {% endif %}
        </div>
            <div style="margin-top: 5px; display: flex; align-items: center; gap: 15px;">
                <p style="margin: 0; color: #888;">LTP: <span style="color: var(--ltp-color); font-weight: bold;">{{ row.latest }}</span></p>
                <div id="ticker-config" style="font-size: 0.75em; background: #1e222d; padding: 4px 10px; border-radius: 4px; border: 1px solid #333;">
                    <span style="color: #666;">TV Ticker:</span> 
                    <input type="text" id="manual-ticker" value="NSE:{{ row.symbol }}" style="background:transparent; border:none; color:var(--accent-color); width: 100px; outline:none; font-family:monospace; font-weight:bold;">
                    <button onclick="updateChart()" style="background:none; border:none; color:#fff; cursor:pointer; padding:0 5px;">🔄</button>
                    | <a href="#" onclick="switchExchange('BSE')" style="color:#aaa; text-decoration:none;">Try BSE</a>
                    | <button onclick="rotateCharts()" id="toggle-btn" style="background: var(--accent-color); color: #000; border: none; padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: bold; cursor: pointer; margin-left: 10px;">VIEW: TV PUBLIC</button>
                </div>
            </div>
        </div>
        <div class="stat-group">
            {% if market_depth %}
            <div style="background: rgba(255,255,255,0.05); padding: 5px 15px; border-radius: 8px; border: 1px solid #333; font-size: 0.75em;">
                <div style="display: flex; gap: 20px;">
                    <div>
                        <span style="color: var(--up-color); font-weight: bold;">BIDS</span>
                        <div style="font-family: monospace; display: grid; grid-template-columns: 60px 50px;">
                            {% for b in market_depth.buy %}
                                <span style="color: #888;">{{ b.price }}</span> <span style="color: var(--up-color);">{{ b.quantity }}</span>
                            {% endfor %}
                        </div>
                    </div>
                    <div>
                        <span style="color: var(--down-color); font-weight: bold;">OFFERS</span>
                        <div style="font-family: monospace; display: grid; grid-template-columns: 60px 50px;">
                            {% for s in market_depth.sell %}
                                <span style="color: #888;">{{ s.price }}</span> <span style="color: var(--down-color);">{{ s.quantity }}</span>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}

            <div class="stat-item">
                <span class="stat-val" style="color: var(--up-color);">{{ row.upside_runway }}%</span>
                <span class="stat-label">Upside Runway</span>
            </div>
            <div class="stat-item">
                <span class="stat-val" style="color: var(--down-color);">{{ row.downside_runway }}%</span>
                <span class="stat-label">Downside Shield</span>
            </div>
            <div class="stat-item">
                <span class="stat-val" style="color: var(--accent-color);">{{ row.clusters|length }}</span>
                <span class="stat-label">Key Zones</span>
            </div>
        </div>
    </div>

    {% if options_data %}
    <div style="margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <!-- CALL SIDE (CE) -->
        <div style="background: rgba(3, 218, 198, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(3, 218, 198, 0.2);">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 10px;">
                <span style="color: var(--up-color); font-weight: bold;">CALLS (CE) - {{ options_data.strike }}</span>
                <span style="color: #666; font-size: 0.8em;">EXPIRY: {{ options_data.expiry }}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-family: monospace; font-size: 0.8em;">
                <div>
                    <div style="color: #888; border-bottom: 1px solid #222; margin-bottom: 5px;">BIDS</div>
                    {% for b in options_data.ce.depth.buy %}
                    <div style="display: flex; justify-content: space-between;"><span style="color: #fff;">{{ b.price }}</span> <span style="color: var(--up-color);">{{ b.quantity }}</span></div>
                    {% endfor %}
                </div>
                <div>
                    <div style="color: #888; border-bottom: 1px solid #222; margin-bottom: 5px;">OFFERS</div>
                    {% for s in options_data.ce.depth.sell %}
                    <div style="display: flex; justify-content: space-between;"><span style="color: #fff;">{{ s.price }}</span> <span style="color: var(--down-color);">{{ s.quantity }}</span></div>
                    {% endfor %}
                </div>
            </div>
        </div>
        <!-- PUT SIDE (PE) -->
        <div style="background: rgba(207, 102, 121, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(207, 102, 121, 0.2);">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 10px;">
                <span style="color: var(--down-color); font-weight: bold;">PUTS (PE) - {{ options_data.strike }}</span>
                <span style="color: #666; font-size: 0.8em;">EXPIRY: {{ options_data.expiry }}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-family: monospace; font-size: 0.8em;">
                <div>
                    <div style="color: #888; border-bottom: 1px solid #222; margin-bottom: 5px;">BIDS</div>
                    {% for b in options_data.pe.depth.buy %}
                    <div style="display: flex; justify-content: space-between;"><span style="color: #fff;">{{ b.price }}</span> <span style="color: var(--up-color);">{{ b.quantity }}</span></div>
                    {% endfor %}
                </div>
                <div>
                    <div style="color: #888; border-bottom: 1px solid #222; margin-bottom: 5px;">OFFERS</div>
                    {% for s in options_data.pe.depth.sell %}
                    <div style="display: flex; justify-content: space-between;"><span style="color: #fff;">{{ s.price }}</span> <span style="color: var(--down-color);">{{ s.quantity }}</span></div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <div class="ladder-wrap">
        <table>
            <thead>
                <tr>
                    {% for i in range(-5, 0) %}<th>P{{i}}</th>{% endfor %}
                    <th class="ltp-cell">LTP</th>
                    {% for i in range(1, 6) %}<th>P+{{i}}</th>{% endfor %}
                </tr>
            </thead>
            <tbody>
                <tr>
                    {# P-5 to P-1 #}
                    {% for i in range(0, 5) %}
                        {% set p_index = i %}
                        {% set gap = row.gaps[p_index] if row.gaps[p_index] is defined else 0 %}
                        {# Base opacity 0.15 + scaled gap #}
                        <td style="background: rgba(207, 102, 121, {{ 0.15 + (gap|abs * 0.5) }});">
                            {% if i == 4 %}<div class="target-tag">SUP</div>{% endif %}
                            <div style="font-weight: bold;">{{ row.prices[p_index] if row.prices[p_index] else '-' }}</div>
                            <div style="font-size: 0.7em; opacity: 0.6;">{{ gap|round(2) }}%</div>
                        </td>
                    {% endfor %}
                    
                    <td class="ltp-cell">{{ row.latest }}</td>
                    
                    {# P+1 to P+5 #}
                    {% for i in range(6, 11) %}
                        {% set p_index = i %}
                        {% set gap = row.gaps[p_index-1] if row.gaps[p_index-1] is defined else 0 %}
                        <td style="background: rgba(3, 218, 198, {{ 0.15 + (gap|abs * 0.5) }});">
                            {% if i == 6 %}<div class="target-tag">RES</div>{% endif %}
                            <div style="font-weight: bold;">{{ row.prices[p_index] if row.prices[p_index] else '-' }}</div>
                            <div style="font-size: 0.7em; opacity: 0.6;">{{ gap|round(2) }}%</div>
                        </td>
                    {% endfor %}
                </tr>
            </tbody>
        </table>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 300px; gap: 20px; margin-top: 50px;">
        <h2 style="margin: 0;">Visual Momentum Flow & Intraday Action</h2>
        <h2 style="margin: 0; color: var(--accent-color);">Symbol News Pulse</h2>
    </div>
    
    <div class="analysis-visual-grid" style="grid-template-columns: 350px 1fr 300px;">
        <div class="vmf-container" id="vmf-main">
            <button onclick="toggleVMF()" style="position: absolute; top: 15px; right: 90px; z-index: 1000; background: rgba(0,0,0,0.8); border: 1px solid var(--accent-color); color: var(--accent-color); border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 10px; font-weight: bold; text-transform: uppercase;"> ✥ Expand View</button>
            <div id="vmf-flow" class="vmf-flow">
                <div id="vmf-chart" class="vmf-chart-overlay"></div>
            </div>
            <div id="vmf-axis" class="vmf-axis"></div>
        </div>

        <div id="tradingview_widget" class="chart-container"></div>
        <div id="kite_chart" class="chart-container" style="display: none;">
            <iframe id="kite-iframe" src="" style="width: 100%; height: 100%; border: none;"></iframe>
        </div>
        <div id="local_chart" class="chart-container" style="display: none;"></div>
        
        <div class="news-sidebar">
            <!-- Section 1: Live News -->
            <div class="news-section" style="border-bottom: 1px solid #333; padding-bottom: 10px;">
                <div class="news-header">Live Market Pulse</div>
                <div class="news-scroll-box">
                    <div class="news-scroll-content">
                        {% for news in news_list %}
                        <div class="news-item" onclick="showNewsModal('{{ news|escape }}', 'LIVE MARKET PULSE')">
                            <div style="color: #d1d4dc; line-height: 1.4;">{{ news }}</div>
                            <div style="font-size: 0.7em; color: #555; margin-top: 5px;">LIVE FEED</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <!-- Section 2: Historical Context -->
            <div class="news-section">
                <div class="news-header">30-Day Context</div>
                <div class="news-scroll-box">
                    <div class="news-scroll-content" style="animation-duration: 35s;"> <!-- Faster scroll context -->
                        {% if hist_news %}
                            {% for hn in hist_news %}
                            <div class="news-item" onclick="showNewsModal('{{ hn.title|escape }}', '{{ hn.source|escape }} - {{ hn.date }}')">
                                <div style="color: #ccc; line-height: 1.4;">{{ hn.title }}</div>
                                <div style="font-size: 0.7em; color: var(--accent-color); margin-top: 5px; display: flex; justify-content: space-between;">
                                    <span>{{ hn.date }}</span>
                                    <span>{{ hn.source }}</span>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="news-item" style="border-color: #333;">
                                <div style="color: #666;">No major news events found for {{ row.symbol }} in the last 30 days.</div>
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>


    <!-- TradingView Widget BEGIN -->
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
        let currentTicker = "NSE:{{ row.symbol }}";
        {% if row.symbol == 'NIFTY' %} currentTicker = 'NSE:NIFTY';
        {% elif row.symbol == 'BANKNIFTY' %} currentTicker = 'NSE:BANKNIFTY';
        {% elif row.symbol == 'SENSEX' %} currentTicker = 'BSE:SENSEX';
        {% endif %}
        
        let chartState = "TV"; // TV, KITE, LOCAL
        let localChartObj = null;
        const kiteToken = "{{ kite_token }}";

        function loadChart(ticker) {
            new TradingView.widget({
                "autosize": true,
                "symbol": ticker,
                "interval": "5",
                "timezone": "Asia/Kolkata",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_widget"
            });
        }

        function rotateCharts() {
            const tv = document.getElementById('tradingview_widget');
            const kiteEl = document.getElementById('kite_chart');
            const local = document.getElementById('local_chart');
            const btn = document.getElementById('toggle-btn');
            
            if (chartState === "TV") {
                // Switch to Kite
                tv.style.display = 'none';
                kiteEl.style.display = 'block';
                chartState = "KITE";
                btn.innerText = "VIEW: KITE PRO";
                
                const symbol = "{{ row.symbol }}";
                const exchange = document.getElementById('manual-ticker').value.split(':')[0] || 'NSE';
                // Official Zerodha Web-TV URL
                document.getElementById('kite-iframe').src = `https://kite.zerodha.com/chart/web/tvc/${exchange}/${symbol}/${kiteToken}`;
                
            } else if (chartState === "KITE") {
                // Switch to Local
                kiteEl.style.display = 'none';
                local.style.display = 'block';
                chartState = "LOCAL";
                btn.innerText = "VIEW: LOCAL CHART";
                initLocalChart();
                
            } else {
                // Switch back to TV
                local.style.display = 'none';
                tv.style.display = 'block';
                chartState = "TV";
                btn.innerText = "VIEW: TV PUBLIC";
            }
        }

        function initLocalChart() {
            if (localChartObj) return;
            const data = {{ intraday|tojson }};
            const histData = {{ history|tojson }};
            const symbol = "{{ row.symbol }}";
            
            const chartOptions = { 
                layout: { textColor: '#d1d4dc', background: { type: 'solid', color: '#131722' } },
                grid: { vertLines: { color: '#2a2e39' }, horzLines: { color: '#2a2e39' } },
                timeScale: { borderColor: '#2a2e39', timeVisible: true }
            };
            localChartObj = LightweightCharts.createChart(document.getElementById('local_chart'), chartOptions);

            if (data && data.length > 0) {
                const series = localChartObj.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350' });
                series.setData(data);
            } else {
                // Fallback to Historical Line Chart
                const lineSeries = localChartObj.addLineSeries({ color: '#bb86fc', lineWidth: 2 });
                const formattedHist = histData.map(d => ({ time: d.Date, value: d[symbol] }));
                lineSeries.setData(formattedHist);
            }
            localChartObj.timeScale().fitContent();
        }

        function updateChart() {
            const val = document.getElementById('manual-ticker').value.trim();
            if(val) loadChart(val);
        }

        function switchExchange(ex) {
            const sym = "{{ row.symbol }}";
            const newTicker = ex + ":" + sym;
            document.getElementById('manual-ticker').value = newTicker;
            loadChart(newTicker);
        }

        // --- VMF INTERACTIVE ENGINE ---
        const vmfData = {
            clusters: {{ row.clusters|tojson }},
            latest: {{ row.latest }},
            min: {{ row.min_p }},
            max: {{ row.max_p }},
            intraday: {{ intraday|tojson }}
        };

        let currentMin = vmfData.min;
        let currentMax = vmfData.max;
        let vmfChartObj = null;

        function renderVMF() {
            const flow = document.getElementById('vmf-flow');
            const axis = document.getElementById('vmf-axis');
            
            Array.from(flow.children).forEach(child => { if(child.id !== 'vmf-chart') child.remove(); });
            axis.innerHTML = '';

            const range = currentMax - currentMin;
            
            vmfData.clusters.forEach((c, idx) => {
                const topP = 100 - ((c.price - currentMin) / range * 92 + 4);
                
                if (idx > 0) {
                    const prevC = vmfData.clusters[idx - 1];
                    const gap = Math.abs((prevC.price - c.price) / c.price * 100);
                    const prevTop = 100 - ((prevC.price - currentMin) / range * 92 + 4);
                    const height = topP - prevTop;

                    if (height > 0) {
                        const midPrice = (prevC.price + c.price) / 2;
                        const isAbove = midPrice > vmfData.latest;
                        
                        // RICH DIRECTIONAL COLORS
                        const rColor = isAbove ? '0, 255, 136' : '255, 51, 102'; // Vivid Emerald : Crimson
                        const glowIntense = Math.min(0.15 + (gap * 0.1), 0.5); // Gradients scale with GAP strength
                        
                        const vac = document.createElement('div');
                        vac.style.cssText = `position: absolute; width: 100%; top: ${prevTop}%; height: ${height}%; 
                                            background: linear-gradient(180deg, rgba(${rColor}, ${glowIntense}) 0%, rgba(${rColor}, 0.05) 50%, rgba(${rColor}, ${glowIntense}) 100%); 
                                            border-left: 6px solid rgba(${rColor}, 0.8); transition: opacity 0.3s;`;
                        
                        // Visibly Graded Text
                        if (gap > 0.4) {
                            const opacityVal = Math.min(0.4 + (gap*0.2), 1);
                            vac.innerHTML = `<div style="position: absolute; width: 100%; text-align: center; top: 50%; transform: translateY(-50%); font-size: 11px; font-weight: 900; color: #fff; text-shadow: 0 0 15px rgb(${rColor}); opacity: ${opacityVal}; letter-spacing: 3px;">✦ ${gap.toFixed(2)}% VACUUM ✦</div>`;
                        }
                        flow.appendChild(vac);
                    }
                }

                // Render Lines (High Visibility Settlement Walls)
                const line = document.createElement('div');
                const isLtp = c.is_ltp || Math.abs(c.price - vmfData.latest) < 0.01;
                line.style.cssText = `position: absolute; width: 100%; height: 2px; top: ${topP}%; background: ${isLtp ? '#ffd700' : 'rgba(61, 90, 254, 0.6)'}; z-index: 10; box-shadow: 0 0 8px ${isLtp ? '#ffd700' : 'transparent'};`;
                flow.appendChild(line);

                // Render Labels on Axis
                const label = document.createElement('div');
                label.style.cssText = `position: absolute; right: 5px; top: ${topP}%; transform: translateY(-50%); font-size: 10px; font-weight: bold; background: #000; padding: 2px 6px; border: 1px solid ${isLtp ? '#ffd700' : '#2a2e39'}; color: ${isLtp ? '#ffd700' : '#888'}; border-radius: 4px; z-index: 20;`;
                label.innerText = c.price.toFixed(2);
                axis.appendChild(label);
            });

            syncVMFChart();
        }

        function initVMFChart() {
            const chartDiv = document.getElementById('vmf-chart');
            const flow = document.getElementById('vmf-flow');
            
            // Get actual dimensions from parent container
            const width = flow.offsetWidth || 280;
            const height = flow.offsetHeight || 600;
            
            const chartOptions = { 
                width: width,
                height: height,
                layout: { textColor: '#d1d4dc', background: { type: 'solid', color: 'transparent' } },
                grid: { vertLines: { visible: false }, horzLines: { visible: false } },
                timeScale: { visible: false },
                rightPriceScale: { visible: false },
                crosshair: { vertLine: { visible: false }, horzLine: { visible: false } },
                handleScroll: false,
                handleScale: false
            };
            vmfChartObj = LightweightCharts.createChart(chartDiv, chartOptions);
            const series = vmfChartObj.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', wickUpColor: '#26a69a', wickDownColor: '#ef5350' });
            if (vmfData.intraday && vmfData.intraday.length > 0) {
                series.setData(vmfData.intraday);
            }
            vmfChartObj.timeScale().fitContent();
            syncVMFChart();
        }

        function syncVMFChart() {
            if (!vmfChartObj) return;
            // Force the chart to match our VMF linear range exactly
            vmfChartObj.priceScale('right').applyOptions({
                autoScale: false,
                scaleMargins: { top: 0.04, bottom: 0.04 },
            });
            // We need to set the precise range based on our currentMin/currentMax
            // Lightweight charts uses internal price range, we override the view
            const range = currentMax - currentMin;
            // Pad it slightly to match our 92% + 4% + 4% logic
            const paddedMin = currentMin - (range * 0.04);
            const paddedMax = currentMax + (range * 0.04);
            
            // This is a manual override hack for alignment
            const candles = vmfChartObj.series()[0];
            vmfChartObj.priceScale('right').setFixedRange({
                from: paddedMin,
                to: paddedMax
            });
        }

        // Wheel Zoom Trigger
        document.getElementById('vmf-axis').addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.05;
            const delta = e.deltaY > 0 ? 1 : -1;
            const range = currentMax - currentMin;
            
            currentMin -= range * zoomSpeed * delta;
            currentMax += range * zoomSpeed * delta;
            
            renderVMF();
        });

        function toggleVMF() {
            const el = document.getElementById('vmf-main');
            const isMax = el.classList.toggle('maximized');
            const btn = el.querySelector('button');
            btn.innerText = isMax ? '✕ Minimize (ESC)' : '✥ Expand View';
            document.body.style.overflow = isMax ? 'hidden' : 'auto';
            
            // Resize chart after toggle animation completes
            setTimeout(() => {
                if(vmfChartObj) {
                    const flow = document.getElementById('vmf-flow');
                    vmfChartObj.resize(flow.offsetWidth, flow.offsetHeight);
                }
                renderVMF();
            }, 350);
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const el = document.getElementById('vmf-main');
                if (el.classList.contains('maximized')) toggleVMF();
            }
        });

        initVMFChart();
        renderVMF();
        loadChart(currentTicker);
    </script>
    <!-- TradingView Widget END -->
</body>
</html>
"""


if __name__ == '__main__':
    app.run(debug=True)
