import pandas as pd
import os
import datetime
import pytz
import yfinance as yf
import numpy as np
from flask import Flask, render_template_string, request, redirect, session, jsonify
import requests
import re
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException

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

import time

# Global Kite Object
kite = None
VOL_BASELINE = {} # Cache for Avg Daily Volume
VOL_SAMPLES = {} # Store recent (time, total_volume) snapshots
NFO_INSTRUMENTS = None # Cache for F&O instruments
INSTRUMENT_MAP = {} # Cache for symbol -> token mapping

# Global Dashboard
NIFTY_50_SYMBOLS = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 
    'BEL', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 
    'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC', 
    'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LTIM', 'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 
    'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SHRIRAMFIN', 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM', 
    'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO', 'ZOMATO'
]

# Global Cache/State
GLOBAL_DASHBOARD_CACHE = {
    'last_scan_time': 0,
    'top_bullish': [],
    'top_bearish': [],
    'heatmap_data': [],
    'index_data': {},
    'all_symbols': [],
    'curr_adv_map': {},
    'last_prices': {}
}
# Connection State
KITE_CONNECTED = False
# Logic State
EOD_UPDATED_DATE = None 

def is_market_open():
    """Returns True if current time is Mon-Fri, 09:00 to 15:35 IST"""
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz)
    
    # Monday=0, Sunday=6
    if now.weekday() > 4: 
        return False
        
    start_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=35, second=0, microsecond=0)
    
    return start_time <= now <= end_time

if KITE_API_KEY:
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        if KITE_ACCESS_TOKEN:
            kite.set_access_token(KITE_ACCESS_TOKEN)
            # Verify Session Validity
            try:
                kite.profile()
                print("DEBUG: Zerodha Token Validated - Connected")
            except Exception as e:
                print(f"DEBUG: Saved Token Invalid/Expired: {e}")
                kite.access_token = None # Clear invalid token so UI shows disconnected
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



# Global Volume State for Intraday Tracking
INTRADAY_VOL_STATE = {} # {symbol: {'last_vol': 0, 'last_time': timestamp, 'open_vol_est': 0}}

def detect_intraday_spike(symbol, current_vol):
    """
    Detects volume spikes by comparing instantaneous volume rate vs daily average rate.
    Returns: (severity_ratio, message_type)
    """
    global INTRADAY_VOL_STATE
    
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    market_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    
    # 1. Initialize State if First Seen
    if symbol not in INTRADAY_VOL_STATE:
        INTRADAY_VOL_STATE[symbol] = {
            'last_vol': current_vol,
            'last_time': current_time,
            'first_seen_vol': current_vol # Approx open vol if seen early, or mid-day snapshot
        }
        return 0, None

    state = INTRADAY_VOL_STATE[symbol]
    last_vol = state['last_vol']
    last_time = state['last_time']
    
    # Update State for next time
    INTRADAY_VOL_STATE[symbol]['last_vol'] = current_vol
    INTRADAY_VOL_STATE[symbol]['last_time'] = current_time
    
    # 2. Calculate Time Deltas
    # Minutes since 9:15 AM (Total Market Time)
    time_since_open = (current_time - market_start).total_seconds() / 60
    if time_since_open < 5: return 0, None # Too early/noisy
    
    # Minutes since last poll (Instantaneous Time)
    time_delta = (current_time - last_time).total_seconds() / 60
    if time_delta < 0.05: return 0, None # Too fast (less than 3 sec)
    
    # 3. Calculate Volume Deltas
    vol_delta = current_vol - last_vol
    if vol_delta <= 0: return 0, None
    
    # 4. Calculate Rates (Shares per Minute)
    # Average Rate for the whole day so far
    # We estimate 'volume so far' as current_vol. 
    # Ideally quote['volume'] is cumulative.
    avg_rate = current_vol / time_since_open
    
    # Instantaneous Rate (Since last poll)
    instant_rate = vol_delta / time_delta
    
    if avg_rate <= 0: return 0, None
    
    # 5. Compare Ratio
    ratio = instant_rate / avg_rate
    
    # 6. determine Alert
    # Ratio > 1.0 means faster than average.
    # We want significant spikes.
    
    if ratio > 5.0 and vol_delta > 5000: # 5x speed + min size
        return ratio, "Flash Spurt"
    elif ratio > 3.0 and vol_delta > 5000:
        return ratio, "Volume Rising"
        
    return ratio, None


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
    global KITE_CONNECTED
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
            
            try:
                ltp_data = kite.ltp(final_instruments)
                # If LTP call succeeds, we are connected
                KITE_CONNECTED = True
                
                for k, v in ltp_data.items():
                    sym = k.replace("NSE:", "").replace("NIFTY 50", "NIFTY").replace("NIFTY BANK", "BANKNIFTY")
                    live_prices[sym] = v['last_price']
                
                
            except TokenException:
                 print("DEBUG: Zerodha Token Expired during LTP fetch.")
                 kite.access_token = None
                 KITE_CONNECTED = False
            except Exception as e:
                print(f"Kite LTP Error: {e}")
                KITE_CONNECTED = False

        # Identify what is missing
        missing_symbols = [s for s in symbols if s not in live_prices]
        if not missing_symbols:
            return live_prices

        # Prio 2: yfinance (Fallback for MISSING only)
        ticker_map = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'CNXFINANCE': 'NIFTY_FIN_SERVICE.NS'
        }
        yf_tickers = [ticker_map.get(sym, f"{sym}.NS") for sym in missing_symbols]
        
        # Download batch data for missing
        # period="1d" gives today's data. interval="1m" gives minute data.
        data = yf.download(yf_tickers, period="1d", interval="1m", group_by='ticker', threads=True)
        
        if not data.empty:
            print(f"DEBUG: Fetched {len(yf_tickers)} missing symbols from YF")
            # Check if we have multiple tickers or just one
            if len(yf_tickers) > 1:
                for sym, yf_sym in zip(missing_symbols, yf_tickers):
                    try:
                        # yfinance structure for multiple tickers: data[Ticker]['Close']
                        if yf_sym in data.columns.levels[0]:
                            closes = data[yf_sym]['Close'].dropna()
                            if not closes.empty:
                                live_prices[sym] = closes.iloc[-1]
                            else:
                                print(f"DEBUG: No close data for {sym}")
                        else:
                            print(f"DEBUG: {yf_sym} not in columns")
                    except Exception as e:
                        print(f"Error extracting data for {sym}: {e}")
            else:
                # Single ticker structure: data['Close']
                sym = missing_symbols[0]
                try:
                    # If multi-index (Ticker, PriceType)
                    if isinstance(data.columns, pd.MultiIndex):
                         closes = data['Close'].dropna()
                    else:
                         closes = data['Close'].dropna()
                         
                    if not closes.empty:
                        live_prices[sym] = closes.iloc[-1]
                except Exception as e:
                    print(f"single ticker extract error: {e}")
        else:
            print("DEBUG: No data received from yfinance")
                    
    except Exception as e:
        print(f"Error fetching batch live prices: {e}")
        
    return live_prices

@app.route('/debug/kite')
def debug_kite():
    if not kite: return jsonify({'status': 'Error', 'msg': 'Kite object not initialized'})
    if not kite.access_token: return jsonify({'status': 'Disconnected', 'msg': 'No Access Token'})
    
    try:
        profile = kite.profile()
        return jsonify({
            'status': 'Connected',
            'user_id': profile.get('user_id'),
            'user_name': profile.get('user_name'),
            'email': profile.get('email'),
            'timestamp': str(datetime.datetime.now())
        })
    except Exception as e:
         return jsonify({'status': 'Error', 'msg': str(e)})

@app.route('/login')
def login():
    global kite
    request_token = request.args.get("request_token")
    
    if request_token:
        try:
            data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
            access_token = data["access_token"]
            kite.set_access_token(access_token)
            # Persist token to .env
            update_env_file("KITE_ACCESS_TOKEN", access_token)
            global KITE_CONNECTED
            KITE_CONNECTED = True
            return redirect("/")
        except Exception as e:
            return f"Login failed: {e}", 400

    if not kite:
        return "Kite Connect not initialized. Check API keys.", 400

    # Redirect to Kite login
    return redirect(kite.login_url())

@app.route('/logout')
def logout():
    global kite, KITE_CONNECTED
    
    # 1. Clear Global State
    if kite: kite.access_token = None
    KITE_CONNECTED = False
    
    # 2. Clear Persistence (.env)
    update_env_file("KITE_ACCESS_TOKEN", "")
    
    print("DEBUG: User manually disconnected from Zerodha.")
    return redirect("/")

def update_env_file(key, value):
    env_path = ".env"
    try:
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        
        found = False
        with open(env_path, "w") as f:
            for line in lines:
                if line.startswith(f"{key}="):
                    f.write(f"{key}='{value}'\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"\n{key}='{value}'\n")
        print(f"DEBUG: Saved {key} to .env")
    except Exception as e:
        print(f"Error saving to .env: {e}")

def get_dashboard_data():
    global GLOBAL_DASHBOARD_CACHE, EOD_UPDATED_DATE
    
    # Check if 5 minutes passed since last scan
    current_time = time.time()
    SCAN_INTERVAL = 300 # 5 minutes
    
    # --- PHASE 1: SCANNING (Cached) ---
    # Market Hours Logic for Scanner
    # 1. Market Open OR 2. EOD Sync needed
    should_scan_market = is_market_open()
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz)
    if now.hour >= 16 and EOD_UPDATED_DATE != now.strftime('%Y-%m-%d'):
        should_scan_market = True # Force scan for EOD

    if should_scan_market and (current_time - GLOBAL_DASHBOARD_CACHE['last_scan_time'] > SCAN_INTERVAL):
        print("DEBUG: Running 5-Minute Scan Refresh...")
        index_data = {}
        top_bullish = []
        top_bearish = []
        heatmap_data = []
        # Fallbacks
        display_symbols = []
        watchlist_symbols = load_watchlist()
        
        try:
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE, low_memory=False)
                df.columns = [col.strip() for col in df.columns]
                index_symbols = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'CNXFINANCE']
                symbol_columns = sorted([col for col in df.columns if col.lower() != 'date' and not col.lower().startswith('unnamed') and len(col.strip()) > 0])
                
                # Indexes
                for col in symbol_columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                for idx_sym in index_symbols:
                    if idx_sym in df.columns: index_data[idx_sym] = round(df[idx_sym].dropna().iloc[-1], 2)
                    
                
                display_symbols = symbol_columns
                GLOBAL_DASHBOARD_CACHE['all_symbols'] = display_symbols # Cache for dropdown
                scan_candidates = [s for s in display_symbols if s not in index_symbols]
                
                # Fetch Scan Prices (Batch)
                fetch_list = scan_candidates + index_symbols
                live_price_map = get_live_prices_batch(fetch_list)
                
                # Update Index Data
                for idx_sym in index_symbols:
                    if idx_sym in live_price_map: index_data[idx_sym] = live_price_map[idx_sym]
                    
                # Scan Logic (Direction-Based)
                opportunities = []
                for symbol in scan_candidates:
                    try:
                        sym_prices = df[symbol].dropna().round(2).unique()
                        if len(sym_prices) < 2: continue
                        latest_val = live_price_map.get(symbol)
                        prev_close = df[symbol].dropna().iloc[-1]
                        
                        if not latest_val: latest_val = prev_close
                        latest_val, prev_close = float(latest_val), float(prev_close)
                        
                        day_change_pct = ((latest_val - prev_close) / prev_close) * 100
                        
                        # Bullish
                        if day_change_pct > 0.05:
                            higher = sorted([p for p in sym_prices if p > latest_val])
                            if higher:
                                runway = ((higher[0] - latest_val)/latest_val)*100
                                if runway > 0.5: opportunities.append({'symbol': symbol, 'gap_up': round(runway, 2), 'gap_down': 0, 'bias': 'bull'})
                        # Bearish
                        elif day_change_pct < -0.05:
                            lower = sorted([p for p in sym_prices if p < latest_val], reverse=True)
                            if lower:
                                runway = ((latest_val - lower[0])/latest_val)*100
                                if runway > 0.5: opportunities.append({'symbol': symbol, 'gap_up': 0, 'gap_down': round(runway, 2), 'bias': 'bear'})
                    except: continue
                
                top_bullish = sorted([o for o in opportunities if o['bias'] == 'bull'], key=lambda x: x['gap_up'], reverse=True)[:10]
                top_bearish = sorted([o for o in opportunities if o['bias'] == 'bear'], key=lambda x: x['gap_down'], reverse=True)[:10]
                
                # NIFTY 50 Heatmap Logic (Cached Calculation)
                heatmap_list = []
                
                # We need live prices/changes for NIFTY 50
                # We reuse live_price_map if they are in it, else fetch
                missing_nifty = [s for s in NIFTY_50_SYMBOLS if s not in live_price_map]
                if missing_nifty:
                    extra_prices = get_live_prices_batch(missing_nifty)
                    live_price_map.update(extra_prices)
                
                # Fetch Changes via Quote if possible for accuracy, or fallback to CSV Prev Close
                nifty_quotes = {}
                if kite and kite.access_token:
                    try:
                        # Batch fetch quotes for Nifty 50 to get precise 'change' %
                        nifty_quotes = kite.quote([f"NSE:{s}" for s in NIFTY_50_SYMBOLS])
                    except: pass
                
                for sym in NIFTY_50_SYMBOLS:
                    lp = live_price_map.get(sym)
                    if not lp: continue
                    
                    change_pct = 0
                    prev_close = 0.0
                    
                    q = nifty_quotes.get(f"NSE:{sym}")
                    if q and 'ohlc' in q and q['ohlc']['close'] > 0:
                         prev_close = float(q['ohlc']['close'])
                         change_pct = ((lp - prev_close) / prev_close) * 100
                    else:
                        # Fallback to CSV prev close
                        try:
                            pc = df[sym].dropna().iloc[-1]
                            prev_close = float(pc)
                            change_pct = ((lp - prev_close) / prev_close) * 100
                        except: change_pct = 0
                    
                    heatmap_list.append({'symbol': sym, 'price': lp, 'change': round(change_pct, 2), 'prev_close': prev_close})
                
                heatmap_data = sorted(heatmap_list, key=lambda x: x['change'], reverse=True)

        except Exception as e: print(f"Scan Error: {e}")
        
        # update cache
        GLOBAL_DASHBOARD_CACHE['last_scan_time'] = current_time
        GLOBAL_DASHBOARD_CACHE['top_bullish'] = top_bullish
        GLOBAL_DASHBOARD_CACHE['top_bearish'] = top_bearish
        GLOBAL_DASHBOARD_CACHE['heatmap_data'] = heatmap_data
        GLOBAL_DASHBOARD_CACHE['heatmap_data'] = heatmap_data
        GLOBAL_DASHBOARD_CACHE['index_data'] = index_data # Store index baseline
        
        # Calculate Watchlist ADV (20-Day Average)
        try:
             # Clean symbols for YF
             ticker_map = {
                 'NIFTY': '^NSEI',
                 'BANKNIFTY': '^NSEBANK',
                 'SENSEX': '^BSESN',
                 'CNXFINANCE': 'NIFTY_FIN_SERVICE.NS'
             }
             yf_tickers = []
             for s in watchlist_symbols:
                 clean_s = s.strip()
                 if clean_s in ticker_map: yf_tickers.append(ticker_map[clean_s])
                 else: yf_tickers.append(f"{clean_s}.NS")
             
             if yf_tickers:
                 # Fetch 1 month to ensure we get 20 days
                 adv_data = yf.download(tickers=yf_tickers, period="1mo", progress=False)['Volume']
                 # Average of last 20 rows
                 mean_vol = adv_data.tail(20).mean()
                 # Map back to clean symbol (remove .NS and map index back)
                 adv_map = {}
                 rev_map = {v: k for k, v in ticker_map.items()}
                 
                 for col in mean_vol.index:
                     if col in rev_map: clean_sym = rev_map[col]
                     else: clean_sym = col.replace('.NS', '')
                     adv_map[clean_sym] = mean_vol[col]
                 GLOBAL_DASHBOARD_CACHE['curr_adv_map'] = adv_map
        except Exception as e: 
             print(f"ADV Fetch Error: {e}")
             GLOBAL_DASHBOARD_CACHE['curr_adv_map'] = {}

    # --- PHASE 2: PRICING UPDATE (Live, Every Call) ---
    # Retrieve Cached Lists
    cached_bullish = GLOBAL_DASHBOARD_CACHE['top_bullish']
    cached_bearish = GLOBAL_DASHBOARD_CACHE['top_bearish']
    cached_heatmap = GLOBAL_DASHBOARD_CACHE['heatmap_data']
    base_watchlist = load_watchlist()
    
    # Collect ALL symbols needed for display
    all_needed_syms = set()
    for o in cached_bullish: all_needed_syms.add(o['symbol'])
    for o in cached_bearish: all_needed_syms.add(o['symbol'])
    for h in cached_heatmap: all_needed_syms.add(h['symbol'])
    for s in base_watchlist: all_needed_syms.add(s)
    index_symbols = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'CNXFINANCE']
    for i in index_symbols: all_needed_syms.add(i)
    
    # Batch Fetch LIVE NOW (With Market Hours Logic)
    # global EOD_UPDATED_DATE (Moved to top)
    should_fetch = False
    
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz)
    
    # 1. Market Open?
    if is_market_open():
        should_fetch = True
        
    # 2. EOD Check (After 16:00, Once per day)
    elif now.hour >= 16:
        today_str = now.strftime('%Y-%m-%d')
        if EOD_UPDATED_DATE != today_str:
            print(f"DEBUG: Triggering EOD Sync for {today_str}")
            should_fetch = True
            EOD_UPDATED_DATE = today_str # Mark done
            
    live_prices_now = {}
    
    # DEBUG AUTH STATUS
    # print(f"DEBUG: Kite Object: {kite is not None}, Token: {kite.access_token if kite else 'None'}")
    
    if should_fetch:
        live_prices_now = get_live_prices_batch(list(all_needed_syms))
        GLOBAL_DASHBOARD_CACHE['last_prices'] = live_prices_now # Update Cache
    else:
        # Sleep Mode - Use Cache
        live_prices_now = GLOBAL_DASHBOARD_CACHE.get('last_prices', {})
        if not live_prices_now:
             # Try one fetch if cache empty even if closed? Or just empty.
             # Better to try once if empty so UI isn't blank on restart
             print("DEBUG: Cache empty in sleep mode, fetching once.")
             live_prices_now = get_live_prices_batch(list(all_needed_syms))
             GLOBAL_DASHBOARD_CACHE['last_prices'] = live_prices_now

    # Rebuild Return Objects with FRESH prices
    final_bullish = []
    for item in cached_bullish:
        lp = live_prices_now.get(item['symbol'])
        # We can re-calc change or gap here if needed, or just pass live price?
        # User wants live "ticks". We will return the object but maybe add 'current_price'?
        # For simplicity, let's keep the structure but update what we can.
        # Actually, the gap might change slightly. Let's strictly update price if possible.
        final_bullish.append(item) 
        
    final_bearish = [x for x in cached_bearish]
    
    # Heatmap Fresh Prices
    final_heatmap = []
    for item in cached_heatmap:
        new_item = item.copy()
        if item['symbol'] in live_prices_now: 
            new_lp = live_prices_now[item['symbol']]
            new_item['price'] = new_lp
            # Dynamic Recalc of % Change
            if 'prev_close' in item and item['prev_close'] > 0:
                new_item['change'] = round(((new_lp - item['prev_close']) / item['prev_close']) * 100, 2)
            
        final_heatmap.append(new_item)

    # Watchlist Fresh (Always Dynamic)
    watchlist_details = []
    inst_alerts = []
    
    # Need DF for reference close (can we cache DF? maybe heavy. Load efficiently or use cache?)
    # For now, load DF just for reference closes for Watchlist (or cache ref closes)
    # Optimization: Loading CSV every 3s is heavy. 
    # Let's rely on caching 'prev_close' map? 
    # For safety, let's do the standard reliable load but maybe optimize later. 
    # Actually, 3s CSV read is bad. 
    # Let's look if we can get change % from Zerodha quote directly?
    # Yes, quote['change'].
    
    # Faster Watchlist Construction using Zerodha 'change' if available
    # Fallback to simple calculation if no Zerodha
    
    # To keep it simple and robust matching previous logic:
    # We will build watchlist using live_prices_now and try to get Close from YF/Zerodha or fallback CSV
    # We will skip CSV load here if possible to keep it fast.
    
    scan_quotes = {}
    if kite and kite.access_token:
         try: scan_quotes = kite.quote([f"NSE:{s}" for s in base_watchlist])
         except: pass

    for i, sym in enumerate(base_watchlist):
        lp = live_prices_now.get(sym, 0)
        change_pct = 0
        
        # Try get change from Zerodha
        q = scan_quotes.get(f"NSE:{sym}")
        prev_close = 0.0
        
        if q and 'ohlc' in q and q['ohlc']['close'] > 0:
            prev_close = float(q['ohlc']['close'])
            change_pct = ((lp - prev_close) / prev_close) * 100
        elif lp > 0:
             # Fallback from CSV
             try:
                 if sym in df.columns:
                     pc = df[sym].dropna().iloc[-1]
                     prev_close = float(pc)
                     change_pct = ((lp - float(pc))/float(pc)) * 100
             except: pass
             
        # Alerts
        # Recalculate Volume alert? Maybe keep alerts cached too?
        # User wants "Instance" alerts. Let's run Vol check live as it is fast.
        vol = q.get('volume', 0) if q else 0
        ratio, msg = detect_intraday_spike(sym, vol)
        
        if ratio > 3: 
            # Get ADV from cache
            adv_map = GLOBAL_DASHBOARD_CACHE.get('curr_adv_map', {})
            adv = adv_map.get(sym, 0)
            
            # Format ADV
            adv_str = "N/A"
            if adv > 10000000: adv_str = f"{round(adv/10000000, 2)}Cr"
            elif adv > 100000: adv_str = f"{round(adv/100000, 2)}L"
            elif adv > 0: adv_str = f"{int(adv)}"
            
            inst_alerts.append({
                'title': f"★ {sym} VOL", 
                'msg': f"{msg} {ratio}x<br><span style='font-size:0.8em; color:#aaa;'>Avg Vol: {adv_str}</span>", 
                'type': 'vol'
            })
        
        watchlist_details.append({'symbol': sym, 'price': lp, 'change': round(change_pct, 2), 'orig_idx': i, 'prev_close': prev_close})
    
    watchlist_details = sorted(watchlist_details, key=lambda x: x['change'], reverse=True)
    all_changes = [abs(x['change']) for x in watchlist_details]
    max_val = max(all_changes) if all_changes else 0
    max_intensity = max_val if max_val > 0 else 1
    
    # Fresh Index Data
    final_index = {}
    for i in index_symbols: final_index[i] = live_prices_now.get(i, 0)

    return {
        'index_data': final_index,
        'top_bullish': final_bullish,
        'top_bearish': final_bearish,
        'heatmap': final_heatmap,
        'watchlist': watchlist_details,
        'inst_alerts': inst_alerts,
        'max_intensity': max_intensity,
        'kite_authenticated': True if (kite and kite.access_token) else False, # Check actual token validity
        'news_list': get_live_news(),
        'symbols': GLOBAL_DASHBOARD_CACHE.get('all_symbols', [])
    }

@app.route('/api/landing_data')
def landing_data_api():
    """API Endpoint for Live Dashboard Updates"""
    return jsonify(get_dashboard_data())

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main Landing Page"""
    data = get_dashboard_data()
    return render_template_string(LANDING_TEMPLATE, **data)

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

def get_intraday_data(symbol):
    intraday_data = []
    
    # Priority 1: Zerodha (Better live data if History API enabled)
    if kite and kite.access_token:
        try:
             # Use 1 minute candles for "live" feel
             token = get_instrument_token(symbol)
             if token:
                 to_date = datetime.datetime.now()
                 from_date = to_date - datetime.timedelta(days=1) # Just today/recent
                 records = kite.historical_data(token, from_date, to_date, "minute")
                 for r in records:
                     intraday_data.append({
                        'time': int(r['date'].timestamp()), 'open': r['open'], 'high': r['high'], 'low': r['low'], 'close': r['close']
                     })
                 if intraday_data: return intraday_data
        except: pass

    # Priority 2: YFinance (Fallback)
    try:
        ticker = f"{symbol}.NS"
        ticker_map = {'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'SENSEX': '^BSESN'}
        yf_ticker = ticker_map.get(symbol, ticker)
        # Switch to 1m for more granular updates
        idf = yf.download(yf_ticker, period="1d", interval="1m", progress=False)
        if not idf.empty:
            for idx, row_data in idf.iterrows():
                try:
                    intraday_data.append({
                        'time': int(idx.timestamp()), 'open': float(row_data['Open']),
                        'high': float(row_data['High']), 'low': float(row_data['Low']), 'close': float(row_data['Close'])
                    })
                except: pass
    except Exception: pass
    
    return intraday_data

def get_analysis_data(symbol):
    try:
        # 0. Fuzzy match symbol if needed (using CSV as master list for now)
        if os.path.exists(CSV_FILE):
             # Only do heavy CSV read if absolutely needed or cached
             pass

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
            intraday_data = get_intraday_data(symbol)
            
            # [LIVE PATCH] Synthesize the latest candle using Real-Time LTP
            # This ensures the chart ticks live even if historical API lags or is closed candle only
            if intraday_data and latest_price:
                last_candle = intraday_data[-1]
                last_ts = last_candle['time']
                
                # Current Minute Timestamp (Floored)
                now_ts = int(datetime.datetime.now().timestamp())
                current_minute_ts = now_ts - (now_ts % 60)
                
                # Scenario 1: We are in the same minute as the last candle -> Update it
                if last_ts == current_minute_ts:
                    last_candle['close'] = latest_price
                    if latest_price > last_candle['high']: last_candle['high'] = latest_price
                    if latest_price < last_candle['low']: last_candle['low'] = latest_price
                    
                # Scenario 2: New minute started -> Append new forming candle
                elif current_minute_ts > last_ts:
                    intraday_data.append({
                        'time': current_minute_ts,
                        'open': latest_price,
                        'high': latest_price,
                        'low': latest_price,
                        'close': latest_price
                    })

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
                except TokenException:
                    print("DEBUG: Zerodha Token Expired during market depth/options fetch.")
                    kite.access_token = None
                except Exception: pass

            kite_token = get_instrument_token(symbol if symbol not in ["NIFTY", "BANKNIFTY"] else (f"NIFTY 50" if symbol == "NIFTY" else "NIFTY BANK"))
            news_list = get_live_news(symbol)
            hist_news = get_historical_news(symbol)
            
            return {
                'row': s_data, 
                'history': history_cloud, 
                'intraday': intraday_data, 
                'news_list': news_list, 
                'hist_news': hist_news,
                'market_depth': market_depth,
                'options_data': options_data,
                'kite_token': kite_token,
                'kite_authenticated': True if (kite and kite.access_token) else False
            }
        return None
    except Exception as e:
        print(f"Analysis Data Error: {e}")
        return None

@app.route('/api/analysis_data/<symbol>')
def analysis_data_api(symbol):
    data = get_analysis_data(symbol)
    if data: return jsonify(data)
    return jsonify({'error': 'Symbol not found'}), 404

@app.route('/analysis/<symbol>')
def analysis(symbol):
    data = get_analysis_data(symbol)
    if data:
        return render_template_string(ANALYSIS_TEMPLATE, **data)
    return f"Symbol {symbol} not found in Cloud or Local Database.", 404

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

        /* TOAST NOTIFICATION */
        #toast-container { position: fixed; top: 20px; right: 20px; width: 350px; z-index: 10000; pointer-events: none; display: flex; flex-direction: column; gap: 10px; }
        .toast { background: rgba(30, 34, 45, 0.95); backdrop-filter: blur(10px); border-left: 5px solid #ffd700; color: #fff; padding: 15px; border-radius: 4px; box-shadow: 0 5px 20px rgba(0,0,0,0.5); transform: translateX(120%); transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); display: flex; align-items: start; gap: 10px; pointer-events: auto; }
        .toast.show { transform: translateX(0); }
        .toast-icon { font-size: 1.5em; }
        .toast-content { flex: 1; }
        .toast-title { font-weight: 900; font-size: 0.9em; margin-bottom: 5px; color: #ffd700; text-transform: uppercase; letter-spacing: 1px; }
        .toast-msg { font-size: 0.8em; color: #ccc; line-height: 1.4; }

        .refresh-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #444;
            color: #aaa;
            font-size: 0.7em;
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 10px;
            text-transform: uppercase;
            transition: all 0.2s;
        }
        .refresh-btn:hover { background: rgba(255, 255, 255, 0.2); color: #fff; border-color: #666; }
        .refresh-btn:active { transform: scale(0.95); }
    </style>
</head>
<body>
    <div id="toast-container"></div>
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
                <a href="/logout" onclick="return confirm('Disconnect from Zerodha API? This will stop live updates.');" style="text-decoration: none;">
                    <div id="connection-status-wrapper" style="background: rgba(3, 218, 198, 0.1); border: 1px solid var(--up-color); color: var(--up-color); padding: 5px 12px; border-radius: 20px; font-size: 0.7em; font-weight: bold; display: inline-flex; align-items: center; gap: 5px; cursor: pointer;">
                        <span style="height: 8px; width: 8px; background: var(--up-color); border-radius: 50%;"></span>
                        ZERODHA KITE CONNECTED (DISCONNECT)
                    </div>
                </a>
                {% else %}
                <div id="connection-status-wrapper">
                    <a href="/login" style="background: var(--accent-color); color: #000; padding: 8px 15px; border-radius: 5px; text-decoration: none; font-weight: bold; font-size: 0.8em; display: inline-block;">
                        ⚡ LOGIN TO ZERODHA
                    </a>
                </div>
                {% endif %}
            </div>
        </div>

        <div class="grid">
            <div class="sidebar">
                <h3>Indexes</h3>
                <div class="index-card" id="index-container">
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
                </form>

                <div class="opportunity-section" style="margin-top: 20px;">
                    <h2 style="color: var(--accent-color); font-size: 1.2em;">
                        💎 Watchlist
                        <button class="refresh-btn" onclick="manualRefresh(this)">Refresh</button>
                    </h2>
                    <p style="font-size: 0.7em; color: #666; margin-bottom: 10px;">L: Analyze | R: Edit</p>
                    <div class="op-grid" id="watchlist-container" style="grid-template-columns: repeat(2, 1fr); gap: 5px;">
                        {% for item in watchlist %}
                        {% set is_up = item.change >= 0 %}
                        {% set base_op = (item.change|abs / 3.0) %}
                        {% if base_op > 0.8 %}{% set base_op = 0.8 %}{% endif %}
                        {% if base_op < 0.2 %}{% set base_op = 0.2 %}{% endif %}
                        {% set bg = 'rgba(0, 200, 83, ' + base_op|string + ')' if is_up else 'rgba(213, 0, 0, ' + base_op|string + ')' %}
                        
                        <div class="op-card" data-index="{{ item.orig_idx }}" onclick="handleWatchlistClick(event, '{{ item.symbol }}')"
                             style="background: {{ bg }}; cursor: pointer; padding: 5px; border: 1px solid #222; min-height: 40px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                            <div class="wl-sym" style="font-weight: 900; font-size: 0.75em; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.8);">{{ item.symbol }}</div>
                            <div style="font-size: 0.65em; color: #fff; font-weight: bold; margin-top: 1px;">{{ item.change }}%</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class="main">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                    <div class="opportunity-section">
                        <h2 style="color: var(--up-color);">
                            🚀 Top Bullish Breakouts
                            <button class="refresh-btn" onclick="manualRefresh(this)">Refresh</button>
                        </h2>
                        <div class="op-grid" id="bullish-container">
                            {% for opt in top_bullish %}
                            <a href="/analysis/{{ opt.symbol }}" class="op-card" style="border-left: 4px solid var(--up-color);">
                                <div style="font-weight: bold; margin-bottom: 5px;">{{ opt.symbol }}</div>
                                <div style="font-size: 0.8em; color: #aaa;">Target Gap: <span style="color: var(--up-color); font-weight: bold;">{{ opt.gap_up }}%</span></div>
                            </a>
                            {% endfor %}
                        </div>
                    </div>

                    <div class="opportunity-section">
                        <h2 style="color: var(--down-color);">
                            📉 Top Bearish Vacuums
                            <button class="refresh-btn" onclick="manualRefresh(this)">Refresh</button>
                        </h2>
                        <div class="op-grid" id="bearish-container">
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
                        <span>📊 NIFTY 50 Heatmap</span>
                        <span id="next-scan-timer" style="font-size: 0.6em; background: #000; border: 1px solid #333; color: var(--accent-color); padding: 5px 10px; border-radius: 4px; display: inline-block; min-width: 120px; text-align: center;">NEXT SCAN: 05:00</span>
                        <button class="refresh-btn" onclick="manualRefresh(this)">Refresh</button>
                    </h2>
                    <div class="op-grid" id="heatmap-container" style="grid-template-columns: repeat(10, 1fr); gap: 5px;">
                        {% for item in heatmap %}
                        {% set is_up = item.change >= 0 %}
                        {% set base_op = (item.change|abs / 3.0) %}
                        {% if base_op > 0.8 %}{% set base_op = 0.8 %}{% endif %}
                        {% if base_op < 0.2 %}{% set base_op = 0.2 %}{% endif %}
                        {% set bg_col = 'rgba(0, 200, 83, ' + base_op|string + ')' if is_up else 'rgba(213, 0, 0, ' + base_op|string + ')' %}
                        
                        <div class="op-card" onclick="window.location.href='/analysis/{{ item.symbol }}'" 
                             style="background: {{ bg_col }}; cursor: pointer; padding: 10px; border: 1px solid #222; min-height: 50px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                            <div style="font-weight: 900; font-size: 0.8em; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.8);">{{ item.symbol }}</div>
                            <div style="font-size: 0.7em; color: #fff; font-weight: bold; margin-top: 2px;">{{ item.change }}%</div>
                            <div style="font-size: 0.6em; color: rgba(255,255,255,0.7);">{{ item.price }}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>


            </div>
        </div>
    </div>

    <script>
        window.isEditing = false;

        // --- NEW: EVENT DELEGATION FOR WATCHLIST ---
        document.addEventListener('DOMContentLoaded', () => {
            const wlContainer = document.getElementById('watchlist-container');
            if (wlContainer) {
                wlContainer.addEventListener('contextmenu', function(e) {
                    const card = e.target.closest('.op-card');
                    if (card) {
                        e.preventDefault(); // Stop context menu
                        e.stopPropagation();
                        
                        console.log("Right-Click Detected on card:", card);
                        const index = card.getAttribute('data-index');
                        startEditing(card, index);
                    }
                });
            }
        });

        function handleWatchlistClick(e, symbol) {
            if (window.isEditing) {
                e.preventDefault();
                return;
            }
            window.location.href = '/analysis/' + symbol;
        }

        function startEditing(el, index) {
            console.log("Starting Edit Mode for Index:", index);
            
            // Visual feedback
            el.style.backgroundColor = '#222';
            el.style.border = '1px solid var(--accent-color)';
            
            const currentSymEl = el.querySelector('.wl-sym');
            if(!currentSymEl) return; // Already editing?
            
            const currentSym = currentSymEl.innerText.trim();
            window.isEditing = true; // Block live updates
            
            // Create Input
            const input = document.createElement('input');
            input.value = currentSym;
            input.style.width = '100%';
            input.style.textAlign = 'center';
            input.style.background = '#000';
            input.style.color = '#fff';
            input.style.border = 'none';
            input.style.fontSize = '1em';
            input.style.fontWeight = 'bold';
            
            el.innerHTML = ''; // Clear card content
            el.appendChild(input);
            input.focus();
            
            // Cleanup on blur or enter
            const finish = async (save) => {
                if (save) {
                    const newSym = input.value.toUpperCase().trim();
                    if (newSym && newSym !== currentSym) {
                        try {
                            const response = await fetch('/update_watchlist', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({index: parseInt(index), symbol: newSym})
                            });
                        } catch(e) { console.error(e); }
                    }
                }
                window.location.reload(); // Always reload to reset state
            };

            // Delay onblur binding slightly to avoid immediate trigger
            setTimeout(() => {
                input.onblur = () => finish(false);
            }, 500);

            input.onkeydown = (e) => {
                if(e.key === 'Enter') finish(true);
                if(e.key === 'Escape') finish(false);
            };
        }

        // --- LIVE POLLING SYSTEM ---
        // --- LIVE POLLING SYSTEM ---
        // --- LIVE POLLING SYSTEM ---
        // --- LIVE POLLING SYSTEM ---
        async function fetchDashboardData() {
            if (window.isEditing) return; // Skip update if user is editing
            try {
                const res = await fetch('/api/landing_data');
                if (!res.ok) return;
                const data = await res.json();
                if (window.isEditing) return; // Race condition fix: check again before updating DOM
                updateDashboard(data);
            } catch (e) { console.error("Update failed", e); }
        }

        async function manualRefresh(btn) {
            if (!btn) return;
            const originalText = btn.innerText;
            btn.innerText = "Refreshing...";
            btn.style.opacity = "0.7";
            btn.disabled = true;
            
            await fetchDashboardData();
            
            setTimeout(() => {
                btn.innerText = originalText;
                btn.style.opacity = "1";
                btn.disabled = false;
            }, 500);
        }
        
        // --- TOAST NOTIFICATIONS ---
        function showToast(alert) {
            const container = document.getElementById('toast-container');
            const el = document.createElement('div');
            el.className = 'toast';
            el.innerHTML = `
                <div class="toast-icon">${alert.type === 'vol' ? '🚀' : '🎯'}</div>
                <div class="toast-content">
                    <div class="toast-title">${alert.title}</div>
                    <div class="toast-msg">${alert.msg}</div>
                </div>
            `;
            container.appendChild(el);
            
            // Trigger animation
            setTimeout(() => el.classList.add('show'), 100);
            
            // Remove after 6s
            setTimeout(() => {
                el.classList.remove('show');
                setTimeout(() => el.remove(), 500);
            }, 6000);
        }

        function updateDashboard(data) {
           // Update Indexes
           const idxContainer = document.getElementById('index-container');
           if (idxContainer) {
               let html = '';
               for (const [k, v] of Object.entries(data.index_data)) {
                   html += `<div class="index-item"><span>${k}</span> <span class="index-val">${v}</span></div>`;
               }
               idxContainer.innerHTML = html;
           }

           // Update Bullish
           updateOpGrid('bullish-container', data.top_bullish, 'var(--up-color)', 'Target Gap', 'gap_up');
           
           // Update Bearish
           updateOpGrid('bearish-container', data.top_bearish, 'var(--down-color)', 'Support Gap', 'gap_down');
           
           // Update Heatmap
           const hmContainer = document.getElementById('heatmap-container');
           if (hmContainer && data.heatmap) {
               let html = '';
               data.heatmap.forEach(item => {
                   let is_up = item.change >= 0;
                   let base_op = (Math.abs(item.change) / 3.0);
                   if (base_op > 0.8) base_op = 0.8;
                   if (base_op < 0.2) base_op = 0.2;
                   let bg_col = is_up ? `rgba(0, 200, 83, ${base_op})` : `rgba(213, 0, 0, ${base_op})`;

                   html += `
                    <div class="op-card" onclick="window.location.href='/analysis/${item.symbol}'" 
                         style="background: ${bg_col}; cursor: pointer; padding: 10px; border: 1px solid #222; min-height: 50px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                        <div style="font-weight: 900; font-size: 0.8em; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.8);">${item.symbol}</div>
                        <div style="font-size: 0.7em; color: #fff; font-weight: bold; margin-top: 2px;">${item.change}%</div>
                        <div style="font-size: 0.6em; color: rgba(255,255,255,0.7);">${item.price}</div>
                    </div>`;
               });
               hmContainer.innerHTML = html;
           }

           // Trigger Institutional Alerts
           if (data.inst_alerts && data.inst_alerts.length > 0) {
               // Stagger toasts to avoid overwhelming
               data.inst_alerts.slice(0, 3).forEach((alert, idx) => {
                   setTimeout(() => showToast(alert), idx * 1500);
               });
           }
           
           // Update Watchlist
           const wlContainer = document.getElementById('watchlist-container');
           if (wlContainer && data.watchlist) {
               let html = '';
               data.watchlist.forEach(item => {
                   let is_up = item.change >= 0;
                   let base_op = (Math.abs(item.change) / 3.0);
                   if (base_op > 0.8) base_op = 0.8;
                   if (base_op < 0.2) base_op = 0.2;
                   let bg = is_up ? `rgba(0, 200, 83, ${base_op})` : `rgba(213, 0, 0, ${base_op})`;
                   
                   html += `
                    <div class="op-card" data-index="${item.orig_idx}" onclick="handleWatchlistClick(event, '${item.symbol}')"
                         style="background: ${bg}; cursor: pointer; padding: 5px; border: 1px solid #222; min-height: 40px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                        <div class="wl-sym" style="font-weight: 900; font-size: 0.75em; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.8);">${item.symbol}</div>
                        <div style="font-size: 0.65em; color: #fff; font-weight: bold; margin-top: 1px;">${item.change}%</div>
                    </div>`;
               });
               wlContainer.innerHTML = html;
           }

            // Update Connection Status Indicator
            const connWrapper = document.getElementById('connection-status-wrapper');
            if (connWrapper) {
                 // console.log("DEBUG: Auth State:", data.kite_authenticated); // Debug
                 if (data.kite_authenticated) {
                     // State: CONNECTED
                     // Ensure we have the wrapper structure for connected state (inside <a>)
                     // Safest way: Replace the PARENT if it's the anchor, or the div itself?
                     // Let's replace the element entirely to match the template structure.
                     
                     // We need to match the template's structure to avoid drift.
                     // Template: 
                     // Connected: <a href="/logout"><div id="wrapper">...</div></a>
                     // Disconnected: <div id="wrapper"><a href="/login">...</a></div>
                     
                     // If we are currently disconnected (Div wrapper), we need to wrap it in an anchor?
                     // Or easier: Just put the Anchor INSIDE the wrapper for both, but styled differently?
                     // Current Template structure is asymmetric. JS needs to handle swap.
                     
                     // Strategy: Always render the "Connected" badge as an A tag, and "Disconnected" as a Div with A tag.
                     // We find the 'container'. The container is the parent of connWrapper.
                     let parent = connWrapper.parentElement;
                     let isParentAnchor = parent.tagName === 'A';
                     
                     if (isParentAnchor) {
                         // Already an anchor (Connected State), just update content text
                         connWrapper.innerHTML = '<span class="status-badge" style="background-color: #00c853; color: black; box-shadow: 0 0 10px #00c853;">ZERODHA CONNECTED (CLICK TO DISCONNECT)</span>';
                     } else {
                         // Currently Disconnected (Div is child of container), switch to Connected
                         // We need to replace the DIV with the ANCHOR-wrapped DIV.
                         connWrapper.outerHTML = `
                         <a href="/logout" onclick="return confirm('Disconnect from Zerodha API?');" style="text-decoration: none;">
                            <div id="connection-status-wrapper" style="background: rgba(3, 218, 198, 0.1); border: 1px solid var(--up-color); color: var(--up-color); padding: 5px 12px; border-radius: 20px; font-size: 0.7em; font-weight: bold; display: inline-flex; align-items: center; gap: 5px; cursor: pointer;">
                                <span style="height: 8px; width: 8px; background: var(--up-color); border-radius: 50%;"></span>
                                ZERODHA CONNECTED (CLICK TO DISCONNECT)
                            </div>
                         </a>`;
                     }

                 } else {
                     // State: DISCONNECTED
                     let parent = connWrapper.parentElement;
                     let isParentAnchor = parent.tagName === 'A';
                     
                     if (isParentAnchor) {
                         // Currently Connected (Anchor wrapper), switch to Disconnected (Div wrapper)
                         // Replace the PARENT (Anchor) with the DIV
                         parent.outerHTML = `
                         <div id="connection-status-wrapper">
                            <a href="/login" class="zerodha-btn blink-red" style="text-decoration:none; color:white;">LOGIN TO ZERODHA</a>
                         </div>`;
                     } else {
                         // Already Disconnected, just ensure content
                         connWrapper.innerHTML = '<a href="/login" class="zerodha-btn blink-red" style="text-decoration:none; color:white;">LOGIN TO ZERODHA</a>';
                     }
                 }
            }
        }
        
        function updateOpGrid(id, items, colorVar, label, key) {
            const container = document.getElementById(id);
            if (!container) return;
            let html = '';
            items.forEach(opt => {
                html += `
                <a href="/analysis/${opt.symbol}" class="op-card" style="border-left: 4px solid ${colorVar};">
                    <div style="font-weight: bold; margin-bottom: 5px;">${opt.symbol}</div>
                    <div style="font-size: 0.8em; color: #aaa;">${label}: <span style="color: ${colorVar}; font-weight: bold;">${opt[key]}%</span></div>
                </a>`;
            });
            container.innerHTML = html;
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

        // --- TIMER & POLLING LOGIC ---
        let scanTimer = 300; // 5 minutes in seconds
        const SCAN_INTERVAL = 300; 

        function updateTimerDisplay() {
            const minutes = Math.floor(scanTimer / 60);
            const seconds = scanTimer % 60;
            const text = `NEXT SCAN: ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            const timerEl = document.getElementById('next-scan-timer');
            if (timerEl) {
                timerEl.innerText = text;
                // Visual urgency
                if (scanTimer < 10) timerEl.style.color = '#ff5722';
                else timerEl.style.color = 'var(--accent-color)';
            }
        }

        function startLiveUpdates() {
            console.log("Starting 5-Minute Dashboard Updates...");
            
            // Initial Fetch
            fetchDashboardData();
            
            // Countdown Ticker (Every 1 second)
            // 1. Live Data Polling (Every 3 seconds) - Restored for Live Prices
            setInterval(fetchDashboardData, 3000);
            
            // 2. Visual Countdown Timer (Every 1 second) - Just for UI
            setInterval(() => {
                scanTimer--;
                if (scanTimer < 0) {
                    scanTimer = SCAN_INTERVAL;
                    // No need to force fetch here, the 3s poll will pick up the 'State Change' from backend naturally
                }
                updateTimerDisplay();
            }, 1000);
        }

        startLiveUpdates();
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
        .vmf-chart-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 8; pointer-events: none; }
        
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

        /* Connection Blink */
        @keyframes blinker { 50% { opacity: 0; } }
        .blink-red { animation: blinker 1s linear infinite; background-color: #d50000 !important; color: white !important; border: 1px solid #ff1744 !important; }
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
                <p style="margin: 0; color: #888;">LTP: <span id="ltp-display" style="color: var(--ltp-color); font-weight: bold;">{{ row.latest }}</span></p>
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
        <table id="ladder-table">
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
                        
                        const devZ = 1;
                        const vac = document.createElement('div');
                        vac.style.cssText = `position: absolute; width: 100%; top: ${prevTop}%; height: ${height}%; 
                                            background: linear-gradient(180deg, rgba(${rColor}, ${glowIntense}) 0%, rgba(${rColor}, 0.05) 50%, rgba(${rColor}, ${glowIntense}) 100%); 
                                            border-left: 6px solid rgba(${rColor}, 0.8); transition: opacity 0.3s; z-index: ${devZ};`;
                        
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

        // --- LIVE ANALYSIS UPDATES ---
        function startLiveAnalysisUpdates() {
            setInterval(async () => {
                try {
                    const res = await fetch(`/api/analysis_data/{{ row.symbol }}`);
                    if (!res.ok) return;
                    const data = await res.json();
                    updateAnalysisUI(data);
                } catch (e) { console.error("Analysis update failed", e); }
            }, 3000);
        }

        function updateAnalysisUI(data) {
            // Update LTP
            const ltpEl = document.getElementById('ltp-display');
            if (ltpEl) ltpEl.innerText = data.row.latest;

            // Update Globals
            vmfData.latest = data.row.latest;
            vmfData.clusters = data.row.clusters;
            vmfData.min = data.row.min_p;
            vmfData.max = data.row.max_p;
            vmfData.intraday = data.intraday; // Update candle data

            // 1. Update VMF Chart (Candles)
            if (vmfChartObj && data.intraday.length > 0) {
                 const series = vmfChartObj.series()[0]; // Assuming only one series (candlestick)
                 series.setData(data.intraday);
            }

            // 2. Update VMF Render (Flow/Vacuum)
            renderVMF();

            // 3. Update Ladder Table
            const ladderTable = document.getElementById('ladder-table');
            if(ladderTable) {
                // Re-generate table rows - simpler to replace body
                let tbodyHtml = '<tr>';
                
                // P-5 to P-1
                for(let i=0; i<5; i++) {
                    let p_idx = i;
                    let gap = (data.row.gaps && data.row.gaps[p_idx]) ? data.row.gaps[p_idx] : 0;
                    let price = data.row.prices[p_idx] ? data.row.prices[p_idx] : '-';
                    let bg = `rgba(207, 102, 121, ${0.15 + (Math.abs(gap) * 0.5)})`;
                    
                    tbodyHtml += `
                    <td style="background: ${bg};">
                        ${i === 4 ? '<div class="target-tag">SUP</div>' : ''}
                        <div style="font-weight: bold;">${price}</div>
                        <div style="font-size: 0.7em; opacity: 0.6;">${gap.toFixed(2)}%</div>
                    </td>`;
                }
                
                // LTP Cell
                tbodyHtml += `<td class="ltp-cell">${data.row.latest}</td>`;

                // P+1 to P+5
                for(let i=6; i<11; i++) {
                    let p_idx = i;
                    let gap = (data.row.gaps && data.row.gaps[p_idx-1]) ? data.row.gaps[p_idx-1] : 0;
                     let price = data.row.prices[p_idx] ? data.row.prices[p_idx] : '-';
                     let bg = `rgba(3, 218, 198, ${0.15 + (Math.abs(gap) * 0.5)})`;
                     
                     tbodyHtml += `
                    <td style="background: ${bg};">
                        ${i === 6 ? '<div class="target-tag">RES</div>' : ''}
                        <div style="font-weight: bold;">${price}</div>
                        <div style="font-size: 0.7em; opacity: 0.6;">${gap.toFixed(2)}%</div>
                    </td>`;
                }
                tbodyHtml += '</tr>';
                ladderTable.querySelector('tbody').innerHTML = tbodyHtml;
            }
        }
        
        startLiveAnalysisUpdates();
    </script>
    <!-- TradingView Widget END -->
</body>
</html>
"""


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
