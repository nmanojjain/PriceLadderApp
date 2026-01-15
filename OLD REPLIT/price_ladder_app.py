import pandas as pd
import os
import datetime
import pytz
import yfinance as yf
from flask import Flask, render_template_string, request

CSV_FILE = r"master_nse_data.csv"

app = Flask(__name__)

def get_live_prices_batch(symbols):
    live_prices = {}
    if not symbols:
        return live_prices
        
    try:
        # Map common indices to yfinance tickers
        ticker_map = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'CNXFINANCE': 'NIFTY_FIN_SERVICE.NS'
        }
        
        yf_tickers = []
        for sym in symbols:
            yf_tickers.append(ticker_map.get(sym, f"{sym}.NS"))
            
        print(f"DEBUG: Fetching live prices for: {yf_tickers}")
        
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

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_symbols = []
    heatmap_rows = []
    index_data = {}

    sort_by = request.form.get('sort_by', 'default') # default, max_gap_up, max_gap_down, ltp_p1, ltp_m1
    view_mode = request.form.get('view_mode', 'eod')
    
    # Load Data
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, low_memory=False)
            # Clean column names
            df.columns = [col.strip() for col in df.columns]
            
            # Numeric conversion
            index_symbols = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'CNXFINANCE']
            symbol_columns = sorted([col for col in df.columns if col.lower() != 'date' and not col.lower().startswith('unnamed') and len(col.strip()) > 0])
            
            for col in symbol_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Index Data
            for index_symbol in index_symbols:
                if index_symbol in df.columns:
                    latest_val = df[index_symbol].dropna().iloc[-1]
                    index_data[index_symbol] = round(latest_val, 2)

            display_symbols = [s for s in symbol_columns if s not in index_symbols]
        else:
            display_symbols = []
            print("CSV file not found.")

        if request.method == 'POST':
            selected_symbols = request.form.getlist('symbols')
            
            # Fetch live prices in batch if needed
            live_prices_map = {}
            if view_mode == 'live' and selected_symbols:
                live_prices_map = get_live_prices_batch(selected_symbols)
            
            for symbol in selected_symbols:
                # Get historical prices for ladder
                prices = df[symbol].dropna().round(2).unique()
                prices = sorted(prices)
                
                latest_csv_price = df[symbol].dropna().iloc[-1].round(2)
                
                live_price = live_prices_map.get(symbol)
                
                latest_price = round(live_price, 2) if live_price else latest_csv_price
                
                # Create Ladder
                # 10 levels below
                lower_prices = sorted([p for p in prices if p < latest_price], reverse=True)[:10]
                # 10 levels above
                higher_prices = sorted([p for p in prices if p > latest_price])[:10]
                
                padded_low = [None] * (10 - len(lower_prices)) + list(reversed(lower_prices))
                padded_high = higher_prices + [None] * (10 - len(higher_prices))
                
                # Full ladder: P-10 ... P-1, LTP, P+1 ... P+10
                ladder_prices = padded_low + [latest_price] + padded_high
                
                # Calculate Gaps
                gaps = []
                max_gap_up = 0
                max_gap_down = 0
                max_gap_up_idx = -1
                max_gap_down_idx = -1
                
                # Indices in ladder_prices:
                # 0-9: Lowers (P-10 to P-1)
                # 10: LTP
                # 11-20: Highers (P+1 to P+10)
                
                for i in range(len(ladder_prices) - 1):
                    curr = ladder_prices[i]
                    next_val = ladder_prices[i+1]
                    
                    gap_pct = 0
                    if curr is not None and next_val is not None and curr != 0:
                        gap_pct = ((next_val - curr) / curr) * 100
                    
                    gaps.append(gap_pct)
                    
                    # Check for max gaps
                    # i goes from 0 to 19. 
                    # LTP is at index 10.
                    # Gaps above LTP: indices 10 to 19 (gap between 10&11, 11&12 ... 19&20)
                    # Gaps below LTP: indices 0 to 9 (gap between 0&1 ... 9&10)
                    
                    if i >= 10: # Above LTP
                        if gap_pct > max_gap_up:
                            max_gap_up = gap_pct
                            max_gap_up_idx = i # Gap starts at i (lower bound of gap)
                    else: # Below LTP
                        if gap_pct > max_gap_down:
                            max_gap_down = gap_pct
                            max_gap_down_idx = i

                # Calculate immediate gaps from LTP
                # LTP is at index 10
                # P+1 is at index 11
                # P-1 is at index 9
                
                gap_ltp_p1 = 0
                if len(ladder_prices) > 11 and ladder_prices[11] is not None and latest_price != 0:
                     gap_ltp_p1 = ((ladder_prices[11] - latest_price) / latest_price) * 100
                
                gap_ltp_m1 = 0
                if len(ladder_prices) > 9 and ladder_prices[9] is not None and latest_price != 0:
                     gap_ltp_m1 = ((ladder_prices[9] - latest_price) / latest_price) * 100

                # Calculate max gap in the row for normalization
                row_max_gap = max([abs(g) for g in gaps if g is not None] + [0.001]) # Avoid div by zero

                row = {
                    'symbol': symbol,
                    'prices': ladder_prices,
                    'latest': latest_price,
                    'gaps': gaps,
                    'row_max_gap': row_max_gap,
                    'max_gap_up': round(max_gap_up, 2),
                    'max_gap_down': round(max_gap_down, 2),
                    'max_gap_up_idx': max_gap_up_idx,
                    'max_gap_down_idx': max_gap_down_idx,
                    'gap_ltp_p1': round(gap_ltp_p1, 2),
                    'gap_ltp_m1': round(gap_ltp_m1, 2)
                }
                heatmap_rows.append(row)

            # Sorting
            if sort_by == 'max_gap_up':
                heatmap_rows.sort(key=lambda x: x['max_gap_up'], reverse=True)
            elif sort_by == 'max_gap_down':
                heatmap_rows.sort(key=lambda x: x['max_gap_down'], reverse=True)
            elif sort_by == 'ltp_p1':
                heatmap_rows.sort(key=lambda x: x['gap_ltp_p1'], reverse=True)
            elif sort_by == 'ltp_m1':
                heatmap_rows.sort(key=lambda x: x['gap_ltp_m1']) 
                
    except Exception as e:
        print(f"Error processing data: {e}")

    return render_template_string(HTML_TEMPLATE, 
                                  symbols=display_symbols, 
                                  selected=selected_symbols, 
                                  heatmap_rows=heatmap_rows, 
                                  index_data=index_data, 
                                  view_mode=view_mode,
                                  sort_by=sort_by)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Price Ladder Pro</title>
    <style>
        :root {
            --bg-color: #121212;
            --text-color: #e0e0e0;
            --sidebar-bg: #1e1e1e;
            --card-bg: #2c2c2c;
            --accent-color: #bb86fc;
            --up-color: #03dac6;
            --down-color: #cf6679;
            --highlight-bg: #3700b3;
            --border-color: #333;
        }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; height: 100vh; background-color: var(--bg-color); color: var(--text-color); }
        .sidebar {
            width: 280px;
            background: var(--sidebar-bg);
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .content {
            flex: 1;
            padding: 20px;
            overflow: auto;
        }
        h2, h3 { margin-top: 0; color: var(--accent-color); }
        select, button, input {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            background: var(--card-bg);
            color: var(--text-color);
            border: 1px solid #444;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button { cursor: pointer; background: var(--accent-color); color: #000; font-weight: bold; border: none; transition: opacity 0.2s; }
        button:hover { opacity: 0.9; }
        
        .index-card {
            background: var(--card-bg);
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .index-item { display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: 500; }
        
        table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 20px; font-size: 0.85em; }
        th, td { padding: 8px; text-align: center; border-bottom: 1px solid var(--border-color); border-right: 1px solid var(--border-color); position: relative; }
        th { background: var(--card-bg); position: sticky; top: 0; z-index: 10; font-weight: 600; border-top: 1px solid var(--border-color); }
        th:first-child, td:first-child { border-left: 1px solid var(--border-color); position: sticky; left: 0; background: var(--bg-color); z-index: 11; }
        th:first-child { z-index: 12; background: var(--card-bg); }
        
        .latest-price { background: #ffd700; color: #000; font-weight: bold; border: 2px solid #fff; }
        
        .gap-bar-container {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: rgba(255,255,255,0.1);
        }
        .gap-bar {
            height: 100%;
            transition: width 0.3s;
        }
        .gap-bar.up { background-color: var(--up-color); }
        .gap-bar.down { background-color: var(--down-color); }
        
        .cell-content { position: relative; z-index: 2; }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-color); }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }

    </style>
    <script>
        function autoRefresh() {
            const liveSelected = document.querySelector('input[name="view_mode"]:checked')?.value === 'live';
            if (liveSelected) {
                setTimeout(() => {
                    document.querySelector('form').submit();
                }, 30000);  // Refresh every 30 seconds
            }
        }
        window.onload = autoRefresh;
    </script>
</head>
<body>
    <div class="sidebar">
        <h3>Market Dashboard</h3>
        <div class="index-card">
            {% for k,v in index_data.items() %}
                <div class="index-item"><span>{{k}}</span> <span>{{v}}</span></div>
            {% endfor %}
        </div>
        
        <form method="post">
            <label>Select Stocks (Hold Ctrl)</label>
            <select name="symbols" multiple size="15">
                {% for sym in symbols %}
                    <option value="{{ sym }}" {% if sym in selected %}selected{% endif %}>{{ sym }}</option>
                {% endfor %}
            </select>
            
            <label style="margin-top: 15px; display: block;">Mode</label>
            <div style="display: flex; gap: 10px;">
                <label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
                    <input type="radio" name="view_mode" value="eod" {% if view_mode != 'live' %}checked{% endif %} style="width: auto; margin: 0;"> EOD
                </label>
                <label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
                    <input type="radio" name="view_mode" value="live" {% if view_mode == 'live' %}checked{% endif %} style="width: auto; margin: 0;"> Live
                </label>
            </div>
            
            <label style="margin-top: 15px; display: block;">Sort Opportunities</label>
            <select name="sort_by">
                <option value="default" {% if sort_by == 'default' %}selected{% endif %}>Default</option>
                <option value="max_gap_up" {% if sort_by == 'max_gap_up' %}selected{% endif %}>Max Gap Up (Bullish)</option>
                <option value="max_gap_down" {% if sort_by == 'max_gap_down' %}selected{% endif %}>Max Gap Down (Bearish)</option>
                <option value="ltp_p1" {% if sort_by == 'ltp_p1' %}selected{% endif %}>LTP -> P+1 % (Bullish)</option>
                <option value="ltp_m1" {% if sort_by == 'ltp_m1' %}selected{% endif %}>LTP -> P-1 % (Bearish)</option>
            </select>
            
            <button type="submit" style="margin-top: 20px;">Analyze Ladder</button>
        </form>
    </div>
    
    <div class="content">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2>Price Ladder Analysis</h2>
            <div style="font-size: 0.9em; color: #888;">
                <span style="color: var(--up-color);">●</span> Gap Up Intensity &nbsp;
                <span style="color: var(--down-color);">●</span> Gap Down Intensity
            </div>
        </div>
        
        {% if heatmap_rows %}
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Max Gap Up</th>
                    <th>Max Gap Down</th>
                    {% for i in range(-10, 0) %}<th>P{{i}}</th>{% endfor %}
                    <th>LTP</th>
                    {% for i in range(1, 11) %}<th>P+{{i}}</th>{% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in heatmap_rows %}
                <tr>
                    <td style="font-weight: bold; color: var(--accent-color);">{{ row.symbol }}</td>
                    <td style="color: var(--up-color); font-weight: bold;">{{ row.max_gap_up }}%</td>
                    <td style="color: var(--down-color); font-weight: bold;">{{ row.max_gap_down }}%</td>
                    
                    {# Lower Prices P-10 to P-1 #}
                    {% for i in range(0, 10) %}
                        {% set gap_val = row.gaps[i] if row.gaps[i] is defined else 0 %}
                        {% set intensity = (gap_val | abs / row.row_max_gap * 100) if row.row_max_gap > 0 else 0 %}
                        <td style="background: rgba(207, 102, 121, {{ intensity * 0.005 }});">
                            <div class="cell-content">{{ row.prices[i] if row.prices[i] else '-' }}</div>
                            {% if gap_val != 0 %}
                                <div class="gap-bar-container">
                                    <div class="gap-bar down" style="width: {{ intensity }}%;"></div>
                                </div>
                            {% endif %}
                        </td>
                    {% endfor %}
                    
                    {# LTP #}
                    <td class="latest-price">
                        {{ row.latest }}
                    </td>
                    
                    {# Higher Prices P+1 to P+10 #}
                    {% for i in range(11, 21) %}
                        {% set gap_idx = i - 1 %}
                        {% set gap_val = row.gaps[gap_idx] if row.gaps[gap_idx] is defined else 0 %}
                        {% set intensity = (gap_val | abs / row.row_max_gap * 100) if row.row_max_gap > 0 else 0 %}
                        
                        <td style="background: rgba(3, 218, 198, {{ intensity * 0.005 }});">
                            <div class="cell-content">{{ row.prices[i] if row.prices[i] else '-' }}</div>
                            {% if gap_val != 0 %}
                                <div class="gap-bar-container">
                                    <div class="gap-bar up" style="width: {{ intensity }}%;"></div>
                                </div>
                            {% endif %}
                        </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
            <div style="text-align: center; margin-top: 50px; color: #666;">
                <h3>Ready to Analyze</h3>
                <p>Select stocks from the sidebar and click "Analyze Ladder" to generate the heatmap.</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
