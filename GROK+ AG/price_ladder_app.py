import pandas as pd
import os
import numpy as np
from flask import Flask, render_template_string, request
from test_live_batch import get_live_prices_batch  # Assume updated version

CSV_FILE = "master_nse_data.csv"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_symbols = []
    ladder_data = []
    best_stock = None
    index_data = {}
    historical_avg_abs_move = None

    sort_by = request.form.get('sort_by', 'default')
    view_mode = request.form.get('view_mode', 'eod')
    
    # Load Data
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, parse_dates=['Date'], low_memory=False)
            df.columns = [col.strip() for col in df.columns]
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            index_symbols = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'CNXFINANCE']
            symbol_columns = [col for col in df.columns if col != 'Date']
            display_symbols = [s for s in symbol_columns if s not in index_symbols]
            
            # Index Data
            for index_symbol in index_symbols:
                if index_symbol in df.columns:
                    latest_val = df[index_symbol].dropna().iloc[-1]
                    index_data[index_symbol] = round(latest_val, 2)
            
            # Historical % moves for best stock
            df_pct = df[symbol_columns].pct_change() * 100
            historical_avg_abs_move = df_pct.abs().mean().sort_values(ascending=False)
            best_stock = historical_avg_abs_move.index[0]
        else:
            display_symbols = []
    except Exception as e:
        print(f"Error loading CSV: {e}")
        display_symbols = []

    if request.method == 'POST':
        selected_symbols = request.form.getlist('symbols')
        
        # Fetch live prices if live mode
        live_prices_map = {}
        if view_mode == 'live' and selected_symbols:
            live_prices_map = get_live_prices_batch(selected_symbols)
        
        # Generate ladders
        for symbol in selected_symbols:
            if symbol in df.columns:
                latest_csv_price = df[symbol].dropna().iloc[-1].round(2)
                live_price = live_prices_map.get(symbol)
                base_price = round(live_price, 2) if live_price else latest_csv_price
                
                # % Ladder (as per your preference)
                step_size = 0.5  # Adjustable
                ladder_levels = np.arange(-5, 5.5, step_size)
                price_ladder = base_price * (1 + ladder_levels / 100)
                
                # Historical unique prices for gaps (keep your original idea)
                historical_prices = df[symbol].dropna().unique()
                lower_prices = sorted([p for p in historical_prices if p < base_price], reverse=True)[:10]
                higher_prices = sorted([p for p in historical_prices if p > base_price])[:10]
                
                # Gaps (your code refined)
                all_levels = lower_prices[::-1] + [base_price] + higher_prices
                gaps = [round(((all_levels[i+1] - all_levels[i]) / all_levels[i] * 100), 2) if all_levels[i+1] is not None else 0 for i in range(len(all_levels)-1)]
                max_gap_up = max([g for g in gaps if g > 0], default=0)
                max_gap_down = min([g for g in gaps if g < 0], default=0)
                
                ladder_data.append({
                    'symbol': symbol,
                    'base_price': base_price,
                    'ladder_levels': ladder_levels,
                    'price_ladder': price_ladder.round(2),
                    'gaps': gaps,
                    'max_gap_up': max_gap_up,
                    'max_gap_down': max_gap_down,
                    'ltp_p1': gaps[len(lower_prices)] if len(lower_prices) == 10 else 0,  # LTP to P+1
                    'ltp_m1': gaps[len(lower_prices)-1] if len(lower_prices) > 0 else 0  # LTP to P-1
                })
        
        # Sort if needed
        if sort_by != 'default':
            ladder_data.sort(key=lambda x: x[sort_by], reverse=(sort_by in ['max_gap_up', 'ltp_p1']))  # Desc for bullish, asc for bearish

    # HTML template (updated with % ladder column)
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Price Ladder App</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .sidebar { width: 300px; background: #2a2a2a; padding: 20px; border-radius: 8px; }
        .content { flex: 1; background: #2a2a2a; padding: 20px; border-radius: 8px; }
        select { width: 100%; height: 300px; background: #3a3a3a; color: #fff; border: 1px solid #4a4a4a; }
        label { display: block; margin: 10px 0 5px; color: #aaa; }
        button { width: 100%; padding: 10px; background: #4a90e2; border: none; color: white; cursor: pointer; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: center; border: 1px solid #4a4a4a; }
        th { background: #3a3a3a; }
        .latest-price { background: #4a4a4a; font-weight: bold; }
        .gap-bar-container { height: 5px; background: transparent; }
        .gap-bar { height: 100%; }
        .up { background: rgb(3, 218, 198); }  /* Teal */
        .down { background: rgb(207, 102, 121); }  /* Pinkish red */
        .cell-content { white-space: nowrap; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Index Snapshot</h2>
            {% for k, v in index_data.items() %}
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span>{{k}}</span> <span>{{v}}</span>
            </div>
            {% endfor %}
            
            <form method="post">
                <label>Select Stocks (Hold Ctrl)</label>
                <select name="symbols" multiple size="15">
                    {% for sym in display_symbols %}
                        <option value="{{ sym }}" {% if sym in selected_symbols %}selected{% endif %}>{{ sym }}</option>
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
                <h2>Price Ladder Analysis (Best Stock: {best_stock} - Avg % Move: {historical_avg_abs_move[best_stock]:.2f}%)</h2>
                <div style="font-size: 0.9em; color: #888;">
                    <span style="color: rgb(3, 218, 198);">●</span> Gap Up Intensity &nbsp;
                    <span style="color: rgb(207, 102, 121);">●</span> Gap Down Intensity
                </div>
            </div>
            
            {% if ladder_data %}
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Max Gap Up</th>
                        <th>Max Gap Down</th>
                        <th>% Ladder</th>  <!-- New column for % ladder -->
                        {% for i in range(-10, 0) %}<th>P{{i}}</th>{% endfor %}
                        <th>LTP</th>
                        {% for i in range(1, 11) %}<th>P+{{i}}</th>{% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in ladder_data %}
                    <tr {% if row.symbol == best_stock %}style="border: 2px solid #4a90e2;"{% endif %}>
                        <td style="font-weight: bold; color: #4a90e2;">{{ row.symbol }}</td>
                        <td style="color: rgb(3, 218, 198); font-weight: bold;">{{ row.max_gap_up }}%</td>
                        <td style="color: rgb(207, 102, 121); font-weight: bold;">{{ row.max_gap_down }}%</td>
                        <td>
                            <div style="display: flex; flex-direction: column; gap: 2px;">
                                {% for j in range(row.ladder_levels|length) %}
                                    <span style="font-size: 0.8em;">{{ row.ladder_levels[j] }}%: {{ row.price_ladder[j] }}</span>
                                {% endfor %}
                            </div>
                        </td>
                        
                        {# Lower Prices #}
                        {% for i in range(0, 10) %}
                            <td style="background: rgba(207, 102, 121, {{ (row.gaps[i] | abs / row.row_max_gap * 0.5) if row.row_max_gap else 0 }});">
                                <div class="cell-content">{{ row.lower_prices[i] if row.lower_prices[i] else '-' }}</div>
                                {% if row.gaps[i] != 0 %}
                                    <div class="gap-bar-container">
                                        <div class="gap-bar down" style="width: {{ (row.gaps[i] | abs / row.row_max_gap * 100) if row.row_max_gap else 0 }}%;"></div>
                                    </div>
                                {% endif %}
                            </td>
                        {% endfor %}
                        
                        <td class="latest-price">{{ row.base_price }}</td>
                        
                        {# Higher Prices #}
                        {% for i in range(0, 10) %}
                            <td style="background: rgba(3, 218, 198, {{ (row.gaps[i+10] | abs / row.row_max_gap * 0.5) if row.row_max_gap else 0 }});">
                                <div class="cell-content">{{ row.higher_prices[i] if row.higher_prices[i] else '-' }}</div>
                                {% if row.gaps[i+10] != 0 %}
                                    <div class="gap-bar-container">
                                        <div class="gap-bar up" style="width: {{ (row.gaps[i+10] | abs / row.row_max_gap * 100) if row.row_max_gap else 0 }}%;"></div>
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
                    <p>Select stocks and click "Analyze Ladder" to generate the heatmap.</p>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)