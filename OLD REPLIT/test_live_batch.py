import yfinance as yf
import pandas as pd

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
            
        print(f"Fetching: {yf_tickers}")
        # Download batch data
        data = yf.download(yf_tickers, period="1d", interval="1m", group_by='ticker', threads=True)
        
        print("Data columns:", data.columns)
        
        if not data.empty:
            if len(yf_tickers) > 1:
                for sym, yf_sym in zip(symbols, yf_tickers):
                    try:
                        # Check if the ticker is in the top level columns
                        if yf_sym in data.columns.levels[0]:
                            closes = data[yf_sym]['Close'].dropna()
                            if not closes.empty:
                                live_prices[sym] = closes.iloc[-1]
                                print(f"Got price for {sym}: {closes.iloc[-1]}")
                            else:
                                print(f"No close data for {sym}")
                        else:
                            print(f"{yf_sym} not in columns")
                    except Exception as e:
                        print(f"Error extracting data for {sym}: {e}")
            else:
                sym = symbols[0]
                closes = data['Close'].dropna()
                if not closes.empty:
                    live_prices[sym] = closes.iloc[-1]
                    print(f"Got price for {sym}: {closes.iloc[-1]}")
                    
    except Exception as e:
        print(f"Error fetching batch live prices: {e}")
        
    return live_prices

symbols = ['RELIANCE', 'TCS', 'INFY', 'NIFTY']
prices = get_live_prices_batch(symbols)
print("Final Prices:", prices)
