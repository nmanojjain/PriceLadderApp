import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

CSV_FILE = 'master_nse_data.csv'

def expand_historical_data():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Loading symbol list...")
    df_existing = pd.read_csv(CSV_FILE, nrows=0)
    symbols = [col for col in df_existing.columns if col != 'Date']
    
    ticker_map = {
        'NIFTY': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'SENSEX': '^BSESN',
        'CNXFINANCE': 'NIFTY_FIN_SERVICE.NS'
    }

    yf_tickers = []
    for sym in symbols:
        if sym in ticker_map:
            yf_tickers.append(ticker_map[sym])
        else:
            yf_tickers.append(f"{sym}.NS")

    print(f"Fetching 500+ trading days for {len(yf_tickers)} symbols...")
    
    # Calculate start date (approx 750 days back to be safe)
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=800) 
    
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        # Download in chunks to avoid URL length issues or timeouts
        chunk_size = 50
        all_data = []
        
        for i in range(0, len(yf_tickers), chunk_size):
            chunk = yf_tickers[i:i+chunk_size]
            print(f"Downloading chunk {i//chunk_size + 1}...")
            data = yf.download(chunk, start=start_date.strftime('%Y-%m-%d'), 
                               end=end_date.strftime('%Y-%m-%d'), group_by='ticker')
            all_data.append(data)

        # Merge all chunks
        final_df_list = []
        for ticker, sym in zip(yf_tickers, symbols):
            # Extract 'Close' for each ticker
            # yf.download(group_by='ticker') returns columns as (Ticker, Attribute)
            
            # Find which chunk this ticker belongs to
            chunk_data = None
            for d in all_data:
                if isinstance(d.columns, pd.MultiIndex):
                    if ticker in d.columns.levels[0]:
                        chunk_data = d[ticker]['Close']
                        break
                else:
                    # Single ticker case
                    if ticker == d.name if hasattr(d, 'name') else None:
                         chunk_data = d
                         break
            
            if chunk_data is not None:
                s = chunk_data.rename(sym)
                final_df_list.append(s)
            else:
                print(f"Warning: Could not find data for {ticker}")

        # Combine all series into one DataFrame
        master_df = pd.concat(final_df_list, axis=1)
        master_df.index = master_df.index.strftime('%d-%m-%Y')
        master_df.index.name = 'Date'
        master_df.reset_index(inplace=True)

        # Back up existing file
        os.rename(CSV_FILE, CSV_FILE + '.bak')
        
        # Save new file
        master_df.to_csv(CSV_FILE, index=False)
        print(f"Successfully updated {CSV_FILE} with {len(master_df)} trading days.")

    except Exception as e:
        print(f"Error during expansion: {e}")
        if os.path.exists(CSV_FILE + '.bak') and not os.path.exists(CSV_FILE):
            os.rename(CSV_FILE + '.bak', CSV_FILE)

if __name__ == '__main__':
    expand_historical_data()
