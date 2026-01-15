import pandas as pd
import yfinance as yf
import os

CSV_FILE = 'master_nse_data.csv'
TO_REMOVE = ['CYIENT', 'HFCL', 'NCC', 'TITAGARH']
TO_ADD = {
    'SWIGGY': 'SWIGGY.NS',
    'BAJAJHLDNG': 'BAJAJHLDNG.NS',
    'WAAREEENER': 'WAAREEENER.NS',
    'PREMIERENE': 'PREMIERENE.NS'
}

def update_data():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Loading existing data...")
    df = pd.read_csv(CSV_FILE)
    df['Date_parsed'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # 1. Remove old stocks
    cols_to_drop = [c for c in TO_REMOVE if c in df.columns]
    if cols_to_drop:
        print(f"Removing stocks: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # 2. Fetch and add new stocks
    start_date = df['Date_parsed'].min().strftime('%Y-%m-%d')
    end_date = df['Date_parsed'].max().strftime('%Y-%m-%d')
    output_end_date = (df['Date_parsed'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d') # yfinance end is exclusive

    for symbol, ticker in TO_ADD.items():
        print(f"Fetching data for {symbol} ({ticker})...")
        try:
            data = yf.download(ticker, start=start_date, end=output_end_date)
            if not data.empty:
                # yfinance returns MultiIndex if only one ticker is requested? Usually no.
                # But let's handle both cases.
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'][ticker]
                else:
                    prices = data['Close']
                
                # Prices is a Series with DatetimeIndex
                # We need to map it back to df['Date_parsed']
                df[symbol] = df['Date_parsed'].map(prices.round(2))
                print(f"Successfully added {symbol}")
            else:
                print(f"Warning: No data found for {ticker}")
                df[symbol] = None
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            df[symbol] = None

    # 3. Cleanup and Save
    df.drop(columns=['Date_parsed'], inplace=True)
    df.to_csv(CSV_FILE, index=False)
    print("Update complete.")

if __name__ == '__main__':
    update_data()
