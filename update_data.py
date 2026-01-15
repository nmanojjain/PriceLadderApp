import pandas as pd
import yfinance as yf
import datetime
import os

CSV_FILE = "master_nse_data.csv"

def get_ticker_map():
    return {
        'NIFTY': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'SENSEX': '^BSESN',
        'CNXFINANCE': 'NIFTY_FIN_SERVICE.NS', # Trying this, if fails might need adjustment
        'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
        'M&M': 'M&M.NS',
        'M&MFIN': 'M&MFIN.NS',
    }

def update_data():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    # Read the last date from the CSV
    try:
        # Read only the first column to find the last date efficiently? 
        # Actually reading the whole file is safer to get columns too.
        df_existing = pd.read_csv(CSV_FILE, low_memory=False)
        last_date_str = df_existing['Date'].iloc[-1]
        last_date = datetime.datetime.strptime(last_date_str, "%d-%m-%Y").date()
        print(f"Last date in CSV: {last_date}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    start_date = last_date + datetime.timedelta(days=1)
    end_date = datetime.date.today()

    if start_date > end_date:
        print("Data is already up to date.")
        return

    print(f"Fetching data from {start_date} to {end_date}...")

    columns = df_existing.columns.tolist()
    # Remove 'Date' from columns to iterate over symbols
    symbols = [col for col in columns if col != 'Date']
    
    ticker_map = get_ticker_map()
    
    new_data = pd.DataFrame()
    new_data['Date'] = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
    # Note: yfinance might return data for non-business days or miss some holidays, 
    # better to fetch and then merge.

    # Let's fetch data for all symbols
    # yfinance can download multiple tickers at once
    
    yf_tickers = []
    for sym in symbols:
        if sym in ticker_map:
            yf_tickers.append(ticker_map[sym])
        else:
            yf_tickers.append(f"{sym}.NS")
            
    # Download in batches to avoid URL too long errors if many symbols
    # But yfinance handles it well usually. Let's try downloading all.
    
    print(f"Downloading data for {len(yf_tickers)} symbols...")
    
    try:
        # Using threads=True for faster download
        data = yf.download(yf_tickers, start=start_date, end=end_date + datetime.timedelta(days=1), group_by='ticker', threads=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    if data.empty:
        print("No new data received.")
        return

    # Process data
    # The data dataframe has MultiIndex columns (Ticker, Price Type)
    
    final_rows = []
    
    # Get all unique dates from the downloaded data
    # data.index is DatetimeIndex
    downloaded_dates = data.index.normalize().unique().sort_values()
    
    for date in downloaded_dates:
        row = {'Date': date.strftime("%d-%m-%Y")}
        for sym in symbols:
            yf_sym = ticker_map.get(sym, f"{sym}.NS")
            try:
                # Access data for the specific ticker and date
                # Handle cases where data might be missing for a specific symbol
                if len(yf_tickers) > 1:
                    val = data[yf_sym]['Close'].loc[date]
                else:
                    # If only one ticker was downloaded, the structure is different (no top level ticker column)
                    # But here we have many tickers.
                    val = data['Close'].loc[date] 
                
                if pd.notna(val):
                    row[sym] = round(val, 2)
                else:
                    row[sym] = "" # Or keep empty/NaN
            except KeyError:
                # Ticker might not be in the downloaded data or column name mismatch
                row[sym] = ""
            except Exception as e:
                # print(f"Error extracting data for {sym} on {date}: {e}")
                row[sym] = ""
        
        final_rows.append(row)

    if not final_rows:
        print("No rows processed.")
        return

    df_new = pd.DataFrame(final_rows)
    
    # Ensure columns are in the same order
    df_new = df_new[columns]
    
    # Append to CSV
    # We open in append mode and write without header
    try:
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print(f"Successfully appended {len(df_new)} rows to {CSV_FILE}")
    except Exception as e:
        print(f"Error appending to CSV: {e}")

if __name__ == "__main__":
    update_data()
