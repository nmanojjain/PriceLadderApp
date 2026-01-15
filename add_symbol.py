"""
Script to add a new symbol to the Supabase database.
Fetches historical closing prices from yfinance and uploads to cloud.
"""
import os
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def add_symbol(symbol, days=2000):
    """Fetch historical data for a symbol and upload to Supabase."""
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Supabase credentials not found in .env")
        return False
    
    print(f"Initializing Supabase client...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Fetch from yfinance
    ticker = f"{symbol}.NS"
    print(f"Fetching {days} days of data for {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d", auto_adjust=True)
        
        if hist.empty:
            print(f"No data found for {ticker}. Trying BSE...")
            ticker = f"{symbol}.BO"
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days}d")
        
        if hist.empty:
            print(f"Error: No data found for {symbol} on NSE or BSE")
            return False
        
        print(f"Retrieved {len(hist)} days of data.")
        
        # Prepare records
        records = []
        for date, row in hist.iterrows():
            close_price = round(float(row['Close']), 2)
            date_str = date.strftime('%Y-%m-%d')
            records.append({
                "symbol": symbol,
                "date": date_str,
                "price": close_price
            })
        
        print(f"Uploading {len(records)} records to Supabase...")
        
        # CLEAR EXISTING DATA FIRST to avoid conflict errors
        print(f"  Clearing existing data for {symbol}...")
        supabase.table("market_data").delete().eq("symbol", symbol).execute()

        # Upload in batches of 500
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            supabase.table("market_data").upsert(batch).execute()
            print(f"  Uploaded batch {i//batch_size + 1}")
        
        print(f"✓ Successfully added {symbol} to database with {len(records)} historical prices!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        add_symbol(symbol)
    else:
        print("Usage: python add_symbol.py SYMBOL")
        print("Example: python add_symbol.py TMPV")
