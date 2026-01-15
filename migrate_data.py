import pandas as pd
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUBAPASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CSV_FILE = "master_nse_data.csv"

def migrate():
    if not SUBAPASE_URL or not SUPABASE_KEY:
        print("Error: Supabase credentials not found in .env")
        return

    print("Initializing Supabase client...")
    supabase: Client = create_client(SUBAPASE_URL, SUPABASE_KEY)

    print(f"Reading {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip() for c in df.columns]

    # Handle Date Formats explicitly for PostgreSQL
    # If date is DD-MM-YYYY, we need to convert to YYYY-MM-DD
    print("Standardizing date formats for PostgreSQL...")
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Date conversion error: {e}")
        return

    # Convert to Long Format
    print("Converting data to long format and uploading...")
    
    symbol_cols = [c for c in df.columns if c != 'Date']
    total_symbols = len(symbol_cols)
    
    for i, symbol in enumerate(symbol_cols):
        # Filter non-null data for this symbol
        stock_df = df[['Date', symbol]].dropna()
        if stock_df.empty:
            continue
            
        records = []
        for index, row in stock_df.iterrows():
            records.append({
                "symbol": symbol,
                "date": row['Date'],
                "price": round(float(row[symbol]), 2)
            })
            
        # Bulk Insert in manageable batches of 1000 records
        if records:
            print(f"[{i+1}/{total_symbols}] Uploading {len(records)} levels for {symbol}...")
            try:
                # Batch upload (Supabase/Postgrest handles large arrays well)
                supabase.table("market_data").upsert(records).execute()
            except Exception as e:
                print(f"Error uploading {symbol}: {e}")

    print("Migration Complete!")

if __name__ == "__main__":
    migrate()
