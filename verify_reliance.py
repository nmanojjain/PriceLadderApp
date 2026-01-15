import yfinance as yf
import pandas as pd

def check_reliance():
    symbol = "RELIANCE.NS"
    print(f"Fetching data for {symbol}...")
    
    # Force auto_adjust=True to ensure we get split/bonus adjusted prices
    stock = yf.Ticker(symbol)
    # Fetch 5 years
    hist = stock.history(period="5y", auto_adjust=True)
    
    if hist.empty:
        print("No data found.")
        return

    latest = round(hist['Close'].iloc[-1], 2)
    print(f"Latest Price: {latest}")
    
    # Find closest raw levels
    prices = sorted(hist['Close'].round(2).unique())
    
    above = sorted([p for p in prices if p > latest])[:5]
    below = sorted([p for p in prices if p < latest], reverse=True)[:5]
    
    print("\n--- NEAREST RAW LEVELS (Calculated from YFinance) ---")
    print("Resistance (Above):", above)
    print("Support (Below):   ", below)
    
    print("\n--- SAMPLE HISTORY (Last 5 days) ---")
    print(hist.tail()[['Close']])

if __name__ == "__main__":
    check_reliance()
