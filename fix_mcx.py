import pandas as pd
import os

CSV_FILE = "master_nse_data.csv"

def fix_mcx_split():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Reading CSV...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    
    # Identify the split point (Ex-date was 03-12-2025 in the data)
    # We'll adjust all rows BEFORE index 414
    
    print("Adjusting MCX prices for 5:1 split...")
    
    # Check if 'MCX' column exists
    if 'MCX' not in df.columns:
        print("Error: MCX column not found.")
        return

    # Convert to numeric just in case
    df['MCX'] = pd.to_numeric(df['MCX'], errors='coerce')
    
    # Split point index is 414 based on our check.
    # However, to be extra safe, we can check the date or the price magnitude.
    # Let's use the index we confirmed.
    
    # Divide first 414 rows by 5
    df.loc[0:413, 'MCX'] = df.loc[0:413, 'MCX'] / 5
    
    print(f"MCX at index 413 (old): {df.loc[413, 'MCX']}")
    print(f"MCX at index 414 (new): {df.loc[414, 'MCX']}")
    
    # Save back to CSV
    print("Saving corrected data...")
    df.to_csv(CSV_FILE, index=False)
    print("Successfully adjusted MCX for split.")

if __name__ == "__main__":
    fix_mcx_split()
