import pandas as pd

# Update this if your Model 3 CSV has a different name
filename = 'Effective_Stress2.csv' 

print(f"--- CHECKING COLUMNS FOR: {filename} ---")

try:
    # Reading it exactly how your training script reads it
    df = pd.read_csv(filename, encoding='ISO-8859-1')
    
    print("\nHere are the EXACT column names Python sees:")
    print("--------------------------------------------------")
    
    for col in df.columns:
        # Printing with brackets so you can see if there are trailing spaces!
        print(f"[{col}]")
        
    print("--------------------------------------------------")
    
except FileNotFoundError:
    print(f"❌ Error: Could not find '{filename}' in this folder.")