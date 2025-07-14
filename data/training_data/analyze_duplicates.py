import pandas as pd
import os

# Input CSV files
ENRICHED_CSV_FILE = "r4a_song_data_enriched_from_db.csv"
ORIGINAL_CSV_FILE = "song_data.csv" # The original source file

def analyze_dataframe_duplicates(df, df_name, key_columns_to_check):
    """
    Performs duplicate analysis on the provided DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        df_name (str): A descriptive name for the DataFrame (for printing).
        key_columns_to_check (dict): Dictionary where keys are analysis names 
                                     and values are lists of column names to check for duplicates.
    """
    print(f"\n\n--- DataFrame Info for {df_name} ---")
    df.info()
    print(f"\n--- {df_name}: DataFrame Head ---")
    print(df.head())
    print(f"\n{df_name} shape: {df.shape}")

    print(f"\n--- {df_name}: Duplicate Analysis ---")
    for analysis_name, cols in key_columns_to_check.items():
        print(f"\n>>> {df_name}: Analyzing duplicates based on: {cols}")
        
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            print(f"  Skipping analysis '{analysis_name}': Missing columns {missing_cols} in {df_name}")
            continue
        
        # Handle potential float precision issues if duration is float, but it should be int/object from CSV
        # For duplicate checking, it's often best to convert key columns to string if types are mixed or to handle NaNs
        # df_subset = df[cols].astype(str) # Example: convert all key cols to string for robust duplication check

        duplicate_mask = df.duplicated(subset=cols, keep=False)
        duplicates_df = df[duplicate_mask]

        if duplicates_df.empty:
            print(f"  No duplicates found in {df_name} based on {cols}.")
        else:
            num_duplicate_rows = duplicates_df.shape[0]
            num_duplicate_groups = df.duplicated(subset=cols, keep='first').sum()
            print(f"  Found {num_duplicate_rows} rows in {df_name} that are part of duplicate sets (across {num_duplicate_groups} unique duplicate groups).")
            print(f"  Showing up to 3 groups of these duplicates from {df_name} based on {cols}:")
            
            sorted_duplicates = duplicates_df.sort_values(by=cols)
            grouped = sorted_duplicates.groupby(cols)
            
            for i, (name, group) in enumerate(grouped):
                if i < 3: # Show first 3 groups
                    print(f"\n  {df_name} - Duplicate Group {i+1} (Key: {name}):")
                    print(group)
                else:
                    break
            if grouped.ngroups > 3:
                print(f"  ... and {grouped.ngroups - 3} more duplicate groups in {df_name}.")

    print(f"\n--- {df_name}: End of Duplicate Analysis ---")
    print(f"Review the output for {df_name} to understand its duplicate nature.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Analyze ORIGINAL song_data.csv ---
    original_csv_full_path = os.path.join(base_dir, ORIGINAL_CSV_FILE)
    print(f"\n======================================================================")
    print(f"Starting duplicate analysis for ORIGINAL file: {original_csv_full_path}")
    print(f"======================================================================")
    if not os.path.exists(original_csv_full_path):
        print(f"ERROR: ORIGINAL Input file not found at {original_csv_full_path}")
    else:
        try:
            df_original = pd.read_csv(original_csv_full_path)
            # Define key columns relevant for the original CSV
            original_key_cols = {
                "song_name_only": ["song_name"],
                "song_name_and_duration": ["song_name", "song_duration_ms"],
                "song_name_duration_and_key": ["song_name", "song_duration_ms", "key"] # Assuming 'key' column exists
            }
            analyze_dataframe_duplicates(df_original, "Original song_data.csv", original_key_cols)
        except Exception as e:
            print(f"Error processing original CSV file {original_csv_full_path}: {e}")

    # --- Analyze ENRICHED r4a_song_data_enriched_from_db.csv ---
    enriched_csv_full_path = os.path.join(base_dir, ENRICHED_CSV_FILE)
    print(f"\n\n======================================================================")
    print(f"Starting duplicate analysis for ENRICHED file: {enriched_csv_full_path}")
    print(f"======================================================================")
    if not os.path.exists(enriched_csv_full_path):
        print(f"ERROR: ENRICHED Input file not found at {enriched_csv_full_path}")
    else:
        try:
            df_enriched = pd.read_csv(enriched_csv_full_path)
            # Define key columns relevant for the enriched CSV
            enriched_key_cols = {
                "song_name_only": ["song_name"],
                "artist_mbid_only": ["artist_mbid"],
                "song_and_artist_name": ["song_name", "mb_artist_name"],
                "song_and_artist_mbid": ["song_name", "artist_mbid"],
            }
            analyze_dataframe_duplicates(df_enriched, "Enriched r4a_song_data_enriched_from_db.csv", enriched_key_cols)
        except Exception as e:
            print(f"Error processing enriched CSV file {enriched_csv_full_path}: {e}")
    
    print("\nAnalysis script finished.")

if __name__ == "__main__":
    main() 