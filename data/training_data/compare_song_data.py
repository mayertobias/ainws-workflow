import pandas as pd
import json
import numpy as np # For handling potential NaN comparisons

def compare_csv_files(external_csv_path, local_csv_path, keys, output_json_path):
    """
    Compares two CSV files based on a set of keys and generates a JSON report.

    Args:
        external_csv_path (str): Path to the first CSV file (e.g., externally enhanced).
        local_csv_path (str): Path to the second CSV file (e.g., local logic).
        keys (list): List of column names to use as keys for matching records.
        output_json_path (str): Path to save the JSON comparison report.
    """
    report = {
        "files_compared": {
            "external_file": external_csv_path,
            "local_file": local_csv_path
        },
        "merge_keys": keys,
        "summary": {},
        "feature_comparison": {},
        "unmatched_records": {}
    }

    try:
        df_external = pd.read_csv(external_csv_path)
        df_local = pd.read_csv(local_csv_path)

        report["summary"]["external_file_original_rows"] = len(df_external)
        report["summary"]["local_file_original_rows"] = len(df_local)

        # --- Standardization ---
        # Standardize duration column name in local df
        if 'duration_ms' in df_local.columns and 'song_duration_ms' not in df_local.columns:
            df_local = df_local.rename(columns={'duration_ms': 'song_duration_ms'})
            print("Renamed 'duration_ms' to 'song_duration_ms' in local_df")
        
        # Standardize valence column name in local df
        if 'valence' in df_local.columns and 'audio_valence' not in df_local.columns:
            df_local = df_local.rename(columns={'valence': 'audio_valence'})
            print("Renamed 'valence' to 'audio_valence' in local_df")

        # Standardize mode column name in local df
        if 'mode' in df_local.columns and 'audio_mode' not in df_local.columns:
            df_local = df_local.rename(columns={'mode': 'audio_mode'})
            print("Renamed 'mode' to 'audio_mode' in local_df")
            
        # Ensure all key columns exist
        for key_col in keys:
            if key_col not in df_external.columns:
                raise ValueError(f"Key column '{key_col}' not found in external CSV: {external_csv_path}")
            if key_col not in df_local.columns:
                raise ValueError(f"Key column '{key_col}' not found in local CSV: {local_csv_path}")

        # Convert key columns to string to ensure consistent merging
        for df in [df_external, df_local]:
            for key_col in keys:
                df[key_col] = df[key_col].astype(str)
        
        print(f"External CSV columns after standardization: {df_external.columns.tolist()}")
        print(f"Local CSV columns after standardization: {df_local.columns.tolist()}")

        # --- Identify Common Feature Columns (excluding keys) ---
        external_cols = set(df_external.columns)
        local_cols = set(df_local.columns)
        common_cols = list(external_cols.intersection(local_cols))
        feature_cols_to_compare = [col for col in common_cols if col not in keys]
        report["summary"]["common_features_compared"] = feature_cols_to_compare
        
        # --- Merging ---
        # Inner merge to find records present in both based on keys
        merged_df = pd.merge(df_external, df_local, on=keys, how='inner', suffixes=['_ext', '_loc'])
        report["summary"]["matched_records_on_keys"] = len(merged_df)

        if report["summary"]["external_file_original_rows"] > 0:
            report["summary"]["percentage_external_matched"] = \
                (len(merged_df) / report["summary"]["external_file_original_rows"]) * 100
        else:
            report["summary"]["percentage_external_matched"] = 0

        if report["summary"]["local_file_original_rows"] > 0:
            report["summary"]["percentage_local_matched"] = \
                (len(merged_df) / report["summary"]["local_file_original_rows"]) * 100
        else:
            report["summary"]["percentage_local_matched"] = 0
            
        # --- Feature Comparison (for matched records) ---
        for feature in feature_cols_to_compare:
            col_ext = feature + '_ext'
            col_loc = feature + '_loc'
            
            # Ensure columns exist in merged_df (they should due to common_cols logic)
            if col_ext not in merged_df.columns or col_loc not in merged_df.columns:
                print(f"Skipping feature {feature} as its suffixed versions are not in merged_df.")
                continue

            comparison_details = {}
            # Handle potential NaN values by treating them as a distinct category for comparison
            # or by converting to a consistent string representation if necessary.
            # For direct comparison, ensure types are compatible or convert.
            
            # Attempt to convert to numeric if possible, otherwise compare as objects/strings
            try:
                merged_df[col_ext] = pd.to_numeric(merged_df[col_ext])
                merged_df[col_loc] = pd.to_numeric(merged_df[col_loc])
                is_numeric = True
            except ValueError:
                is_numeric = False
                # Fill NaN with a placeholder string for comparison if not numeric, to make np.isclose work or for direct string comparison
                merged_df[col_ext] = merged_df[col_ext].fillna("MISSING_VALUE_EXT")
                merged_df[col_loc] = merged_df[col_loc].fillna("MISSING_VALUE_LOC")


            if is_numeric:
                 # For numeric, use np.isclose for float comparisons, direct for int
                if pd.api.types.is_float_dtype(merged_df[col_ext]) or pd.api.types.is_float_dtype(merged_df[col_loc]):
                    # Replace NaN with a value that won't match, or handle explicitly
                    # np.isclose handles NaNs by returning False, which is desired (NaN != NaN for matching)
                    are_equal = np.isclose(merged_df[col_ext].fillna(np.nan), merged_df[col_loc].fillna(np.nan), equal_nan=False)
                else: # Integer types
                    are_equal = (merged_df[col_ext] == merged_df[col_loc])
            else: # String/Object types
                are_equal = (merged_df[col_ext].astype(str) == merged_df[col_loc].astype(str))

            comparison_details["identical_values_count"] = int(are_equal.sum())
            comparison_details["differing_values_count"] = int((~are_equal).sum())

            if is_numeric and comparison_details["differing_values_count"] > 0:
                differences = merged_df[col_ext][~are_equal] - merged_df[col_loc][~are_equal]
                comparison_details["difference_stats"] = {
                    "mean_absolute_difference": float(differences.abs().mean()),
                    "min_difference": float(differences.min()),
                    "max_difference": float(differences.max()),
                    "std_dev_difference": float(differences.std())
                }
                # Descriptive stats for each source for this feature on matched records
                comparison_details[f"{feature}_ext_stats"] = merged_df[col_ext].describe().to_dict()
                comparison_details[f"{feature}_loc_stats"] = merged_df[col_loc].describe().to_dict()


            elif not is_numeric and comparison_details["differing_values_count"] > 0:
                mismatches_df = merged_df[~are_equal][[feature + '_ext', feature + '_loc']]
                # Count occurrences of each specific mismatch pair
                mismatch_counts = mismatches_df.groupby([feature + '_ext', feature + '_loc']).size().reset_index(name='count')
                comparison_details["mismatch_examples_counts"] = mismatch_counts.to_dict(orient='records')
            
            report["feature_comparison"][feature] = comparison_details

        # --- Unmatched Records Analysis ---
        # Using indicator=True in an outer merge to find unique records
        outer_merged_df = pd.merge(df_external, df_local, on=keys, how='outer', suffixes=['_ext', '_loc'], indicator=True)
        
        report["unmatched_records"]["only_in_external_file_count"] = len(outer_merged_df[outer_merged_df['_merge'] == 'left_only'])
        report["unmatched_records"]["only_in_local_file_count"] = len(outer_merged_df[outer_merged_df['_merge'] == 'right_only'])

    except FileNotFoundError as e:
        report["error"] = f"File not found: {e}"
        print(report["error"])
    except ValueError as e:
        report["error"] = f"Value error: {e}"
        print(report["error"])
    except Exception as e:
        report["error"] = f"An unexpected error occurred: {e}"
        print(report["error"])

    with open(output_json_path, 'w') as f:
        json.dump(report, f, indent=4, default=lambda x: str(x) if pd.isna(x) else x) # Handle NaN for JSON
    print(f"Comparison report saved to {output_json_path}")
    return report

if __name__ == '__main__':
    # Define file paths and keys
    external_file = "/Users/manojveluchuri/saas/r1/simpleui/backend/data/training_data/r4a_trng_data_with_ext_f_for_cmp.csv"
    local_file = "/Users/manojveluchuri/saas/r1/simpleui/backend/data/song_features_with_popularity.csv"
    # Keys as confirmed by user
    comparison_keys = ["song_name", "song_duration_ms", "song_popularity"] 
    output_file = "/Users/manojveluchuri/saas/r1/simpleui/backend/data/training_data/song_comparison_report.json"

    print(f"Starting comparison between {external_file} and {local_file}")
    print(f"Using keys: {comparison_keys}")
    
    # Create dummy files for testing if they don't exist (REMOVE IN PRODUCTION)
    # import os
    # if not os.path.exists(external_file):
    #     print(f"Creating dummy external file: {external_file}")
    #     pd.DataFrame({'song_name':['A','B','C'], 'song_duration_ms':['100','200','300'], 'song_popularity':['10','20','30'], 'feature1':[1,2,3]}).to_csv(external_file, index=False)
    # if not os.path.exists(local_file):
    #     print(f"Creating dummy local file: {local_file}")
    #     pd.DataFrame({'song_name':['A','B','D'], 'duration_ms':['100','200','400'], 'song_popularity':['10','20','40'], 'feature1':[1,2.1,4], 'valence':[0.5,0.6,0.7]}).to_csv(local_file, index=False)


    compare_csv_files(external_file, local_file, comparison_keys, output_file)
    print("Comparison script finished.") 