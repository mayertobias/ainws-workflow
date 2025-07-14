import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

def generate_comparison_plots(report_data, output_directory, plot_suffix=''):
    """
    Generates and saves bar plots for MAD and RMSE from the comparison report data.
    Handles both overall and binned statistics.
    """
    # Overall plots
    overall_summary_stats = report_data.get('overall_summary_stats', {})
    if overall_summary_stats:
        features = list(overall_summary_stats.keys())
        mad_values = [overall_summary_stats[f].get('mean_absolute_difference') for f in features]
        rmse_values = [overall_summary_stats[f].get('rmse') for f in features]

        plot_data_overall = []
        for i, feature_name in enumerate(features):
            if mad_values[i] is not None and rmse_values[i] is not None:
                plot_data_overall.append({'Feature': feature_name, 'MAD': mad_values[i], 'RMSE': rmse_values[i]})
        
        if plot_data_overall:
            df_plot_overall = pd.DataFrame(plot_data_overall)
            # Plot Overall MAD
            plt.figure(figsize=(12, 7))
            plt.bar(df_plot_overall['Feature'], df_plot_overall['MAD'], color='skyblue')
            plt.xlabel('Audio Feature')
            plt.ylabel('Mean Absolute Difference (MAD)')
            plt.title(f'Overall MAD (Original vs Calculated){plot_suffix}')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            mad_plot_path = os.path.join(output_directory, f"internal_mad_overall_plot{plot_suffix}.png")
            try: plt.savefig(mad_plot_path); print(f"Overall MAD plot saved to {mad_plot_path}")
            except Exception as e: print(f"Error saving overall MAD plot: {e}")
            plt.close()

            # Plot Overall RMSE
            plt.figure(figsize=(12, 7))
            plt.bar(df_plot_overall['Feature'], df_plot_overall['RMSE'], color='lightcoral')
            plt.xlabel('Audio Feature')
            plt.ylabel('Root Mean Squared Error (RMSE)')
            plt.title(f'Overall RMSE (Original vs Calculated){plot_suffix}')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            rmse_plot_path = os.path.join(output_directory, f"internal_rmse_overall_plot{plot_suffix}.png")
            try: plt.savefig(rmse_plot_path); print(f"Overall RMSE plot saved to {rmse_plot_path}")
            except Exception as e: print(f"Error saving overall RMSE plot: {e}")
            plt.close()
        else:
            print("No valid overall MAD/RMSE data to plot.")
    else:
        print("No overall_summary_stats found in report for plotting.")

    # Binned plots
    binned_summary_stats = report_data.get('binned_summary_stats', {})
    if not binned_summary_stats:
        print("No binned_summary_stats found in report for plotting.")
        return

    features_analyzed = report_data.get("features_analyzed", [])
    pop_bin_labels = report_data.get("popularity_bin_definition", {}).get("labels", [])
    if not features_analyzed or not pop_bin_labels:
        print("Feature list or bin labels missing for binned plots.")
        return

    metrics_to_plot = [
        ('mean_absolute_difference', 'MAD', 'skyblue'),
        ('rmse', 'RMSE', 'lightcoral'),
        ('pearson_correlation', 'Pearson Correlation', 'lightgreen')
    ]

    for metric_key, metric_name, color in metrics_to_plot:
        plot_data_binned = []
        for feature in features_analyzed:
            for bin_label in pop_bin_labels:
                value = binned_summary_stats.get(feature, {}).get(bin_label, {}).get(metric_key)
                if value is not None: 
                    plot_data_binned.append({'Feature': feature, 'Bin': bin_label, metric_name: value})
        
        if not plot_data_binned:
            print(f"No data for binned {metric_name} plot.")
            continue
            
        df_plot_binned = pd.DataFrame(plot_data_binned)
        
        if df_plot_binned.empty:
            print(f"DataFrame for binned {metric_name} plot is empty.")
            continue

        plt.figure(figsize=(15, 8))
        try:
            pivot_df = df_plot_binned.pivot(index='Feature', columns='Bin', values=metric_name)
            if not pivot_df.empty:
                pivot_df = pivot_df.reindex(columns=pop_bin_labels)
                pivot_df.plot(kind='bar', ax=plt.gca(), colormap='viridis') 
                plt.title(f'{metric_name} by Feature and Popularity Bin{plot_suffix}')
                plt.xlabel('Audio Feature')
                plt.ylabel(metric_name)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Popularity Bin')
                plt.grid(axis='y', linestyle='--', alpha=0.7) 
                plt.tight_layout()
                plot_path = os.path.join(output_directory, f"internal_{metric_key}_binned_plot{plot_suffix}.png")
                try: plt.savefig(plot_path); print(f"Binned {metric_name} plot saved to {plot_path}")
                except Exception as e: print(f"Error saving binned {metric_name} plot: {e}")
            else:
                print(f"Pivot table for binned {metric_name} is empty.")
        except KeyError as e:
            print(f"KeyError during pivot for {metric_name}: {e}. Check if all bins have data for all features.")
        except Exception as e:
            print(f"General error during plotting binned {metric_name}: {e}")
        finally:
            plt.close()


def get_popularity_bin(popularity, bins, labels):
    if popularity is None:
        return "Unknown"
    try:
        pop_val = float(popularity) # Ensure popularity is numeric
        return pd.cut([pop_val], bins=bins, labels=labels, right=False, include_lowest=True)[0]
    except (ValueError, TypeError):
        return "Invalid_Popularity_Value"


def compare_features_in_json(json_path, features_to_compare_map, output_report_path, plot_suffix=''):
    """
    Compares specified original and calculated features within a single JSON file.
    Segments comparison by popularity bins.
    """
    # Popularity bin definition
    pop_bins = [-np.inf, 20, 40, 60, 80, np.inf] # Using -np.inf and np.inf for inclusive ranges
    pop_labels = ['0-20', '21-40', '41-60', '61-80', '81+']


    report = {
        "source_file": json_path,
        "features_analyzed": list(features_to_compare_map.keys()),
        "popularity_bin_definition": {"bins": [str(b) for b in pop_bins], "labels": pop_labels}, # Store bins as strings for JSON
        "overall_summary_stats": {},
        "binned_summary_stats": {},
        # "detailed_comparison": {} # Kept commented out
    }

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON data is not a list of records.")

        report["total_records_processed"] = len(data)
        all_records_data = []

        for record_index, song_entry in enumerate(data):
            record_comparison_data = {'record_index': record_index, 'song_name': song_entry.get('song_name', 'N/A')}
            
            original_features_data = song_entry.get('original_data', {})
            calculated_features_data = song_entry.get('audio_features', {})

            if not original_features_data: # Calculated can be empty if we only have original
                # print(f"Skipping record {record_index} due to missing original_data.")
                continue 

            # Get popularity and assign bin
            popularity = original_features_data.get('training_song_popularity')
            record_comparison_data['popularity_bin'] = get_popularity_bin(popularity, pop_bins, pop_labels)
            record_comparison_data['popularity_original'] = popularity


            for generic_name, (original_key, calculated_key) in features_to_compare_map.items():
                original_value = original_features_data.get(original_key)
                # Calculated value might be missing if audio_features block is not there for a song
                calculated_value = calculated_features_data.get(calculated_key) if calculated_features_data else None
                
                record_comparison_data[f'{generic_name}_original'] = original_value
                record_comparison_data[f'{generic_name}_calculated'] = calculated_value

                if original_value is None or calculated_value is None:
                    record_comparison_data[f'{generic_name}_differs'] = 'missing_data'
                else:
                    try:
                        val_orig_num = pd.to_numeric(original_value)
                        val_calc_num = pd.to_numeric(calculated_value)
                        if np.isclose(val_orig_num, val_calc_num):
                            record_comparison_data[f'{generic_name}_differs'] = False
                        else:
                            record_comparison_data[f'{generic_name}_differs'] = True
                            record_comparison_data[f'{generic_name}_difference'] = val_calc_num - val_orig_num
                    except ValueError: 
                        if str(original_value) == str(calculated_value):
                            record_comparison_data[f'{generic_name}_differs'] = False
                        else:
                            record_comparison_data[f'{generic_name}_differs'] = True
                            record_comparison_data[f'{generic_name}_difference'] = 'non_numeric_diff'
            
            all_records_data.append(record_comparison_data)

        comparison_df = pd.DataFrame(all_records_data)
        if comparison_df.empty:
            report["error"] = "No valid records found or processed for comparison."
            # Save partial report and return
            with open(output_report_path, 'w', encoding='utf-8') as f_out:
                json.dump(report, f_out, indent=4, default=lambda x: str(x) if pd.isna(x) else x)
            return

        # --- Overall Aggregate Statistics ---
        for generic_name in features_to_compare_map.keys():
            stats = {}
            original_col = f'{generic_name}_original'
            calculated_col = f'{generic_name}_calculated'
            differs_col = f'{generic_name}_differs'
            difference_col = f'{generic_name}_difference'

            if differs_col not in comparison_df.columns:
                stats['error'] = f"Column {differs_col} not found for overall stats."
                report['overall_summary_stats'][generic_name] = stats
                continue
            
            stats['total_pairs_found'] = int(comparison_df[differs_col].notna().sum())
            stats['identical_values_count'] = int((comparison_df[differs_col] == False).sum())
            stats['differing_values_count'] = int((comparison_df[differs_col] == True).sum())
            stats['missing_data_count'] = int((comparison_df[differs_col] == 'missing_data').sum())

            numeric_diffs_overall = pd.to_numeric(comparison_df[difference_col], errors='coerce').dropna()
            valid_original_overall = pd.to_numeric(comparison_df[original_col], errors='coerce').dropna()
            valid_calculated_overall = pd.to_numeric(comparison_df[calculated_col], errors='coerce').dropna()
            common_idx_overall = valid_original_overall.index.intersection(valid_calculated_overall.index)
            aligned_original_overall = valid_original_overall.loc[common_idx_overall]
            aligned_calculated_overall = valid_calculated_overall.loc[common_idx_overall]

            if not numeric_diffs_overall.empty:
                stats['mean_absolute_difference'] = float(numeric_diffs_overall.abs().mean())
                # Ensure aligned series for RMSE are from the subset where numeric_diffs_overall is defined
                rmse_indices = numeric_diffs_overall.index.intersection(common_idx_overall)
                if not rmse_indices.empty:
                    stats['rmse'] = float(np.sqrt(mean_squared_error(aligned_original_overall.loc[rmse_indices], aligned_calculated_overall.loc[rmse_indices])))
                else:
                    stats['rmse'] = None

                stats['mean_difference'] = float(numeric_diffs_overall.mean())
                stats['std_dev_difference'] = float(numeric_diffs_overall.std())
                stats['min_difference'] = float(numeric_diffs_overall.min())
                stats['max_difference'] = float(numeric_diffs_overall.max())
            
            if len(aligned_original_overall) > 1 and len(aligned_calculated_overall) > 1:
                corr, _ = pearsonr(aligned_original_overall, aligned_calculated_overall)
                stats['pearson_correlation'] = float(corr)
            else:
                stats['pearson_correlation'] = None
            report['overall_summary_stats'][generic_name] = stats
            
        # --- Binned Aggregate Statistics ---
        report['binned_summary_stats'] = {}
        for generic_name in features_to_compare_map.keys():
            report['binned_summary_stats'][generic_name] = {}
            for bin_label in pop_labels + ["Unknown", "Invalid_Popularity_Value"]: # Include potential other bin values
                df_bin = comparison_df[comparison_df['popularity_bin'] == bin_label]
                if df_bin.empty:
                    # report['binned_summary_stats'][generic_name][bin_label] = {"message": "No data for this bin"}
                    continue # Skip if bin is empty

                stats_bin = {}
                original_col = f'{generic_name}_original'
                calculated_col = f'{generic_name}_calculated'
                differs_col = f'{generic_name}_differs'
                difference_col = f'{generic_name}_difference'

                if differs_col not in df_bin.columns: # Should exist if overall check passed
                    stats_bin['error'] = f"Column {differs_col} not found for bin {bin_label}."
                    report['binned_summary_stats'][generic_name][bin_label] = stats_bin
                    continue

                stats_bin['records_in_bin'] = len(df_bin)
                stats_bin['total_pairs_found'] = int(df_bin[differs_col].notna().sum())
                stats_bin['identical_values_count'] = int((df_bin[differs_col] == False).sum())
                stats_bin['differing_values_count'] = int((df_bin[differs_col] == True).sum())
                stats_bin['missing_data_count'] = int((df_bin[differs_col] == 'missing_data').sum())

                numeric_diffs_bin = pd.to_numeric(df_bin[difference_col], errors='coerce').dropna()
                valid_original_bin = pd.to_numeric(df_bin[original_col], errors='coerce').dropna()
                valid_calculated_bin = pd.to_numeric(df_bin[calculated_col], errors='coerce').dropna()
                common_idx_bin = valid_original_bin.index.intersection(valid_calculated_bin.index)
                aligned_original_bin = valid_original_bin.loc[common_idx_bin]
                aligned_calculated_bin = valid_calculated_bin.loc[common_idx_bin]

                if not numeric_diffs_bin.empty:
                    stats_bin['mean_absolute_difference'] = float(numeric_diffs_bin.abs().mean())
                    rmse_indices_bin = numeric_diffs_bin.index.intersection(common_idx_bin)
                    if not rmse_indices_bin.empty and len(aligned_original_bin.loc[rmse_indices_bin]) > 0 : # Check for empty result after loc
                         stats_bin['rmse'] = float(np.sqrt(mean_squared_error(aligned_original_bin.loc[rmse_indices_bin], aligned_calculated_bin.loc[rmse_indices_bin])))
                    else:
                        stats_bin['rmse'] = None
                    stats_bin['mean_difference'] = float(numeric_diffs_bin.mean())
                    stats_bin['std_dev_difference'] = float(numeric_diffs_bin.std())
                    # Min/Max might be misleading if only one diff, add check
                    if len(numeric_diffs_bin) > 0:
                        stats_bin['min_difference'] = float(numeric_diffs_bin.min())
                        stats_bin['max_difference'] = float(numeric_diffs_bin.max())
                    else:
                        stats_bin['min_difference'] = None
                        stats_bin['max_difference'] = None

                if len(aligned_original_bin) > 1 and len(aligned_calculated_bin) > 1: # Pearson needs at least 2 points
                    try:
                        corr_bin, _ = pearsonr(aligned_original_bin, aligned_calculated_bin)
                        stats_bin['pearson_correlation'] = float(corr_bin) if not np.isnan(corr_bin) else None
                    except ValueError: # Can happen if std dev is zero
                        stats_bin['pearson_correlation'] = None
                else:
                    stats_bin['pearson_correlation'] = None
                
                report['binned_summary_stats'][generic_name][bin_label] = stats_bin
        
    except FileNotFoundError:
        report["error"] = f"File not found: {json_path}"
        print(report["error"])
    except json.JSONDecodeError:
        report["error"] = f"Error decoding JSON from file: {json_path}"
        print(report["error"])
    except ValueError as ve:
        report["error"] = str(ve)
        print(report["error"])
    except Exception as e:
        report["error"] = f"An unexpected error occurred: {e}"
        print(f"Error details: {type(e).__name__}, {e.args}") # More detailed error logging

    with open(output_report_path, 'w', encoding='utf-8') as f_out:
        json.dump(report, f_out, indent=4, default=lambda x: str(x) if pd.isna(x) else x)
    print(f"Internal feature comparison report saved to {output_report_path}")
    
    # Generate plots after saving the report
    output_dir = os.path.dirname(output_report_path)
    if not report.get("error"): # Only plot if main processing was successful
        generate_comparison_plots(report, output_dir, plot_suffix=plot_suffix)
    else:
        print(f"Skipping plot generation due to error in report: {report.get('error')}")


if __name__ == '__main__':
    input_json_file = "/Users/manojveluchuri/saas/r1/simpleui/backend/data/merged_song_data.json"
    
    # Exclude 'tempo' from this mapping
    feature_mapping = {
        'liveness': ('training_liveness', 'calculated_liveness'),
        'speechiness': ('training_speechiness', 'calculated_speechiness'),
        'valence': ('training_audio_valence', 'calculated_valence'),
        # 'tempo': ('training_tempo', 'calculated_tempo'), # Excluded
        'danceability': ('training_danceability', 'calculated_danceability'),
        'energy': ('training_energy', 'calculated_energy'),
        'loudness': ('training_loudness', 'calculated_loudness'),
        'acousticness': ('training_acousticness', 'calculated_acousticness')
    }
    
    plot_suffix_val = "_no_tempo"
    output_report_file = f"/Users/manojveluchuri/saas/r1/simpleui/backend/data/training_data/internal_feature_comparison_report_v3{plot_suffix_val}.json"

    print(f"Starting internal feature comparison for {input_json_file} with popularity binning (excluding tempo).")
    compare_features_in_json(input_json_file, feature_mapping, output_report_file, plot_suffix=plot_suffix_val)
    print(f"Internal feature comparison script (with binning, excluding tempo) finished.") 