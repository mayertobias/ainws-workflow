import csv
import os
import pandas as pd # Added import
# Placeholder for API client if you choose to use one, e.g., spotipy for Spotify
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

def deduplicate_csv(csv_path, output_path=None):
    """
    Deduplicate a CSV file by song_name and song_duration_ms.
    For duplicates, keeps the entry with the highest song_popularity.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the deduplicated CSV. 
                                    If None, overwrites the input file.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if output_path is None:
        output_path = csv_path
    
    try:
        # Read all entries from the CSV
        all_entries = []
        with open(csv_path, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"Error: The CSV file {csv_path} is empty or has no header.")
                return False
                
            fieldnames = reader.fieldnames
            required_fields = ["song_name", "song_duration_ms", "song_popularity"]
            if not all(field in fieldnames for field in required_fields):
                print(f"Error: CSV {csv_path} must contain all required fields: {required_fields}")
                print(f"Available fields: {fieldnames}")
                return False
                
            for row in reader:
                all_entries.append(row)
        
        if not all_entries:
            print(f"No data found in {csv_path} to deduplicate.")
            return False
            
        # Dictionary to track the best entry for each unique (song_name, duration) combo
        unique_entries = {}
        duplicates_removed = 0
        
        for entry in all_entries:
            # Create the unique key
            entry_key = (entry["song_name"], entry["song_duration_ms"])
            
            # Try to convert popularity to integer for comparison
            try:
                popularity = int(entry["song_popularity"])
            except (ValueError, TypeError):
                popularity = 0
                
            # If this key already exists, only keep the higher popularity version
            if entry_key in unique_entries:
                duplicates_removed += 1
                existing_entry = unique_entries[entry_key]
                
                try:
                    existing_popularity = int(existing_entry["song_popularity"])
                except (ValueError, TypeError):
                    existing_popularity = 0
                    
                if popularity > existing_popularity:
                    unique_entries[entry_key] = entry
                    print(f"Replaced duplicate for '{entry['song_name']}' with higher popularity version ({existing_popularity} → {popularity})")
            else:
                unique_entries[entry_key] = entry
                
        # Convert back to list for writing
        deduplicated_entries = list(unique_entries.values())
        
        # Write the deduplicated data
        with open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deduplicated_entries)
            
        print(f"Deduplication complete: Processed {len(all_entries)} entries")
        print(f"Removed {duplicates_removed} duplicates")
        print(f"Final dataset contains {len(deduplicated_entries)} unique entries")
        print(f"Saved to {output_path}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return False
    except Exception as e:
        print(f"An error occurred during deduplication: {e}")
        return False

def extract_initial_data(input_csv_path, output_csv_path):
    """
    Extracts song_name, song_popularity, and song_duration_ms from the input CSV
    and saves it to the output CSV.
    Performs deduplication based on song_name and song_duration_ms combination.
    """
    extracted_data = []
    header = ["song_name", "song_popularity", "song_duration_ms"]
    unique_songs = set()  # Track unique song name + duration combinations

    try:
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            # Verify required columns exist
            if not all(col in reader.fieldnames for col in ["song_name", "song_popularity", "song_duration_ms"]):
                print(f"Error: Input CSV {input_csv_path} is missing one of the required columns: song_name, song_popularity, song_duration_ms")
                expected_cols = ["song_name", "song_popularity", "song_duration_ms"]
                missing_cols = [col for col in expected_cols if col not in reader.fieldnames]
                print(f"Missing columns: {missing_cols}")
                print(f"Available columns: {reader.fieldnames}")
                return False

            total_rows = 0
            duplicates = 0
            for row in reader:
                total_rows += 1
                song_key = (row["song_name"], row["song_duration_ms"])
                
                # Skip duplicates
                if song_key in unique_songs:
                    duplicates += 1
                    continue
                
                unique_songs.add(song_key)
                extracted_data.append({
                    "song_name": row["song_name"],
                    "song_popularity": row["song_popularity"],
                    "song_duration_ms": row["song_duration_ms"]
                })
            
            print(f"Processed {total_rows} rows, found {duplicates} duplicates, extracted {len(extracted_data)} unique songs.")
        
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(extracted_data)
        print(f"Successfully extracted unique data to {output_csv_path}")
        return True
    except FileNotFoundError:
        print(f"Error: Input file {input_csv_path} not found.")
        return False
    except Exception as e:
        print(f"An error occurred during initial data extraction: {e}")
        return False

def get_artist_for_song(song_name: str, song_duration_ms: str) -> str:
    """
    Placeholder function to find the artist for a given song name and duration.
    YOU NEED TO IMPLEMENT THIS FUNCTION using a music API like Spotify, MusicBrainz, etc.

    Args:
        song_name (str): The name of the song.
        song_duration_ms (str): The duration of the song in milliseconds.

    Returns:
        str: The determined artist name, or "Unknown Artist" if not found.
    """
    print(f"Attempting to find artist for: {song_name} (Duration: {song_duration_ms}ms)")
    
    # --- START OF YOUR IMPLEMENTATION ---
    # Example pseudocode for using an API (e.g., Spotify):
    # 1. Initialize your API client (e.g., Spotipy)
    #    client_credentials_manager = SpotifyClientCredentials(client_id="YOUR_CLIENT_ID", client_secret="YOUR_CLIENT_SECRET")
    #    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    #
    # 2. Search for the track:
    #    results = sp.search(q=f'track:{song_name}', type='track', limit=10)
    #    tracks = results.get('tracks', {}).get('items', [])
    #
    # 3. Iterate through results and match by duration:
    #    try:
    #        target_duration_ms = int(song_duration_ms)
    #    except ValueError:
    #        print(f"Warning: Could not parse duration '{song_duration_ms}' for {song_name}")
    #        return "Unknown Artist (Bad Duration)"

    #    for track in tracks:
    #        api_duration_ms = track.get('duration_ms')
    #        if api_duration_ms:
    #            # Allow for a tolerance (e.g., +/- 5 seconds)
    #            if abs(api_duration_ms - target_duration_ms) < 5000:
    #                artists = track.get('artists', [])
    #                if artists:
    #                    artist_names = [artist['name'] for artist in artists]
    #                    print(f"Found artist(s): {', '.join(artist_names)} for {song_name}")
    #                    return ", ".join(artist_names) # Join if multiple artists

    # If no suitable match is found after checking all results:
    # print(f"Could not find a matching artist for {song_name} with duration {song_duration_ms}ms")
    # return "Unknown Artist"
    # --- END OF YOUR IMPLEMENTATION ---

    # As this is a placeholder, we'll return a default value.
    # Remove this line once you implement the actual API calls.
    return "Unknown Artist (Placeholder)"


def enrich_with_artist_names(csv_path):
    """
    Reads the given CSV, fetches artist names for each song, 
    and writes the artist name back to the same CSV file in a new column.
    Uses a cache to avoid redundant API calls for the same song.
    """
    rows = []
    fieldnames = []
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"Error: The CSV file {csv_path} is empty or has no header.")
                return False
            fieldnames = reader.fieldnames
            if "song_name" not in fieldnames or "song_duration_ms" not in fieldnames:
                print(f"Error: CSV {csv_path} must contain 'song_name' and 'song_duration_ms' columns.")
                return False
            
            for row in reader:
                rows.append(row)
        
        if not rows:
            print(f"No data found in {csv_path} to enrich.")
            return False

        updated_rows = []
        new_fieldnames = fieldnames + ["artist_name"] if "artist_name" not in fieldnames else fieldnames

        # Cache for artist lookups to avoid redundant API calls
        artist_cache = {}

        for i, row in enumerate(rows):
            song_name = row.get("song_name")
            song_duration_ms = row.get("song_duration_ms")
            song_key = (song_name, song_duration_ms)
            
            print(f"Processing row {i+1}/{len(rows)}: {song_name}")
            
            # Check if we already have artist info for this song
            if song_key in artist_cache:
                print(f"  Using cached artist for: {song_name}")
                artist_name = artist_cache[song_key]
            else:
                # Look up artist and cache the result
                artist_name = get_artist_for_song(song_name, song_duration_ms)
                artist_cache[song_key] = artist_name
                
            row["artist_name"] = artist_name
            updated_rows.append(row)

        with open(csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        print(f"Successfully enriched {csv_path} with artist names.")
        return True

    except FileNotFoundError:
        print(f"Error: File {csv_path} not found for enrichment.")
        return False
    except Exception as e:
        print(f"An error occurred during artist enrichment: {e}")
        return False

def enhance_data_with_external_file(main_csv_path, external_csv_path, output_csv_path):
    """
    Enhances the main CSV by cross-referencing with an external CSV.
    Uses song_name, song_duration_ms, and song_popularity as keys for merging.
    Adds all other columns from the external CSV to the main CSV.

    Args:
        main_csv_path (str): Path to the main CSV file to be enhanced.
        external_csv_path (str): Path to the external CSV file for cross-referencing.
        output_csv_path (str): Path to save the enhanced CSV.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        print(f"Reading main CSV: {main_csv_path}")
        main_df = pd.read_csv(main_csv_path)
        print(f"Reading external CSV: {external_csv_path}")
        external_df = pd.read_csv(external_csv_path)

        # Define the merge keys
        # Assuming 'song_name', 'song_duration_ms', 'song_popularity' are the column names.
        # Adjust if column names are different in song_data.csv
        merge_keys = ["song_name", "song_duration_ms", "song_popularity"]

        # Check if merge keys exist in both dataframes
        for key in merge_keys:
            if key not in main_df.columns:
                print(f"Error: Merge key '{key}' not found in {main_csv_path}")
                print(f"Available columns: {main_df.columns.tolist()}")
                return False
            if key not in external_df.columns:
                print(f"Error: Merge key '{key}' not found in {external_csv_path}")
                print(f"Available columns: {external_df.columns.tolist()}")
                return False
        
        # Convert merge key columns to string to ensure consistent merging if types are mixed
        for key in merge_keys:
            main_df[key] = main_df[key].astype(str)
            external_df[key] = external_df[key].astype(str)


        print(f"Merging dataframes on keys: {merge_keys}")
        # Perform a left merge to keep all rows from the main_df and add matching data from external_df
        enhanced_df = pd.merge(main_df, external_df, on=merge_keys, how='left', suffixes=('', '_external'))

        # Identify columns from external_df that were newly added (excluding merge keys)
        # and handle potential duplicate column names by prioritizing original main_df columns
        # or by renaming _external columns.
        
        # For simplicity, if a column exists in both and is not a merge key,
        # the one from main_df is kept, and the one from external_df (now with _external suffix)
        # can be dropped if it's an exact duplicate content-wise or handled as needed.
        # Here, we just keep both with suffixes if names clash outside merge keys.

        print(f"Saving enhanced data to {output_csv_path}")
        enhanced_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        print(f"Successfully enhanced data and saved to {output_csv_path}")
        print(f"Original main CSV rows: {len(main_df)}, External CSV rows: {len(external_df)}")
        print(f"Enhanced CSV rows: {len(enhanced_df)}")
        return True

    except FileNotFoundError:
        print(f"Error: One of the CSV files was not found. Searched for {main_csv_path} and {external_csv_path}")
        return False
    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the CSV files is empty. {e}")
        return False
    except Exception as e:
        print(f"An error occurred during data enhancement: {e}")
        return False

def main():
    # Assuming the script is in backend/data/training_data/
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    input_file = os.path.join(base_dir, "song_data.csv")
    output_file = os.path.join(base_dir, "r4a_song_data_training.csv")

    print(f"Input file path: {input_file}")
    print(f"Output file path: {output_file}")

    # Check if any arguments were passed
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # Just deduplicate the training data file
        if command == "deduplicate":
            target_file = "/Users/manojveluchuri/saas/r1/simpleui/backend/data/training_data/r4a_song_data_training.csv"
            print(f"Running deduplication on {target_file}")
            if deduplicate_csv(target_file):
                print("Deduplication completed successfully.")
            else:
                print("Deduplication failed.")
            return
        
        elif command == "enhance":
            main_file_to_enhance = "/Users/manojveluchuri/saas/r1/simpleui/backend/data/training_data/r4a_song_data_training.csv"
            # Assuming song_data.csv is in the same directory as the script
            external_source_file = os.path.join(base_dir, "song_data.csv") 
            output_enhanced_file = "/Users/manojveluchuri/saas/r1/simpleui/backend/data/training_data/r4a_trng_data_with_ext_f_for_cmp.csv"

            print(f"Running enhancement:")
            print(f"  Main file: {main_file_to_enhance}")
            print(f"  External source: {external_source_file}")
            print(f"  Output file: {output_enhanced_file}")
            
            if not os.path.exists(main_file_to_enhance):
                print(f"Error: Main file for enhancement not found: {main_file_to_enhance}")
                return
            if not os.path.exists(external_source_file):
                print(f"Error: External source file not found: {external_source_file}")
                return

            if enhance_data_with_external_file(main_file_to_enhance, external_source_file, output_enhanced_file):
                print("Data enhancement completed successfully.")
            else:
                print("Data enhancement failed.")
            return
    
    # Normal processing flow
    if not os.path.exists(input_file):
        print(f"Error: The input file song_data.csv was not found in the directory {base_dir}")
        print("Please ensure 'song_data.csv' is in the same directory as this script.")
        return

    if extract_initial_data(input_file, output_file):
        print("Initial data extraction complete.")
        print(f"Attempting to enrich {output_file} with artist names...")
        if enrich_with_artist_names(output_file):
            print("Artist enrichment process completed.")
        else:
            print("Artist enrichment failed or was skipped.")
    else:
        print("Initial data extraction failed. Skipping artist enrichment.")

if __name__ == "__main__":
    main() 