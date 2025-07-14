import csv
import os
import psycopg2 # Make sure to install with: pip install psycopg2-binary

# --- BEGIN USER CONFIGURATION ---
DB_HOST = "20.55.0.20"
DB_PORT = "5432"
DB_NAME = "musicbrainz"  # Your MusicBrainz database name
DB_USER = "postgres"
DB_PASSWORD = "artorama##01"

# Input CSV file (created by extract_and_enrich_songs.py or similar)
INPUT_CSV_FILE = "r4a_song_data_training.csv" 
# Output CSV file with enriched data
OUTPUT_CSV_FILE = "r4a_song_data_enriched_from_db.csv"

# Temporary table name in the database
TEMP_TABLE_NAME = "r4a_song_data_temp_staging"
# Duration tolerance for matching songs in milliseconds (e.g., 5000ms = 5 seconds)
DURATION_TOLERANCE_MS = 5000
# Batch size for processing songs from the staging table
BATCH_SIZE = 500  # Adjust as needed (e.g., 100, 500, 1000)
# --- END USER CONFIGURATION ---

def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def check_table_accessibility(conn, cursor):
    """Checks if the required MusicBrainz tables are accessible."""
    required_tables = {
        "musicbrainz.recording": "(should contain song titles, lengths, artist_credit FK)",
        "musicbrainz.artist_credit": "(should link recordings to artist credit names)",
        "musicbrainz.artist_credit_name": "(should link artist_credits to actual artists and their positions)",
        "musicbrainz.artist": "(should contain artist names and GIDs/MBIDs)",
        "musicbrainz.isrc": "(should link ISRCs to recordings)"
    }
    all_tables_accessible = True
    print("\n--- Checking Accessibility of Required MusicBrainz Tables ---")
    
    for table_name, description in required_tables.items():
        try:
            # Try to select a single row to check for existence and permissions
            # Using COUNT(*) can be slow on very large tables if not needed.
            # SELECT 1 is faster just for an existence/permission check.
            cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1;")
            # cursor.fetchone() # We don't actually need the data, just that it didn't error
            print(f"SUCCESS: Table '{table_name}' {description} is accessible.")
        except psycopg2.Error as e:
            print(f"FAILURE: Could not access table '{table_name}' {description}.")
            print(f"  Error: {e}")
            print(f"  Please verify that the table '{table_name}' exists and that the user '{DB_USER}' has SELECT permissions.")
            print(f"  If your table has a different name, you will need to adjust the main SQL query in enrich_data_and_export().")
            all_tables_accessible = False
        except Exception as e:
            print(f"FAILURE: An unexpected error occurred while checking table '{table_name}'. Error: {e}")
            all_tables_accessible = False

    if all_tables_accessible:
        print("All required MusicBrainz tables appear to be accessible.")
    else:
        print("One or more required MusicBrainz tables are not accessible. Please review the errors above.")
    print("--- End Table Accessibility Check ---\n")
    return all_tables_accessible

def create_and_load_staging_table(conn, cursor, input_csv_path):
    """Creates a temporary staging table and loads data from the input CSV."""
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {TEMP_TABLE_NAME};")
        cursor.execute(f"""
        CREATE TABLE {TEMP_TABLE_NAME} (
            song_name TEXT,
            song_popularity INTEGER,
            song_duration_ms INTEGER,
            artist_name TEXT
        );
        """)
        conn.commit()
        print(f"Table {TEMP_TABLE_NAME} created successfully.")

        with open(input_csv_path, 'r', encoding='utf-8') as f:
            # Skip header row for COPY command
            next(f) 
            cursor.copy_expert(f"COPY {TEMP_TABLE_NAME} FROM STDIN WITH CSV", f)
        conn.commit()
        print(f"Data from {input_csv_path} loaded into {TEMP_TABLE_NAME}.")
        return True
    except psycopg2.Error as e:
        print(f"Error during staging table creation or loading: {e}")
        conn.rollback()
        return False
    except FileNotFoundError:
        print(f"Error: Input CSV file {input_csv_path} not found.")
        return False


def enrich_data_and_export(conn, cursor, output_csv_path):
    """
    Queries the database to enrich the staged song data with MusicBrainz info
    and exports the result to a new CSV file.
    
    THIS IS THE CORE FUNCTION YOU MAY NEED TO CUSTOMIZE, ESPECIALLY THE SQL QUERY.
    """
    
    header = []
    first_batch = True
    offset = 0
    total_rows_processed = 0

    try:
        # Get total number of rows to process for progress reporting
        cursor.execute(f"SELECT COUNT(*) FROM {TEMP_TABLE_NAME}")
        total_staged_rows = cursor.fetchone()[0]
        if total_staged_rows == 0:
            print(f"No rows found in staging table {TEMP_TABLE_NAME} to process.")
            return False
        print(f"Found {total_staged_rows} songs in the staging table to enrich in batches of {BATCH_SIZE}.")

        while offset < total_staged_rows:
            print(f"Processing batch starting at offset {offset}... ({total_rows_processed}/{total_staged_rows} processed so far)")
            
            # The main query is now wrapped with a CTE to process in batches
            # The ORDER BY inside the CTE is important for consistent pagination with OFFSET
            batch_sql_query = f"""
            WITH current_batch AS (
                SELECT *
                FROM {TEMP_TABLE_NAME}
                ORDER BY song_name, song_duration_ms -- Ensure consistent ordering for pagination
                LIMIT {BATCH_SIZE} OFFSET {offset}
            )
            SELECT
                s.song_name,
                s.song_popularity,
                s.song_duration_ms,
                s.artist_name AS original_artist_name, -- Keep the original placeholder artist
                main_artist.name AS mb_artist_name,
                main_artist.gid AS artist_mbid,
                STRING_AGG(DISTINCT isrc.isrc, ', ') AS isrcs
            FROM
                current_batch s
            LEFT JOIN
                musicbrainz.recording rec ON s.song_name = rec.name
                                        AND rec.length BETWEEN (s.song_duration_ms - {DURATION_TOLERANCE_MS})
                                                           AND (s.song_duration_ms + {DURATION_TOLERANCE_MS})
            LEFT JOIN
                musicbrainz.artist_credit ac ON rec.artist_credit = ac.id
            LEFT JOIN
                musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit AND acn.position = 0
            LEFT JOIN
                musicbrainz.artist main_artist ON acn.artist = main_artist.id
            LEFT JOIN
                musicbrainz.isrc isrc ON rec.id = isrc.recording
            GROUP BY
                s.song_name,
                s.song_popularity,
                s.song_duration_ms,
                s.artist_name, -- Group by the original artist name as well
                main_artist.name,
                main_artist.gid
            ORDER BY
                s.song_name;
            """

            if first_batch:
                print("\n--- SQL Query for Enrichment (Batch Version) ---")
                print(batch_sql_query) # Print query for the first batch for review
                print("--- End SQL Query ---\n")
                print("Please review the SQL query above. You may need to adjust table names, column names, and join conditions.")

            cursor.execute(batch_sql_query)
            results = cursor.fetchall()
            
            current_batch_row_count = len(results)

            if first_batch:
                if not results:
                    print("No enriched data found in the first batch. The SQL query might not be returning any rows or there might be an issue.")
                    # Optionally, you could decide to return False here if the first batch must yield results.
                header = [desc[0] for desc in cursor.description]
                with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    if results:
                        writer.writerows(results)
                first_batch = False
            elif results: # Only append if there are results for subsequent batches
                with open(output_csv_path, 'a', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(results)
            
            if current_batch_row_count > 0:
                 print(f"Successfully processed and wrote {current_batch_row_count} rows for this batch.")
            else:
                 print(f"No matching data found in MusicBrainz for this batch (offset {offset}).")

            total_rows_processed += BATCH_SIZE # Assume we attempted to process BATCH_SIZE, actual found rows might be less
            offset += BATCH_SIZE

        print(f"\nEnrichment data export to {output_csv_path} complete. Total songs attempted: {total_staged_rows}.")
        return True

    except psycopg2.Error as e:
        print(f"Database error during enrichment query or export: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during enrichment and export: {e}")
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_full_path = os.path.join(base_dir, INPUT_CSV_FILE)
    output_csv_full_path = os.path.join(base_dir, OUTPUT_CSV_FILE)

    conn = get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cursor:
            if not check_table_accessibility(conn, cursor):
                print("Aborting due to inaccessible MusicBrainz tables.")
                return

            print(f"Preparing to process {input_csv_full_path}...")
            if not create_and_load_staging_table(conn, cursor, input_csv_full_path):
                print("Failed to create or load staging table. Aborting.")
                return

            print("Attempting to enrich data from database...")
            if not enrich_data_and_export(conn, cursor, output_csv_full_path):
                print("Enrichment process failed or produced no data.")
            else:
                print("Enrichment and export process completed.")
            
            # Optional: Clean up the temporary table
            # print(f"Cleaning up temporary table {TEMP_TABLE_NAME}...")
            # cursor.execute(f"DROP TABLE IF EXISTS {TEMP_TABLE_NAME};")
            # conn.commit()
            # print("Temporary table dropped.")

    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    print("Starting database enrichment script...")
    main()
    print("Script finished.")

    # --- TEMPORARY CONNECTION TEST ---
    # print("Attempting to test database connection...")
    # conn = get_db_connection()
    # if conn:
    #     print("SUCCESS: Database connection established successfully!")
    #     try:
    #         conn.close()
    #         print("Database connection closed.")
    #     except Exception as e:
    #         print(f"Error while closing connection: {e}")
    # else:
    #     print("FAILURE: Database connection could not be established.")
    #     print("Please check the DB_HOST, DB_PORT, DB_NAME, DB_USER, and DB_PASSWORD variables in this script.")
    #     print("Also, ensure your PostgreSQL server is running and accessible.")
    # --- END TEMPORARY CONNECTION TEST --- 