import os
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import lyricsgenius
import json
from datetime import datetime
import hashlib
import math
import aiohttp
from bs4 import BeautifulSoup
import re

# Import components from local services
from services.youtube_downloader import YouTubeDownloader
from services.youtube_api import YouTubeAPI
from services.billboard_scraper import BillboardScraper
from services.proxy_manager import ProxyManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SongDownloader:
    def __init__(self, csv_path: str, songs_per_bin_batch: int = 5, state_file_path: Optional[str] = None, min_popularity_bin: int = 0):
        """
        Initialize the song downloader.
        
        Args:
            csv_path (str): Path to the CSV file with song data
            songs_per_bin_batch (int): Number of songs to download per popularity bin in each round-robin cycle.
            state_file_path (Optional[str]): Path to a specific state file to use for tracking (overrides default)
            min_popularity_bin (int): Minimum popularity bin to consider for downloading (0-10, default: 0)
        """
        self.csv_path = csv_path
        self.songs_per_bin_batch = songs_per_bin_batch
        self.min_popularity_bin = min_popularity_bin
        
        # Get API keys from environment variables
        self.api_keys = self._get_youtube_api_keys()
        if not self.api_keys:
            raise ValueError("No YouTube API keys found in environment variables")
            
        self.genius_token = os.getenv("GENIUS_API_TOKEN")
        
        # Initialize directories with absolute paths
        self.base_dir = "/Users/manojveluchuri/saas/r1/simpleui/backend"
        self.data_dir = os.path.join(self.base_dir, "data")
        self.downloads_dir = os.path.join(self.base_dir, "downloads")
        self.lyrics_dir = os.path.join(self.base_dir, "lyrics")
        self.state_dir = os.path.join(self.base_dir, "state")
        
        # Create necessary directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.downloads_dir, exist_ok=True)
        os.makedirs(self.lyrics_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Initialize state tracking
        if state_file_path:
            self.state_file = state_file_path
            logger.info(f"Using provided state file: {state_file_path}")
        else:
            self.state_file = os.path.join(self.state_dir, f"download_state_{self._get_file_hash(csv_path)}.json")
            logger.info(f"Using default state file: {self.state_file}")
            
        self.state = self._load_state()
        
        # Initialize YouTube components
        self.youtube_api = YouTubeAPI(api_keys=self.api_keys)
        
        # Initialize Genius API client if token is provided
        if self.genius_token:
            self.genius = lyricsgenius.Genius(self.genius_token)
            self.genius.verbose = False
            self.genius.remove_section_headers = True
            self.genius.skip_non_songs = True
            self.genius.excluded_terms = ["(Remix)", "(Live)"]
            
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the input file to track state."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    def _load_state(self) -> Dict[str, Any]:
        """Load the download state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Handle potential differences in state file structure
                # Make sure all required keys exist
                required_keys = {"total_songs", "completed_songs", "batches"}
                
                # Convert older state format if needed
                if "downloaded_songs" in state and "failed_songs" in state:
                    logger.info("Converting older state format to new format")
                    if "processed_songs_count" not in state:
                        state["processed_songs_count"] = state.get("downloaded_songs", 0) + state.get("failed_songs", 0)
                    
                    if "successful_audio_downloads" not in state:
                        state["successful_audio_downloads"] = state.get("downloaded_songs", 0)
                    
                    if "successful_lyrics_downloads" not in state:
                        state["successful_lyrics_downloads"] = 0
                    
                    if "failed_songs_count" not in state:
                        state["failed_songs_count"] = state.get("failed_songs", 0)
                    
                    if "failed_songs_info" not in state and "failed_songs_list" in state:
                        # Convert older failed_songs_list format to failed_songs_info
                        state["failed_songs_info"] = []
                        for failed_song in state.get("failed_songs_list", []):
                            state["failed_songs_info"].append({
                                "song_name": failed_song.get("song_name", ""),
                                "artist": "Unknown Artist",
                                "bin": "",
                                "reason": failed_song.get("error", "Unknown error")
                            })
                
                # Add any missing required keys with default values
                for key in required_keys:
                    if key not in state:
                        if key == "completed_songs":
                            state[key] = []
                        elif key == "batches":
                            state[key] = []
                        else:
                            state[key] = 0
                
                logger.info(f"Loaded state file with {len(state.get('completed_songs', []))} completed songs")
                return state
            except Exception as e:
                logger.error(f"Error loading state file: {e}")
                return self._create_initial_state()
        return self._create_initial_state()
        
    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial state structure."""
        logger.info("Creating new state file")
        return {
            "last_run": None,
            "total_songs": 0,
            "processed_songs_count": 0,
            "successful_audio_downloads": 0,
            "successful_lyrics_downloads": 0,
            "failed_songs_count": 0,
            "completed_songs": [],
            "failed_songs_info": [],
            "current_batch": 0,
            "batches": []
        }
        
    def _save_state(self) -> None:
        """Save the current state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state file: {e}")
            
    def _get_youtube_api_keys(self) -> List[str]:
        """Get YouTube API keys from environment variables."""
        api_key = os.getenv("YOUTUBE_API_KEY")  # Fixed typo in environment variable name
        if not api_key:
            logger.error("No YouTube API key found in environment variables")
            return []
            
        # Split by comma and strip whitespace and newlines
        api_keys = [key.strip() for key in api_key.replace('\n', '').split(',') if key.strip()]
        logger.info(f"Loaded {len(api_keys)} YouTube API keys")
        return api_keys
        
    def _get_popularity_bin(self, popularity: float) -> str:
        """Get the popularity bin name for a given popularity score."""
        # Convert popularity to integer and calculate bin
        pop_int = math.ceil(popularity)
        bin_num = math.ceil(pop_int / 10)
        return f"popularity_{bin_num}"
        
    def load_song_data(self) -> Optional[pd.DataFrame]:
        """
        Load and process song data from CSV.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with song information
        """
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV file not found: {self.csv_path}")
                return None
                
            logger.info(f"Loading song data from {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Create popularity bins based on ranges
            df['popularity_bin'] = df['song_popularity'].apply(self._get_popularity_bin)
            
            # Add empty artist column if it doesn't exist or clean existing ones
            if 'artist_name' not in df.columns:
                logger.info("No artist_name column found in CSV. Creating default column.")
                df['artist_name'] = ""
            else:
                # Clean up any "Unknown Artist" values or placeholders to be consistent empty strings
                df['artist_name'] = df['artist_name'].apply(
                    lambda x: "" if pd.isna(x) or str(x).lower() in ["unknown artist", "unknown artist (placeholder)"] else str(x)
                )
                logger.info(f"Cleaned artist_name column - removed {df['artist_name'].isna().sum()} NA values and standardized Unknown Artist entries")
            
            # Create directories for each popularity bin
            for bin_name in df['popularity_bin'].unique():
                # Create directory for downloads
                bin_dir = os.path.join(self.downloads_dir, bin_name)
                os.makedirs(bin_dir, exist_ok=True)
                
                # Create directory for lyrics
                lyrics_bin_dir = os.path.join(self.lyrics_dir, bin_name)
                os.makedirs(lyrics_bin_dir, exist_ok=True)
            
            # Update state with total songs
            self.state["total_songs"] = len(df)
            self.state["last_run"] = datetime.now().isoformat()
            self._save_state()
            
            logger.info(f"Loaded {len(df)} songs, organized into 10 popularity bins")
            return df
            
        except Exception as e:
            logger.error(f"Error loading song data: {e}")
            return None
            
    async def fetch_lyrics(self, song_name: str, artist: Optional[str] = None) -> Optional[str]:
        """
        Fetch lyrics for a song using Genius API.
        
        Args:
            song_name (str): Name of the song
            artist (Optional[str]): Name of the artist
            
        Returns:
            Optional[str]: Lyrics text if successful, None otherwise
        """
        if not self.genius_token:
            logger.warning("Genius API token not provided. Skipping lyrics fetching.")
            return None
            
        try:
            # Search for the song
            song = self.genius.search_song(song_name, artist)
            if song:
                return song.lyrics
            return None
        except Exception as e:
            logger.error(f"Error fetching lyrics for {song_name}: {e}")
            return None
            
    async def save_lyrics(self, lyrics: str, song_name: str, popularity_bin: str) -> str:
        """
        Save lyrics to a file.
        
        Args:
            lyrics (str): Lyrics text
            song_name (str): Name of the song
            popularity_bin (str): Popularity bin directory
            
        Returns:
            str: Path to the saved lyrics file
        """
        # Sanitize filename
        safe_song_name = "".join(c for c in song_name if c.isalnum() or c in (' ', '-', '_')).strip()
        lyrics_path = os.path.join(self.lyrics_dir, popularity_bin, f"{safe_song_name}.txt")
        
        with open(lyrics_path, 'w', encoding='utf-8') as f:
            f.write(lyrics)
            
        return lyrics_path
        
    def _is_song_completed(self, song_name: str, popularity_bin: str) -> bool:
        """Check if a song has been successfully downloaded and processed."""
        # Check if song is in completed list
        if song_name in self.state["completed_songs"]:
            return True
            
        # Check if song exists in any popularity directory
        song_filename = f"{song_name}.mp3"
        song_path = os.path.join(self.downloads_dir, popularity_bin, song_filename)
        if os.path.exists(song_path):
            # Add to completed songs if found
            self.state["completed_songs"].append(song_name)
            self._save_state()
            return True
                
        return False
        
    async def _download_lyrics(self, song_name: str, artist: Optional[str] = None, popularity_bin: str = "") -> bool:
        """Download lyrics for a song using Genius API."""
        if not self.genius_token:
            logger.warning("No Genius API token provided, skipping lyrics download")
            return False
            
        try:
            # Check for any "Unknown Artist (Placeholder)" values and convert to empty string
            if artist and artist.lower() in ["unknown artist", "unknown artist (placeholder)"]:
                artist = ""
                
            # Create lyrics directory for the bin if it doesn't exist
            bin_lyrics_dir = os.path.join(self.lyrics_dir, str(popularity_bin))
            os.makedirs(bin_lyrics_dir, exist_ok=True)
            
            # Clean the song name for file paths
            clean_song_name = song_name.replace("/", "_").replace("\\", "_")
            
            # Check if lyrics file already exists
            lyrics_file = os.path.join(bin_lyrics_dir, f"{clean_song_name}.txt")
            artist_lyrics_file = None
            if artist and artist.strip():
                artist_lyrics_file = os.path.join(bin_lyrics_dir, f"{clean_song_name} - {artist}.txt")
                
            # Check if either file already exists
            if os.path.exists(lyrics_file):
                logger.info(f"Lyrics already exist at: {lyrics_file}")
                return True
                
            if artist_lyrics_file and os.path.exists(artist_lyrics_file):
                logger.info(f"Lyrics already exist at: {artist_lyrics_file}")
                return True
            
            # Search query - Only include artist if it has meaningful content
            search_query = song_name
            if artist and artist.strip():
                search_query = f"{song_name} {artist}"
                
            logger.info(f"Searching lyrics for: {search_query}")
            search_url = f"https://api.genius.com/search?q={search_query}"
            headers = {"Authorization": f"Bearer {self.genius_token}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Genius API error: {response.status}")
                        return False
                        
                    data = await response.json()
                    hits = data.get("response", {}).get("hits", [])
                    
                    # If no results found with artist, try again with just the song name
                    if not hits and artist and artist.strip():
                        logger.info(f"No lyrics found with artist name. Retrying with just song name: {song_name}")
                        backup_url = f"https://api.genius.com/search?q={song_name}"
                        
                        async with session.get(backup_url, headers=headers) as backup_response:
                            if backup_response.status != 200:
                                logger.error(f"Genius API error on backup search: {backup_response.status}")
                                return False
                                
                            backup_data = await backup_response.json()
                            hits = backup_data.get("response", {}).get("hits", [])
                    
                    if not hits:
                        logger.warning(f"No lyrics found for {song_name}")
                        return False
                        
                    # Get the first hit's URL
                    song_url = hits[0]["result"]["url"]
                    
                    # Scrape the lyrics from the page
                    async with session.get(song_url) as page_response:
                        if page_response.status != 200:
                            logger.error(f"Failed to fetch lyrics page: {page_response.status}")
                            return False
                            
                        html = await page_response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        
                        # Find the lyrics container
                        lyrics_div = soup.find("div", {"class": "Lyrics__Container-sc-1ynbvzw-1"})
                        if not lyrics_div:
                            logger.warning(f"Could not find lyrics container for {song_name}")
                            return False
                            
                        # Extract and clean the lyrics
                        lyrics = lyrics_div.get_text("\n")
                        lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Verse], [Chorus], etc.
                        lyrics = re.sub(r'\n\s*\n', '\n', lyrics)  # Remove empty lines
                        lyrics = lyrics.strip()
                        
                        # Save the lyrics - include artist in filename if available and meaningful
                        selected_file_path = lyrics_file
                        if artist and artist.strip():
                            selected_file_path = artist_lyrics_file
                            
                        with open(selected_file_path, "w", encoding="utf-8") as f:
                            f.write(lyrics)
                            
                        artist_info = f" by {artist}" if artist and artist.strip() else ""
                        logger.info(f"Downloaded lyrics for {song_name}{artist_info} to {selected_file_path}")
                        return True
                        
        except Exception as e:
            logger.error(f"Error downloading lyrics for {song_name}: {str(e)}")
            return False

    async def download_songs(self):
        """Download songs and their lyrics using a round-robin strategy across popularity bins."""
        try:
            df = self.load_song_data()
            if df is None or df.empty:
                logger.error("No song data loaded. Exiting download process.")
                return
                
            # Get unique popularity bins, sorted to ensure consistent processing order
            popularity_bins = sorted(df['popularity_bin'].unique())
            if not popularity_bins:
                logger.error("No popularity bins found in the data. Exiting download process.")
                return
                
            # Filter bins based on minimum popularity threshold
            popularity_bins = [bin_name for bin_name in popularity_bins 
                             if int(bin_name.split('_')[1]) >= self.min_popularity_bin]
            
            if not popularity_bins:
                logger.error(f"No popularity bins found above threshold {self.min_popularity_bin}. Exiting download process.")
                return
                
            logger.info(f"Starting round-robin download for {len(df)} songs across {len(popularity_bins)} bins (min bin: {self.min_popularity_bin}), with batch size {self.songs_per_bin_batch}.")
            self.state["total_songs"] = len(df) # Ensure total_songs is up-to-date

            songs_processed_in_current_full_cycle = True # Flag to continue main loop
            while songs_processed_in_current_full_cycle:
                songs_processed_in_current_full_cycle = False # Reset for this cycle
                
                # Check if all songs are completed before starting a new cycle
                if len(self.state["completed_songs"]) + len(self.state["failed_songs_info"]) >= self.state["total_songs"]:
                    logger.info("All songs have been processed (either completed or marked as failed).")
                    break

                for bin_name in popularity_bins:
                    # Get songs for the current bin that are not yet completed
                    bin_songs_df = df[(df['popularity_bin'] == bin_name) & (~df['song_name'].isin(self.state["completed_songs"]))]
                    
                    # Further filter out songs that are already in the failed list for this run to avoid reprocessing indefinitely in a single run
                    # Note: A more robust failure handling might involve retry counts, etc.
                    failed_song_names_this_run = [f_song['song_name'] for f_song in self.state["failed_songs_info"]]
                    pending_bin_songs_df = bin_songs_df[~bin_songs_df['song_name'].isin(failed_song_names_this_run)]

                    if pending_bin_songs_df.empty:
                        # logger.info(f"No pending songs in bin {bin_name} for this cycle.")
                        continue

                    # Select a batch from this bin
                    batch_to_download_df = pending_bin_songs_df.head(self.songs_per_bin_batch)
                    if batch_to_download_df.empty:
                        continue
                        
                    logger.info(f"Processing batch of {len(batch_to_download_df)} songs from bin: {bin_name}")
                    songs_processed_in_current_full_cycle = True # Mark that we found songs to process in this cycle

                    tasks = []
                    songs_in_batch_info = [] # To map results back to song names
                    for _, row in batch_to_download_df.iterrows():
                        song_name = row['song_name']
                        # Get artist name if available, otherwise use None
                        artist = row.get('artist_name', None)
                        # popularity_bin is already bin_name
                        
                        songs_in_batch_info.append({"name": song_name, "artist": artist, "bin": bin_name})
                
                        # Create tasks for both song and lyrics download
                        # Using new helper methods for clarity
                        song_task = asyncio.create_task(
                            self._download_single_song_audio(song_name, artist, bin_name)
                        )
                        lyrics_task = asyncio.create_task(
                            self._download_lyrics(song_name, artist, bin_name) # Existing method
                        )
                        tasks.extend([song_task, lyrics_task])
                
                    if not tasks:
                        continue

                    # Wait for all tasks in this small batch to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)
            
                    # Process results for this batch
                    for i in range(len(batch_to_download_df)):
                        song_info = songs_in_batch_info[i]
                        song_name = song_info["name"]
                        artist = song_info["artist"]
                        
                        audio_result_index = i * 2
                        lyrics_result_index = i * 2 + 1
                        
                        audio_downloaded_successfully = False
                        lyrics_downloaded_successfully = False
                        failure_reason = None

                        # Process audio download result
                        audio_res = results[audio_result_index]
                        if isinstance(audio_res, Exception):
                            logger.error(f"Audio download task for '{song_name}' failed: {str(audio_res)}")
                            failure_reason = f"Audio download error: {str(audio_res)}"
                        elif audio_res is False: # Assuming _download_single_song_audio returns False on failure
                            logger.warning(f"Audio download for '{song_name}' indicated failure.")
                            failure_reason = "Audio download failed (returned False)."
                        elif audio_res is True: # Assuming _download_single_song_audio returns True on success
                            audio_downloaded_successfully = True
                            self.state["successful_audio_downloads"] += 1
                            logger.info(f"Successfully downloaded audio for: {song_name}")
                        
                        # Process lyrics download result (only if audio was attempted, successful or not)
                        lyrics_res = results[lyrics_result_index]
                        if isinstance(lyrics_res, Exception):
                            logger.error(f"Lyrics download task for '{song_name}' failed: {str(lyrics_res)}")
                            # Don't overwrite audio failure reason if it exists
                            if not failure_reason: failure_reason = f"Lyrics download error: {str(lyrics_res)}"
                        elif lyrics_res is True: # _download_lyrics returns True on success
                            lyrics_downloaded_successfully = True
                            self.state["successful_lyrics_downloads"] += 1
                            logger.info(f"Successfully downloaded lyrics for: {song_name}")
                        
                        # Update overall state based on batch results
                        if audio_downloaded_successfully:
                            if song_name not in self.state["completed_songs"]:
                                self.state["completed_songs"].append(song_name)
                                self.state["processed_songs_count"] +=1
                        else:
                            # If audio failed, the whole song processing is considered failed for this attempt
                            if song_name not in failed_song_names_this_run and song_name not in self.state["completed_songs"]:
                                self.state["failed_songs_count"] += 1
                                self.state["failed_songs_info"].append({
                                    "song_name": song_name, 
                                    "artist": artist, 
                                    "bin": bin_name,
                                    "reason": failure_reason or "Unknown audio download failure"
                                })
                                self.state["processed_songs_count"] += 1 # It was processed (attempted)

                    # Save state after each bin's batch is processed
                    self.state["last_run"] = datetime.now().isoformat()
                    self._save_state()
                    logger.info(f"State saved after processing batch for bin {bin_name}.")
            
            logger.info(f"Round-robin download process completed. Total songs processed (attempted): {self.state['processed_songs_count']}.")
            logger.info(f"Successfully downloaded {self.state['successful_audio_downloads']} audio files and {self.state['successful_lyrics_downloads']} lyrics files.")
            logger.info(f"Failed to process {self.state['failed_songs_count']} songs. See state file for details.")
            
        except Exception as e:
            logger.error(f"Error in main download_songs (round-robin): {str(e)}", exc_info=True)
            # Save state on unexpected error too
            self.state["last_run"] = datetime.now().isoformat()
            self._save_state()
            logger.info("State saved due to unexpected error in download_songs.")
            # raise # Optionally re-raise

    async def _download_single_song_audio(self, song_name: str, artist: Optional[str] = None, popularity_bin: str = "") -> bool:
        """Download audio for a single song."""
        try:
            logger.info(f"Starting download for song: {song_name} (Artist: {artist or 'Unknown'}) in bin {popularity_bin}")
            
            # Create output directory for the popularity bin
            output_dir = os.path.join(self.downloads_dir, popularity_bin)
            os.makedirs(output_dir, exist_ok=True)
            
            # Try each API key until one works
            for api_key in self.api_keys:
                try:
                    yt_downloader = YouTubeDownloader(
                        output_dir=output_dir,
                        youtube_api_key=api_key
                    )
                    
                    success = await yt_downloader.download_song(song_name, artist or "", 0)
                    
                    if success:
                        logger.info(f"✅ Successfully downloaded: {song_name} using API key: {api_key[:5]}...")
                        return True
                    else:
                        logger.warning(f"Failed to download {song_name} with API key: {api_key[:5]}..., trying next key")
                except Exception as e:
                    logger.warning(f"Error with API key {api_key[:5]}...: {str(e)}, trying next key")
                    continue
            
            logger.error(f"❌ All API keys failed for song: {song_name}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Exception during audio download for '{song_name}': {str(e)}")
            return False
        
    async def run(self):
        """Run the complete song downloading process."""
        logger.info("Starting song download process")
        
        # Step 1: Load and process song data
        data = self.load_song_data()
        if data is None or data.empty:
            logger.error("Failed to load song data")
            return
            
        # Step 2: Download songs and fetch lyrics
        await self.download_songs()
        
        logger.info("Song download process completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Download songs as MP3s organized by popularity using a round-robin strategy.')
    parser.add_argument('--csv-path', type=str, required=True, help='Path to the CSV file with song data')
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=5, 
        help='Number of songs to download per popularity bin in each round-robin cycle (default: 5)'
    )
    parser.add_argument(
        '--state-file',
        type=str,
        help='Path to a specific state file to use for tracking downloads'
    )
    parser.add_argument(
        '--min-popularity-bin',
        type=int,
        default=0,
        help='Minimum popularity bin to consider for downloading (0-10, default: 0)'
    )
    args = parser.parse_args()
    
    # Initialize song downloader
    downloader = SongDownloader(
        csv_path=args.csv_path,
        songs_per_bin_batch=args.batch_size,
        state_file_path=args.state_file,
        min_popularity_bin=args.min_popularity_bin
    )
    
    # Run the downloader
    asyncio.run(downloader.run())

if __name__ == "__main__":
    main() 