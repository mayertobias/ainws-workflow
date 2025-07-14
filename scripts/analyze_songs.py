import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime
import hashlib
from tqdm import tqdm
import argparse
from services.audio_analyzer import AudioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SongAnalyzer:
    def __init__(self, downloads_dir: str = "downloads", output_csv: str = "data/training_data/songs/song_features.csv", 
                 recursive: bool = False, skip_merge: bool = False,
                 song_data_csv: str = "data/training_data/song_data.csv", 
                 merged_output: str = "data/merged_song_data.json",
                 include_genre: bool = False):
        """
        Initialize the song analyzer.
        
        Args:
            downloads_dir (str): Directory containing songs to analyze
            output_csv (str): Path to save the analysis results
            recursive (bool): Whether to search for songs recursively
            skip_merge (bool): Whether to skip merging with song_data_csv
            song_data_csv (str): Path to the original song data CSV
            merged_output (str): Path to save the merged data
            include_genre (bool): Whether to include genre extraction
        """
        self.downloads_dir = downloads_dir
        self.output_csv = output_csv
        self.song_data_csv = song_data_csv
        self.merged_output = merged_output
        self.recursive = recursive
        self.skip_merge = skip_merge
        self.include_genre = include_genre
        self.audio_analyzer = AudioAnalyzer()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        if not self.skip_merge:
            os.makedirs(os.path.dirname(merged_output), exist_ok=True)
        
        # Initialize or load existing analysis data
        self.analysis_data = self._load_analysis_data()
        
    def _load_analysis_data(self) -> pd.DataFrame:
        """Load existing analysis data from CSV or create new DataFrame."""
        if os.path.exists(self.output_csv):
            logger.info(f"Loading existing analysis data from {self.output_csv}")
            return pd.read_csv(self.output_csv)
        else:
            logger.info("No existing analysis data found. Creating new DataFrame.")
            return pd.DataFrame()
            
    def _save_analysis_data(self):
        """Save analysis data to CSV."""
        try:
            self.analysis_data.to_csv(self.output_csv, index=False)
            logger.info(f"Analysis data saved to {self.output_csv}")
        except Exception as e:
            logger.error(f"Error saving analysis data: {e}")
            
    def _get_song_hash(self, file_path: str) -> str:
        """Generate a hash for the song file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            return ""
            
    def _is_song_analyzed(self, song_hash: str) -> bool:
        """Check if a song has already been analyzed."""
        return not self.analysis_data.empty and song_hash in self.analysis_data['song_hash'].values
    
    def _find_mp3_files(self) -> Dict[str, List[str]]:
        """
        Find all MP3 files in the downloads directory.
        
        Returns:
            Dict mapping directory paths to lists of MP3 filenames
        """
        mp3_files_by_dir = {}
        
        if self.recursive:
            # Recursively walk through all subdirectories
            for root, _, files in os.walk(self.downloads_dir):
                mp3_files = [f for f in files if f.endswith('.mp3')]
                if mp3_files:
                    mp3_files_by_dir[root] = mp3_files
        else:
            # Only look at popularity bins (original behavior)
            popularity_bins = [d for d in os.listdir(self.downloads_dir)
                            if os.path.isdir(os.path.join(self.downloads_dir, d))
                            and d.startswith('popularity_')]
            
            for bin_name in popularity_bins:
                bin_path = os.path.join(self.downloads_dir, bin_name)
                mp3_files = [f for f in os.listdir(bin_path) if f.endswith('.mp3')]
                if mp3_files:
                    mp3_files_by_dir[bin_path] = mp3_files
                    
        return mp3_files_by_dir
                
    def analyze_songs(self):
        """Analyze all songs in downloads directory."""
        try:
            # Find all MP3 files
            mp3_files_by_dir = self._find_mp3_files()
            if not mp3_files_by_dir:
                logger.error(f"No MP3 files found in {self.downloads_dir}")
                return
            
            total_dirs = len(mp3_files_by_dir)
            total_files = sum(len(files) for files in mp3_files_by_dir.values())
            logger.info(f"Found {total_files} MP3 files in {total_dirs} directories")
            
            # Process each directory
            for dir_path, mp3_files in mp3_files_by_dir.items():
                dir_name = os.path.basename(dir_path)
                logger.info(f"Processing songs in {dir_path}")
                
                # Process each song
                for mp3_file in tqdm(mp3_files, desc=f"Analyzing songs in {dir_name}"):
                    try:
                        song_path = os.path.join(dir_path, mp3_file)
                        
                        # Generate hash for the song
                        song_hash = self._get_song_hash(song_path)
                        if not song_hash:
                            continue
                            
                        # Skip if already analyzed
                        if self._is_song_analyzed(song_hash):
                            logger.debug(f"Skipping already analyzed song: {mp3_file}")
                            continue
                            
                        # Analyze the song
                        logger.info(f"Analyzing song: {mp3_file}")
                        features = self.audio_analyzer.analyze_audio(song_path)

                        # Convert loudness_features to pipe-separated string
                        if features and 'loudness_features' in features:
                            loudness_data = features['loudness_features']
                            if isinstance(loudness_data, dict):
                                ordered_keys = ["integrated_lufs", "loudness_range_lu", "max_momentary_lufs", "max_short_term_lufs", "true_peak_dbtp"]
                                # Use .get(k, '') to default missing keys to empty string, then convert to str
                                str_values = [str(loudness_data.get(k, '')) for k in ordered_keys]
                                features['loudness_features'] = "|".join(str_values)
                                logger.info(f"Successfully converted loudness_features dict for {mp3_file} to: {features['loudness_features']}")
                            elif isinstance(loudness_data, str) and all(key_name in loudness_data for key_name in ["integrated_lufs", "max_momentary_lufs"]):
                                # Heuristic: If it's a string and contains some of our expected key names, it's likely the problematic string.
                                logger.warning(f"loudness_features for {mp3_file} was ALREADY a string and looks like feature names: '{loudness_data}'. THIS IS UNEXPECTED. Problem might be upstream.")
                                # Keep it as is, as the issue is likely before this conversion step.
                            elif isinstance(loudness_data, (list, tuple)):
                                logger.warning(f"loudness_features for {mp3_file} was a list/tuple: {loudness_data}. Joining its string elements.")
                                features['loudness_features'] = "|".join(map(str, loudness_data))
                            else:
                                logger.warning(f"loudness_features for {mp3_file} was an unexpected type: {type(loudness_data)}. Converting to string: '{str(loudness_data)}'")
                                features['loudness_features'] = str(loudness_data)
                        elif features:
                            logger.warning(f"No 'loudness_features' key in features for {mp3_file}. Keys found: {list(features.keys())}")
                            features['loudness_features'] = "||||" # Ensure column exists with empty values if key missing
                        else:
                            logger.warning(f"Features dictionary is None for {mp3_file}. Cannot process loudness_features.")
                            # Ensure 'features' is a dict so later updates don't fail, and add placeholder
                            features = {'loudness_features': "||||"}
                        
                        # Extract genre information if requested
                        if self.include_genre:
                            try:
                                logger.info(f"Extracting genre information for: {mp3_file}")
                                genre_info = self.audio_analyzer.extract_genre(song_path)
                                
                                # Add primary genre and top genres to features
                                features['primary_genre'] = genre_info.get('primary_genre', 'unknown')
                                
                                # Convert top genres list to a string format using PIPE delimiter for CSV storage
                                top_genres = genre_info.get('top_genres', [])
                                if top_genres:
                                    # Format as "genre1 (0.75) | genre2 (0.65) | ..."
                                    top_genres_str = " | ".join([f"{g[0]} ({g[1]:.2f})" for g in top_genres])
                                    features['top_genres'] = top_genres_str
                                else:
                                    features['top_genres'] = ""
                                    
                                logger.info(f"Genre identified: {features['primary_genre']}")
                                
                            except Exception as e:
                                logger.error(f"Error extracting genre for {mp3_file}: {e}")
                                features['primary_genre'] = "unknown"
                                features['top_genres'] = ""
                        
                        if features:
                            # Add metadata
                            features.update({
                                'song_name': os.path.splitext(mp3_file)[0],
                                'directory': dir_path,
                                'song_hash': song_hash,
                                'analysis_date': datetime.now().isoformat()
                            })
                            
                            # Add to analysis data
                            self.analysis_data = pd.concat([
                                self.analysis_data,
                                pd.DataFrame([features])
                            ], ignore_index=True)
                            
                            # Save after each successful analysis
                            self._save_analysis_data()
                            
                    except Exception as e:
                        logger.error(f"Error analyzing song {mp3_file}: {e}")
                        continue
                        
            logger.info("Song analysis completed")
            
        except Exception as e:
            logger.error(f"Error in analyze_songs: {e}")
            
    def merge_song_data(self):
        """
        Merge the analyzed song features with the original song data.
        Creates a JSON file with combined data for each song.
        """
        if self.skip_merge:
            logger.info("Skipping merge step as requested")
            return
            
        try:
            # Load original song data
            if not os.path.exists(self.song_data_csv):
                logger.error(f"Original song data CSV not found: {self.song_data_csv}")
                return
                
            logger.info(f"Loading original song data from {self.song_data_csv}")
            song_data = pd.read_csv(self.song_data_csv)
            
            # Load analyzed features
            if self.analysis_data.empty:
                logger.error("No analyzed features found. Run analyze_songs first.")
                return
                
            # Create merged data
            merged_data = []
            
            # Process each analyzed song
            for _, analyzed_row in self.analysis_data.iterrows():
                # Trim song name by removing trailing spaces and '-'
                song_name = analyzed_row['song_name'].rstrip(' -').strip()
                
                # Find matching song in original data
                matching_songs = song_data[song_data['song_name'] == song_name]
                
                if not matching_songs.empty:
                    # Get the first match (assuming unique song names)
                    original_row = matching_songs.iloc[0]
                    
                    # Create merged record
                    merged_record = {
                        'song_name': song_name,
                        'original_data': {f'training_{k}': v for k, v in original_row.to_dict().items()},
                        'audio_features': {f'calculated_{k}': v for k, v in analyzed_row.drop(['song_name', 'directory', 'song_hash', 'analysis_date']).to_dict().items()},
                        'metadata': {
                            'directory': analyzed_row['directory'],
                            'song_hash': analyzed_row['song_hash'],
                            'analysis_date': analyzed_row['analysis_date']
                        }
                    }
                    
                    merged_data.append(merged_record)
                else:
                    logger.warning(f"No matching song found in original data for: {song_name}")
            
            # Save merged data as JSON
            with open(self.merged_output, 'w') as f:
                json.dump(merged_data, f, indent=2)
                
            logger.info(f"Merged data saved to {self.merged_output}")
            logger.info(f"Total songs merged: {len(merged_data)}")
            
        except Exception as e:
            logger.error(f"Error merging song data: {e}")
            
def main():
    parser = argparse.ArgumentParser(description='Analyze songs and extract audio features')
    parser.add_argument('--downloads-dir', type=str, default="songs", help='Directory containing songs to analyze')
    parser.add_argument('--output-csv', type=str, default="data/training_data/songs/song_features.csv", help='Path to save analysis results')
    parser.add_argument('--recursive', action='store_true', help='Recursively search for songs in all subdirectories')
    parser.add_argument('--skip-merge', action='store_true', help='Skip merging with song data CSV')
    parser.add_argument('--song-data-csv', type=str, default="data/training_data/r4a_song_data_training.csv", help='Path to original song data CSV')
    parser.add_argument('--merged-output', type=str, default="data/training_data/songs/merged_song_data.json", help='Path to save merged data')
    parser.add_argument('--merge-only', action='store_true', help='Only merge data without analyzing songs')
    parser.add_argument('--include-genre', action='store_true', help='Include genre extraction in the analysis')
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SongAnalyzer(
        downloads_dir=args.downloads_dir,
        output_csv=args.output_csv,
        recursive=args.recursive,
        skip_merge=args.skip_merge,
        song_data_csv=args.song_data_csv,
        merged_output=args.merged_output,
        include_genre=args.include_genre
    )
    
    if not args.merge_only:
        # Run analysis
        analyzer.analyze_songs()
    
    # Merge data (will be skipped if skip_merge is True)
    analyzer.merge_song_data()

if __name__ == "__main__":
    main() 