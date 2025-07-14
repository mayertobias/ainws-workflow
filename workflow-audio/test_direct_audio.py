import os
import sys
from pathlib import Path
import json
import asyncio
from datetime import datetime
import numpy as np
from typing import Optional

# Add the backend directory to the Python path
backend_path = str(Path(__file__).parent.parent / 'backend')
sys.path.append(backend_path)

from services.audio_analyzer import AudioAnalyzer
from services.lyrics_service import LyricsService
from services.hit_song_science import HitSongScience
from services.ai_agent import AIInsightsGenerator as StandardAIInsightsGenerator
from services.ai_agent_cmp import AIInsightsGenerator as ComprehensiveAIInsightsGenerator
from utils.report_generator import ReportGenerator
from tasks import scale_hlf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cache_path(song_name: str, ai_mode: str) -> Path:
    """Get the path for cached analysis data."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    # Changed filename to reflect comprehensive cache
    return cache_dir / f"{song_name}_{ai_mode}_analysis_cache.json"

def load_cached_data(cache_path: Path) -> Optional[dict]: # Renamed and updated return type
    """Load cached analysis data if it exists."""
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                logger.info(f"Loading comprehensive analysis data from {cache_path}")
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached analysis data: {e}")
    return None

def save_analysis_to_cache(data_to_cache: dict, cache_path: Path): # Renamed and updated parameter
    """Save comprehensive analysis data to cache."""
    try:
        with open(cache_path, 'w') as f:
            # Use a custom default handler for numpy types if any creep in, though ideally they are converted before this stage.
            json.dump(data_to_cache, f, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
        logger.info(f"Saved comprehensive analysis data to {cache_path}")
    except Exception as e:
        logger.error(f"Error saving analysis data to cache: {e}")

async def main(ai_mode: str = "comprehensive", use_cache: bool = True):
    # Initialize the services
    analyzer = AudioAnalyzer()
    lyrics_service = LyricsService()
    hss_analyzer = HitSongScience()
    standard_ai_insights = StandardAIInsightsGenerator()
    comprehensive_ai_insights = ComprehensiveAIInsightsGenerator()
    report_generator = ReportGenerator()
    
    # Path to your test audio file
    # audio_path = "/Users/manojveluchuri/Documents/rasie4art/hss-downloader/downloads/2010/001_Eminem - Not Afraid.mp3"
    # audio_path = "/Users/manojveluchuri/Documents/rasie4art/hss-downloader/downloads/2011/003_Adele - Rolling In The Deep.mp3"
    # audio_path = "/Users/manojveluchuri/Documents/rasie4art/hss-downloader/downloads/2014/007_Katy Perry - Roar.mp3"
    # audio_path = "/Users/manojveluchuri/Documents/rasie4art/hss-downloader/downloads/2014/006_Katy Perry - Dark Horse ft. Juicy J.mp3"
    # audio_path = "/Users/manojveluchuri/Downloads/ERNMILL BOUNCE.mp3"
    # audio_path = "/Users/manojveluchuri/Downloads/Crazy Praize.mp3"
    # audio_path = "/Users/manojveluchuri/Downloads/01 Deja Edit.mp3"
    # audio_path = "/Users/manojveluchuri/saas/r1/simpleui/backend/downloads/popularity_8/Boulevard of Broken Dreams - .mp3"
    # audio_path = "/Users/manojveluchuri/saas/r1/simpleui/backend/downloads/popularity_6/Mambo No 5 - .mp3"
    # audio_path = "/Users/manojveluchuri/saas/r1/simpleui/backend/downloads/popularity_9/Hotel California - Remastered - .mp3"
    audio_path = "/Users/manojveluchuri/saas/r1/simpleui/backend/downloads/popularity_9/Billie Jean - .mp3"
    # Read lyrics
    # lyrics_path = Path(__file__).parent.parent / 'backend' / 'lyrics' / 'transcribed' / 'ERNMILL BOUNCE.txt'
    # lyrics_path = "/Users/manojveluchuri/saas/r1/simpleui/backend/lyrics/popularity_9/Hotel California - Remastered -.txt"
    lyrics_path = "/Users/manojveluchuri/saas/r1/simpleui/backend/lyrics/popularity_9/Billie Jean.txt"
    with open(lyrics_path, 'r') as f:
        lyrics_text = f.read()
    
    genre = "Pop"  # Use the genre from your metadata
    
    # Define keys for cached components
    RAW_AUDIO_FEATURES_KEY = 'raw_audio_features'
    LYRICS_ANALYSIS_DATA_KEY = 'lyrics_analysis_data'
    HIT_ANALYSIS_DATA_KEY = 'hit_analysis_data'
    SCALED_HLF_DATA_KEY = 'scaled_hlf_data'
    INSIGHTS_KEY = 'insights'

    cached_data = None
    # Create metadata first as it's used in cache path and potentially if cache exists
    metadata = {
        "song_name": "Billie Jean",
        "artist_name": "Michael Jackson",
        "album_name": "Thriller",
        "genre": genre,
        "analysis_date": datetime.now().isoformat(), # This will update each run; consider caching creation time too
        "ai_mode_used": ai_mode
    }
    cache_path = get_cache_path(metadata["song_name"], ai_mode)

    if use_cache:
        cached_data = load_cached_data(cache_path)

    # Initialize variables for data components
    raw_audio_features = None
    lyrics_analysis_data = None
    hit_analysis_data = None
    scaled_hlf_data = None
    insights = None

    try:
        if cached_data and all(key in cached_data for key in [RAW_AUDIO_FEATURES_KEY, LYRICS_ANALYSIS_DATA_KEY, HIT_ANALYSIS_DATA_KEY, SCALED_HLF_DATA_KEY, INSIGHTS_KEY]):
            logger.info(f"Using all cached analysis components from {cache_path}")
            raw_audio_features = cached_data[RAW_AUDIO_FEATURES_KEY]
            lyrics_analysis_data = cached_data[LYRICS_ANALYSIS_DATA_KEY]
            hit_analysis_data = cached_data[HIT_ANALYSIS_DATA_KEY]
            scaled_hlf_data = cached_data[SCALED_HLF_DATA_KEY]
            insights = cached_data[INSIGHTS_KEY]
            # Potentially update metadata like analysis_date if needed from cache, or keep current
            if 'metadata' in cached_data and 'analysis_date' in cached_data['metadata']:
                 metadata['analysis_date'] = cached_data['metadata']['analysis_date'] # Use cached analysis date
        else:
            logger.info("One or more components not found in cache or cache disabled. Processing from scratch.")
            # Analyze the audio file
            logger.info(f"Analyzing audio file: {audio_path}")
            raw_audio_features = analyzer.analyze_audio(audio_path)
            
            # Perform full lyrics analysis
            logger.info("Analyzing lyrics...")
            lyrics_analysis_data = lyrics_service.analyze(lyrics_text)

            # Prepare HSS input features
            logger.info("Preparing features for Hit Song Science evaluation...")
            hss_input_audio_features = {
                'tempo': raw_audio_features.get('tempo', raw_audio_features.get('tempo_bpm', 0.0)),
                'energy': raw_audio_features.get('energy', 0.0),
                'danceability': raw_audio_features.get('danceability', 0.0),
                'valence': raw_audio_features.get('valence', 0.0),
                'acousticness': raw_audio_features.get('acousticness', 0.0),
                'loudness': raw_audio_features.get('loudness_integrated', raw_audio_features.get('loudness', -60.0))
            }
            hss_input_lyrics_features = {
                # 'sentiment': lyrics_analysis_data.get('sentiment', {}).get('polarity', 0.0), # Sentiment removed from model
                'complexity': lyrics_analysis_data.get('complexity', {}).get('lexical_diversity', 0.5)
            }

            # Evaluate with HitSongScience
            logger.info("Evaluating song with Hit Song Science...")
            hit_analysis_data = hss_analyzer.evaluate_song(hss_input_audio_features, hss_input_lyrics_features)

            # Scale HLF features
            logger.info("Scaling HLF features...")
            scaled_hlf_data = scale_hlf(raw_audio_features)
            
            # Generate AI insights
            logger.info(f"Generating {ai_mode} AI insights...")
            selected_ai_insights_generator = standard_ai_insights if ai_mode == "standard" else comprehensive_ai_insights
            insights = await selected_ai_insights_generator.generate_insights(
                audio_features=raw_audio_features,
                lyrics_text=lyrics_analysis_data, # Changed from lyrics_analysis to lyrics_text
                genre=genre,
                hit_analysis=hit_analysis_data,
                scaled_hlf=scaled_hlf_data
            )
            
            # Save all processed data to cache
            data_to_cache = {
                RAW_AUDIO_FEATURES_KEY: raw_audio_features,
                LYRICS_ANALYSIS_DATA_KEY: lyrics_analysis_data,
                HIT_ANALYSIS_DATA_KEY: hit_analysis_data,
                SCALED_HLF_DATA_KEY: scaled_hlf_data,
                INSIGHTS_KEY: insights,
                'metadata': metadata # Save metadata too, especially if using cached analysis_date
            }
            save_analysis_to_cache(data_to_cache, cache_path)
        
        # Combine all analysis results for the report
        final_analysis_results = {
            "raw_audio_features": raw_audio_features,
            "scaled_hlf": scaled_hlf_data,
            "lyrics_analysis": lyrics_analysis_data,
            "hit_analysis": hit_analysis_data,
            "ai_insights": insights,
            "metadata": metadata,
            "ai_mode_used": ai_mode
        }
        
        # Create output directory for reports
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        # Generate reports
        logger.info("Generating reports...")
        reports = report_generator.generate_report(
            analysis=final_analysis_results,
            metadata=metadata, # Pass the potentially updated metadata
            output_dir=str(output_dir)
        )
        
        # Print the results
        logger.info("Analysis completed successfully")
        print("\nRaw Audio Features:")
        print("----------------------")
        print(json.dumps(raw_audio_features, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))

        print("\nHit Analysis Data:")
        print("------------------")
        print(json.dumps(hit_analysis_data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
            
        print(f"\n{ai_mode.title()} AI Insights:")
        print("------------")
        print(json.dumps(insights, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
            
        print("\nGenerated Reports:")
        print("-----------------")
        for format, path in reports.items():
            print(f"{format.upper()} report saved to: {path}")
            
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        raise

if __name__ == "__main__":
    # You can change the AI mode here: "standard" or "comprehensive"
    # Set use_cache=False to force regeneration of insights
    asyncio.run(main(ai_mode="comprehensive", use_cache=False)) 


#     Audio Analysis Results for EMinem - Not Afraid   :
# ----------------------
# tempo: 86.03233337402344
# danceability: 1.4523555040359497
# energy: 0.9261923432350159
# valence: 0.0
# acousticness: 0.0
# loudness_integrated: -15.568532943725586
# key: N/A
# mode: N/A
# dynamic_complexity: 3.4566657543182373
# average_loudness: 0.9261923432350159
# instrumentalness: 0.0
# speechiness: 0.0

# tempo: 86.03233337402344
# onset_rate: 4.610600471496582
# beats_loudness_mean: 0.030324947088956833
# spectral_energy_mean: 0.018953688442707062
# loudness_ebu128_integrated: -15.568532943725586
# dynamic_complexity: 3.4566657543182373
# spectral_spread_mean: 6079623.5
# spectral_centroid_mean: 1909.9443359375
# spectral_entropy_mean: 7.75318717956543
# spectral_complexity_mean: 15.984116554260254
# silence_rate_60dB_mean: 0.143216073513031
# spectral_flux_mean: 0.08828413486480713
# spectral_kurtosis_mean: 4.526937484741211
# key: Eb
# mode: major
# danceability: 1.4523555040359497
# energy: 0.9261923432350159
# valence: 0.0
# acousticness: 0.0
# instrumentalness: 0.0
# speechiness: 0.0


# Audio Analysis Results:
# ----------------------
# danceability: 0.37010288685560233
# energy: 0.3104716583093008
# valence: 0.7647573700000001
# acousticness: 0.7037061632869241
# instrumentalness: 0.15977423166581872
# liveness: 0.01376039771231699
# speechiness: 0.16322739124298097
# grooviness: 0.37010288685560233
# tempo: 86.03233337402344
# key: Eb
# mode: major

# Audio Analysis Results:
# ----------------------
# danceability: 1.4523555040359497
# energy: 0.9261923432350159
# valence: 0.76479403
# acousticness: 0.7037061460404936
# instrumentalness: 0.15977423166581872
# liveness: 0.013761631398564933
# speechiness: 0.16322739124298097
# grooviness: 0.37010288685560233
# tempo: 86.03233337402344
# key: Eb
# mode: major

# Roar - Audio Analysis Results:
# ----------------------
# danceability: 0.9693353772163391
# energy: 0.8747801184654236
# valence: 0.68928313
# acousticness: 0.7327082417597575
# instrumentalness: 0.19847168438914725
# liveness: 0.010422295974870445
# speechiness: 0.1619398921728134
# grooviness: 0.33759110589822133
# tempo: 89.92548370361328
# key: Eb
# mode: major

# song_name	song_popularity	song_duration_ms	acousticness	danceability	energy	instrumentalness	key	liveness	loudness	audio_mode	speechiness	tempo	time_signature	audio_valence
# Rolling in the Deep  75      228093
# 0.138   0.73    0.77    0.0     8       0.0473  -5.114  1       0.0298  104.948 4
# 0.507
# Roar	77	223546	0.00487	0.5540000000000000	0.772	6.6E-06	7	0.354	-4.821000000000000	0	0.0418	179.984	4	0.455
# Dark Horse	79	215672	0.00314	0.645	0.585	0.0	6	0.165	-6.122000000000000	1	0.0513	131.931	4	0.353

# Dark Horse from extractors - Audio Analysis Results:
# ----------------------
# Audio Analysis Results:
# ----------------------
# acousticness: 0.7799160025865829
# instrumentalness: 0.12345774758533896
# liveness: 0.013838416167004106
# speechiness: 0.013496053977976374
# brightness: 0.10032808285667782
# complexity: 0.2694185595959425
# warmth: 0.6194739890628391
# valence: 0.6406271800000001
# harmonic_strength: 0.36997638146082557
# key: Eb
# mode: 0
# tempo: 0.5461691284179687
# danceability: 0.48761810034513475
# energy: 0.6089892846345901
# loudness: -15.658187866210938
# duration_ms: 224978
# time_signature: 4

# Audio Analysis Results:
# ----------------------
# danceability: 0.9125674962997437
# energy: 0.8904218077659607
# valence: 0.66101655
# acousticness: 0.7799472617088864
# instrumentalness: 0.18384277404082977
# liveness: 0.01384135599030733
# speechiness: 0.14099013805389404
# grooviness: 0.6547561266024907
# tempo: 131.9253692626953
# key: Eb
# mode: minor



# Rolling in deep - Audio Analysis Results:
# ----------------------
# danceability: 1.3007864952087402
# energy: 0.8766043186187744
# valence: 0.5642115299999999
# acousticness: 0.8194494604115841
# instrumentalness: 0.18937465979565615
# liveness: 0.012292880687203168
# speechiness: 0.12566858008503912
# grooviness: 0.46357041686773304
# tempo: 104.99420928955078
# key: C
# mode: minor