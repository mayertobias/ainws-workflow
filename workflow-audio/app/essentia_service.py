from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import essentia
import essentia.standard as es
import numpy as np
import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Essentia Audio Analysis Service", version="1.0.0")

class AudioAnalyzer:
    """Audio analysis using Essentia"""
    
    def __init__(self):
        # Initialize Essentia algorithms
        self.loader = es.MonoLoader()
        self.windowing = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.spectral_peaks = es.SpectralPeaks()
        self.mfcc = es.MFCC()
        self.tempo_estimator = es.RhythmExtractor2013()
        
        # High-level descriptors
        self.rhythm_extractor = es.RhythmExtractor2013()
        self.key_extractor = es.KeyExtractor()
        self.loudness = es.Loudness()
        
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio file and extract features"""
        try:
            # Load audio
            audio = self.loader(audio_path)
            
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            # Basic audio properties
            sample_rate = 44100  # Essentia default
            duration = len(audio) / sample_rate
            
            # Extract rhythm features (tempo, beats)
            bpm, beats, beats_confidence, _, beats_intervals = self.rhythm_extractor(audio)
            
            # Extract key and mode
            key, scale, key_strength = self.key_extractor(audio)
            
            # Extract loudness
            loudness_value = self.loudness(audio)
            
            # Calculate spectral features
            spectral_features = self._extract_spectral_features(audio)
            
            # High-level features (simplified versions of what would be danceability, energy, etc.)
            hlf_features = self._extract_high_level_features(audio, bpm, spectral_features)
            
            results = {
                # Basic properties
                'duration': float(duration),
                'sample_rate': sample_rate,
                
                # Rhythm features
                'tempo': float(bpm),
                'beats_confidence': float(beats_confidence),
                
                # Tonal features
                'key': key,
                'mode': scale,
                'key_strength': float(key_strength),
                
                # Loudness
                'loudness': float(loudness_value),
                
                # High-level features (approximations)
                'energy': hlf_features['energy'],
                'danceability': hlf_features['danceability'],
                'valence': hlf_features['valence'],
                'acousticness': hlf_features['acousticness'],
                
                # Spectral features
                'spectral_centroid': spectral_features['centroid'],
                'spectral_rolloff': spectral_features['rolloff'],
                'zero_crossing_rate': spectral_features['zcr'],
                
                # Metadata
                'analysis_sample_rate': sample_rate,
                'essentia_version': essentia.__version__
            }
            
            logger.info(f"Successfully analyzed audio: {audio_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing audio {audio_path}: {str(e)}")
            raise
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features from audio"""
        # Frame-based analysis
        frame_size = 2048
        hop_size = 1024
        
        centroids = []
        rolloffs = []
        zcrs = []
        
        for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
            # Apply windowing
            windowed_frame = self.windowing(frame)
            
            # Compute spectrum
            spectrum = self.spectrum(windowed_frame)
            
            # Spectral centroid
            centroid = es.SpectralCentroid()(spectrum)
            centroids.append(centroid)
            
            # Spectral rolloff
            rolloff = es.SpectralRolloff()(spectrum)
            rolloffs.append(rolloff)
            
            # Zero crossing rate
            zcr = es.ZeroCrossingRate()(frame)
            zcrs.append(zcr)
        
        return {
            'centroid': float(np.mean(centroids)),
            'rolloff': float(np.mean(rolloffs)),
            'zcr': float(np.mean(zcrs))
        }
    
    def _extract_high_level_features(self, audio: np.ndarray, bpm: float, spectral_features: Dict) -> Dict[str, float]:
        """Extract high-level features (approximations)"""
        
        # Energy: based on RMS energy
        rms_energy = np.sqrt(np.mean(audio**2))
        energy = min(1.0, rms_energy * 10)  # Normalize to 0-1
        
        # Danceability: based on tempo and rhythm regularity
        # Higher for tempos in dance range (90-140 BPM)
        tempo_factor = 1.0 - abs(bpm - 120) / 120
        danceability = max(0.0, min(1.0, tempo_factor * energy))
        
        # Valence: based on spectral features and key
        # Higher spectral centroid often correlates with brighter, happier sound
        valence = min(1.0, spectral_features['centroid'] / 4000)
        
        # Acousticness: inverse of spectral complexity
        # Lower spectral rolloff suggests more acoustic content
        acousticness = max(0.0, min(1.0, 1.0 - spectral_features['rolloff'] / 8000))
        
        return {
            'energy': float(energy),
            'danceability': float(danceability),
            'valence': float(valence),
            'acousticness': float(acousticness)
        }

# Initialize analyzer
analyzer = AudioAnalyzer()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "essentia-audio-analysis"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio file"""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Analyze audio
            results = analyzer.analyze_audio(temp_file.name)
            
            return JSONResponse(content={
                "status": "success",
                "filename": file.filename,
                "analysis": results
            })
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/analyze_file")
async def analyze_file_path(file_path: str):
    """Analyze audio file by file path (for internal use)"""
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        results = analyzer.analyze_audio(file_path)
        
        return JSONResponse(content={
            "status": "success",
            "file_path": file_path,
            "analysis": results
        })
        
    except Exception as e:
        logger.error(f"Analysis failed for {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 