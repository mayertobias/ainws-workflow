import json
import os
import numpy as np
import essentia
import essentia.standard as es
from .base import BaseFeatureExtractor

class GenreExtractor(BaseFeatureExtractor):
    """
    Extracts genre information from audio data using Essentia's deep learning models.
    Uses the Discogs-trained EfficientNet model to extract embeddings and
    a classification head trained on genre_discogs400 dataset to predict genres.
    """
    
    def __init__(self):
        """Initialize the genre extractor with required models."""
        super().__init__()
        
        # Define paths to the models - use environment variable or default
        base_path = os.getenv('ESSENTIA_MODELS_PATH', '/Users/manojveluchuri/saas/workflow/shared-models/essentia')
        
        self.embedding_model_path = os.path.join(
            base_path, 
            "discogs-effnet-bs64-1.pb"
        )
        
        self.genre_model_path = os.path.join(
            base_path, 
            "genre_discogs400-discogs-effnet-1.pb"
        )
        
        self.genre_labels_path = os.path.join(
            base_path, 
            "genre_discogs400-discogs-effnet-1.json"
        )
        
        # Load genre labels
        try:
            with open(self.genre_labels_path, 'r') as f:
                metadata = json.load(f)
                self.genre_labels = metadata['classes']
        except Exception as e:
            # If labels can't be loaded, create a placeholder
            print(f"Warning: Could not load genre labels: {str(e)}")
            self.genre_labels = [f"genre_{i}" for i in range(400)]
    
    def predict_from_audio(self, audio_path):
        """
        Predict genres directly from an audio file.
        This is used when we have access to the audio file but not pre-computed features.
        """
        try:
            # Load the audio file
            audio = es.MonoLoader(filename=audio_path, sampleRate=16000)()
            
            # Load the embedding model
            embedding_model = es.TensorflowPredictEffnetDiscogs(
                graphFilename=self.embedding_model_path,
                output='PartitionedCall:1'
            )
            
            # Extract embeddings
            embeddings = embedding_model(audio)
            
            # Load the genre classification model
            genre_model = es.TensorflowPredict2D(
                graphFilename=self.genre_model_path,
                input='serving_default_model_Placeholder',
                output='PartitionedCall:0'
            )
            
            # Predict genre
            predictions = genre_model(embeddings)
            
            # Compute average predictions
            avg_predictions = np.mean(predictions, axis=0)
            
            # Get all genres with probabilities
            genre_probs = {self.genre_labels[i]: float(avg_predictions[i]) for i in range(len(self.genre_labels))}
            
            # Get top 5 genres
            top_indices = avg_predictions.argsort()[-5:][::-1]
            top_genres = [(self.genre_labels[idx], float(avg_predictions[idx])) for idx in top_indices]
            
            result = {
                "top_genres": top_genres,
                "genre_probabilities": genre_probs,
                "primary_genre": self.genre_labels[top_indices[0]]
            }
            
            return result
        
        except Exception as e:
            print(f"Error in genre prediction: {str(e)}")
            return {
                "top_genres": [("unknown", 1.0)],
                "genre_probabilities": {"unknown": 1.0},
                "primary_genre": "unknown"
            }
    
    def extract(self, data):
        """
        Extract genre information from Essentia JSON data.
        
        If the file has already been analyzed and includes genre information, 
        we'll use that. Otherwise, we need the original audio file path 
        from the metadata to perform the analysis.
        """
        # If data is a string, assume it's a direct file path
        if isinstance(data, str) and os.path.exists(data):
            print(f"Direct file path provided: {data}")
            return self.predict_from_audio(data)
            
        # If data is a dictionary with 'audio_file' key, use that path
        if isinstance(data, dict) and 'audio_file' in data and os.path.exists(data['audio_file']):
            print(f"Audio file path found in data dictionary: {data['audio_file']}")
            return self.predict_from_audio(data['audio_file'])
        
        # Check if genre information is already in the data
        if isinstance(data, dict) and 'genres' in data:
            return data['genres']
        
        # Get the original audio file path from metadata
        try:
            audio_file = None
            
            # Try to find the original audio file path
            if isinstance(data, dict):
                if '_metadata' in data and 'source_file' in data['_metadata']:
                    # If it's an HLF file with metadata pointing to original JSON
                    source_file = data['_metadata']['source_file']
                    # Read the original JSON to get audio file path
                    with open(source_file, 'r') as f:
                        source_data = json.load(f)
                        if 'metadata' in source_data and 'tags' in source_data['metadata']:
                            file_name = source_data['metadata']['tags']['file_name']
                            # Guess the path to the original audio file
                            audio_file = os.path.join("/Users/manojveluchuri/Documents/rasie4art/hss-essentia/input", file_name)
                
                # Alternatively, check if the original data has file name
                elif 'metadata' in data and 'tags' in data['metadata']:
                    file_name = data['metadata']['tags']['file_name']
                    # Try to find the audio file by looking in input folder
                    audio_file = os.path.join("/Users/manojveluchuri/Documents/rasie4art/hss-essentia/input", file_name)
            
            if audio_file and os.path.exists(audio_file):
                # Perform genre analysis on the audio file
                return self.predict_from_audio(audio_file)
            else:
                print(f"Warning: Could not find audio file for genre extraction")
                return {
                    "top_genres": [("unknown", 1.0)],
                    "genre_probabilities": {"unknown": 1.0},
                    "primary_genre": "unknown"
                }
                
        except Exception as e:
            print(f"Error extracting genre: {str(e)}")
            return {
                "top_genres": [("unknown", 1.0)],
                "genre_probabilities": {"unknown": 1.0},
                "primary_genre": "unknown"
            } 