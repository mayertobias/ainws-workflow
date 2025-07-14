#!/usr/bin/env python3
"""
Mood feature extractor for audio files
"""

import os
import essentia.standard as es
from .base import BaseFeatureExtractor
from collections import Counter

class MoodExtractor(BaseFeatureExtractor):
    """
    Extracts mood characteristics from audio using Essentia's TensorflowPredict2D models.
    Uses pre-trained models to predict mood characteristics such as happy, sad, aggressive, etc.
    """
    
    MOODS = [
        ['mood_happy', 'mood_happy-msd-musicnn-1.pb'],
        ['mood_sad', 'mood_sad-msd-musicnn-1.pb'],
        ['mood_aggressive', 'mood_aggressive-msd-musicnn-1.pb'],
        ['mood_relaxed', 'mood_relaxed-msd-musicnn-1.pb'],
        ['mood_party', 'mood_party-msd-musicnn-1.pb'],
        ['mood_electronic', 'mood_electronic-msd-musicnn-1.pb'],
        ['mood_acoustic', 'mood_acoustic-msd-musicnn-1.pb']
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use environment variable or default path
        self.models_path = os.getenv('ESSENTIA_MODELS_PATH', '/Users/manojveluchuri/saas/workflow/shared-models/essentia')
        self.embedding_model_path = os.path.join(self.models_path, 'msd-musicnn-1.pb')
    
    def extract(self, features):
        """
        Extract mood information from an audio file
        
        Args:
            features (dict): Dictionary containing at least 'audio_file'
            
        Returns:
            dict: Dictionary containing mood probabilities
        """
        audio_file = features.get('audio_file')
        if not audio_file:
            raise ValueError("No audio file provided")
        
        # Load audio
        audio = es.MonoLoader(filename=audio_file, sampleRate=16000)()
        
        # Compute MusicNN Embedding (required for all mood models)
       
        embedding_model = es.TensorflowPredictMusiCNN(
            graphFilename=self.embedding_model_path,
            output='model/dense/BiasAdd')  # This is the embedding model
        embedding = embedding_model(audio)
        # Process through mood classifiers
        mood_results = {}
        for mood in self.MOODS:
            model_path = os.path.join(self.models_path, mood[1])
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found. Skipping {mood[0]}.")
                continue
                
            # Apply model and get predictions
            model = es.TensorflowPredict2D(graphFilename=model_path,
            # input='model/Placeholder',
            output='model/Softmax')
            predictions = model(embedding)
            
            # # Store value (take average if multiple values returned)
            # mood_notsad_value = float(predictions[0].mean())
            # mood_sad_value = float(predictions[1].mean())

            mood_classes = []

            for pred in predictions:
                # Extract probabilities for positive and negative classes
                negative_probability = pred[0] 
                positive_probability = pred[1]

                # Classify based on the highest probability
                if positive_probability > negative_probability:
                    mood_class = mood[0].replace("mood_", "") # e.g. "happy", "sad", "aggressive" etc
                else:
                    mood_class = "not " + mood[0].replace("mood_", "")
                mood_classes.append(mood_class)

            # Print the results for each prediction
            # print(mood_classes)


            # Count the occurrences of each class
            count = Counter(mood_classes)

            # Get the most common mood
            most_common_mood = count.most_common(1)[0][0]

            # Output the final classification
            print(f"The final classification is: {most_common_mood}")
            mood_results[mood[0]] = most_common_mood
        
        return {
            "moods": mood_results
        } 