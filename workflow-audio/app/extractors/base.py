class BaseFeatureExtractor:
    """Base class for all feature extractors."""
    
    @staticmethod
    def normalize(value, min_val, max_val):
        """Safely normalize a value to [0, 1] range with clamping."""
        if min_val == max_val:
            return 0.0
        norm = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, norm))
    
    def extract(self, data):
        """
        Extract feature from the provided data.
        
        Args:
            data: Dictionary containing Essentia extracted features
            
        Returns:
            float: Normalized feature value between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement extract method") 