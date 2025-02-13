import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from typing import List, Tuple

from src.utils import check_path_exists, get_logger, read_csv_file

class InputFilesNotFoundError(Exception):
    pass

logger = get_logger()

class PhraseSimilarityCalculator:
    def __init__(self, word2vec_path: str, phrases_path: str, distance_metric: str = 'cosine'):
        """
        Initialize the phrase similarity calculator.
        
        Args:
            word2vec_path: Path to the Word2Vec model file
            distance_metric: 'cosine' or 'euclidean'
        """

        assert self.validate_input_files(word2vec_path, phrases_path)

        self.w2v_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
        self.distance_metric = distance_metric
        self.phrase_embeddings = None
        self.phrases = read_csv_file(phrases_path)

    def validate_input_files(self, word2vec_path, phrases_path) -> bool:
        error_detected = False
        if not check_path_exists(word2vec_path):
            logger.error(f"Word2Vec model file not found at {word2vec_path}")
            error_detected = True
        if not check_path_exists(phrases_path):
            logger.error(f"Phrases CSV file file not found at {phrases_path}")
            error_detected = True
        if error_detected:
            raise InputFilesNotFoundError("Input file(s) not found.")
        else:
            logger.info("Input files found.")
            return True

        
    def get_phrase_embedding(self, phrase: str) -> np.ndarray:
        """
        Convert a phrase to its embedding by summing word vectors and normalizing.
        
        Args:
            phrase: Input phrase
            
        Returns:
            Normalized phrase embedding vector
        """
        # Tokenize and lowercase
        words = phrase.lower().split()
        
        # Get embeddings for words that exist in the model
        word_embeddings = []
        for word in words:
            try:
                word_embeddings.append(self.w2v_model[word])
            except KeyError:
                continue
                
        if not word_embeddings:
            return np.zeros(self.w2v_model.vector_size)
            
        # Sum the word embeddings
        phrase_embedding = np.sum(word_embeddings, axis=0)
        
        # Normalize the embedding
        norm = np.linalg.norm(phrase_embedding)
        if norm > 0:
            phrase_embedding = phrase_embedding / norm
            
        return phrase_embedding


    def batch_compute_embeddings(self) -> None:
        """
        Compute embeddings for all phrases in the CSV
        
        Args:
            phrases_path: Path to phrases.csv
        """        
            
        # Compute embeddings
        self.phrase_embeddings = np.vstack([
            self.get_phrase_embedding(phrase) for phrase in self.phrases
        ])
        
    
    def batch_compute_distances(self, threshold: float = None, top_k: int = None) -> List[dict]:
        """
        Compute pairwise distances between all phrases and return detailed results.
        
        Args:
            threshold: Optional float to filter distances below this threshold
            top_k: Optional int to return only top K closest matches per phrase
            
        Returns:
            List of dictionaries containing phrase pairs and their distances:
            [
                {
                    'phrase': 'original phrase',
                    'matches': [
                        {'match_phrase': 'matching phrase', 'distance': 0.75},
                        ...
                    ]
                },
                ...
            ]
        """
        if self.phrase_embeddings is None or self.phrases is None:
            raise ValueError("Must call batch_compute_embeddings first!")
            
        # Compute distance matrix
        if self.distance_metric == 'cosine':
            distances = cosine_distances(self.phrase_embeddings)
        else:
            distances = euclidean_distances(self.phrase_embeddings)
            
        results = []
        
        # Process each phrase
        for i, phrase in enumerate(self.phrases):
            # Get distances for current phrase (excluding self-comparison)
            phrase_distances = [(j, dist) for j, dist in enumerate(distances[i]) 
                              if i != j and (threshold is None or dist <= threshold)]
            
            # Sort by distance
            phrase_distances.sort(key=lambda x: x[1])
            
            # Apply top_k filter if specified
            if top_k is not None:
                phrase_distances = phrase_distances[:top_k]
            
            # Format matches for current phrase
            matches = [
                {'match_phrase': self.phrases[idx], 'distance': dist}
                for idx, dist in phrase_distances
            ]
            
            results.append({
                'phrase': phrase,
                'matches': matches
            })
            
        return results
        
    
    def find_closest_match(self, query: str, n: int = 1) -> List[Tuple[str, float]]:
        """
        Find the closest matching phrase(s) for a query string.
        
        Args:
            query: Input phrase to match
            n: Number of closest matches to return
            
        Returns:
            List of (phrase, distance) tuples for the n closest matches
        """
        if self.phrase_embeddings is None:
            raise ValueError("Must call batch_compute_embeddings first!")
            
        # Get query embedding
        query_embedding = self.get_phrase_embedding(query)
        
        # Compute distances
        if self.distance_metric == 'cosine':
            distances = cosine_distances(query_embedding.reshape(1, -1), self.phrase_embeddings)[0]
        else:
            distances = euclidean_distances(query_embedding.reshape(1, -1), self.phrase_embeddings)[0]
            
        # Get top N closest matches
        closest_indices = np.argsort(distances)[:n]
        
        return [(self.phrases[idx], distances[idx]) for idx in closest_indices]
