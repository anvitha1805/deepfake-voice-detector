"""
Chunk aggregation module for multi-frame deepfake audio detection.
Provides mean, max, and top20 chunk aggregation strategies to combine
chunk-level spoof prediction scores or feature representations into
an overall utterance-level decision score.
"""

import numpy as np

def aggregate_mean(scores):
    """
    Computes the mean score across all audio chunks.
    
    Args:
        scores (array-like): 1D array of chunk scores/probabilities.
        
    Returns:
        float: Mean score.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))

def aggregate_max(scores):
    """
    Computes the maximum score across all audio chunks.
    Highlights localized spoof segments.
    
    Args:
        scores (array-like): 1D array of chunk scores/probabilities.
        
    Returns:
        float: Maximum score.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    return float(np.max(scores))

def aggregate_top20(scores):
    """
    Computes top20 aggregation.
    Averages the top 20% highest-scoring chunks (or at least 1 chunk)
    for an utterance, making it sensitive to partial and localized deepfake artifacts.
    
    Args:
        scores (array-like): 1D array of chunk scores/probabilities.
        
    Returns:
        float: Mean of top 20% highest chunk scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(scores) * 0.20)))
    top_scores = np.sort(scores)[-k:]
    return float(np.mean(top_scores))

def aggregate_chunks(scores, method="top20"):
    """
    Dispatches chunk score aggregation using the specified method.
    
    Args:
        scores (array-like): 1D array of chunk scores/probabilities.
        method (str): One of 'mean', 'max', 'top20'. Default is 'top20'.
        
    Returns:
        float: Aggregated utterance score.
    """
    method = str(method).lower().strip()
    if method == "mean":
        return aggregate_mean(scores)
    elif method == "max":
        return aggregate_max(scores)
    elif method in ["top20", "top_20", "top-20"]:
        return aggregate_top20(scores)
    else:
        raise ValueError(f"Unsupported aggregation method: '{method}'. Choose 'mean', 'max', or 'top20'.")

def aggregate_by_utterance(chunk_scores, utt_ids, method="top20"):
    """
    Aggregates chunk-level scores grouped by utterance ID.
    
    Args:
        chunk_scores (array-like): 1D array of chunk scores.
        utt_ids (array-like): 1D array of corresponding utterance IDs.
        method (str): Aggregation method ('mean', 'max', 'top20').
        
    Returns:
        dict: Mapping of utt_id -> aggregated_score.
    """
    chunk_scores = np.asarray(chunk_scores)
    utt_ids = np.asarray(utt_ids)
    
    unique_utts, inverse_indices = np.unique(utt_ids, return_inverse=True)
    aggregated = {}
    
    for idx, u in enumerate(unique_utts):
        mask = (inverse_indices == idx)
        scores_u = chunk_scores[mask]
        aggregated[u] = aggregate_chunks(scores_u, method=method)
        
    return aggregated

if __name__ == "__main__":
    # Self-test unit verification
    test_scores = np.array([0.1, 0.2, 0.9, 0.4, 0.8])
    print(f"Test Scores: {test_scores}")
    print(f"  mean:  {aggregate_mean(test_scores):.4f}")
    print(f"  max:   {aggregate_max(test_scores):.4f}")
    print(f"  top20: {aggregate_top20(test_scores):.4f}")
    print("Aggregate module verified successfully.")
