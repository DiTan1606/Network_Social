"""
Link Prediction Module for Co-author Network Analysis.

This module provides functions for calculating link predictions using
the Stochastic Block Model (SBM) from pre-trained model data.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import networkx as nx


@dataclass
class PredictionResult:
    """Represents a single link prediction result."""
    author_id: int
    author_name: str
    raw_score: float  # SBM probability
    normalized_score: float  # 0-100%
    
    @property
    def score_pct(self) -> str:
        return f"{self.normalized_score:.1f}%"


def normalize_score(raw_score: float, threshold: float, max_score: float) -> float:
    """
    Normalize a raw SBM score to 0-100% range.
    
    Uses min-max normalization: (score - threshold) / (max - threshold) * 100
    The threshold acts as the minimum value, and max_score as the maximum.
    
    Args:
        raw_score: The raw SBM probability score to normalize
        threshold: The minimum threshold value (from SBM model metadata)
        max_score: The maximum observed score for normalization
    
    Returns:
        float: Normalized score in the range [0, 100]
    
    Requirements: 3.2
    """
    # Handle edge case where max_score equals or is below threshold
    if max_score <= threshold:
        return 0.0
    
    # Apply min-max normalization
    normalized = ((raw_score - threshold) / (max_score - threshold)) * 100
    
    # Clamp to [0, 100] to ensure bounds are respected
    return max(0.0, min(100.0, normalized))


def calculate_sbm_score(node_a: int, node_b: int, sbm_data: dict) -> Optional[float]:
    """
    Calculate SBM link probability between two nodes.
    
    Uses: P(link) = probs[block_a][block_b]
    where block_a = node_block[node_a], block_b = node_block[node_b]
    
    Args:
        node_a: First node ID
        node_b: Second node ID
        sbm_data: Dictionary containing 'node_block' mapping and 'probs' matrix
    
    Returns:
        float: Probability score from the SBM model, or None if either node
               doesn't have a block assignment.
    
    Requirements: 4.2
    """
    node_block = sbm_data.get('node_block', {})
    probs = sbm_data.get('probs')
    
    if probs is None:
        return None
    
    # Get block assignments for both nodes
    block_a = node_block.get(node_a)
    block_b = node_block.get(node_b)
    
    # Handle missing block assignments gracefully
    if block_a is None or block_b is None:
        return None
    
    # Return probability from the block-to-block matrix
    try:
        return float(probs[block_a][block_b])
    except (IndexError, KeyError):
        return None


def get_link_predictions(
    author_id: int,
    prediction_data: dict,
    existing_graph: nx.Graph,
    top_n: int = 5
) -> List[PredictionResult]:
    """
    Calculate top N predicted collaborators for an author.
    
    Args:
        author_id: Node ID of the focused author
        prediction_data: Loaded PKL data containing 'data_store', 'id_map', and 'top_3_meta'
        existing_graph: Current graph (for filtering existing connections)
        top_n: Number of predictions to return (default: 5)
    
    Returns:
        List of PredictionResult objects sorted by score descending, limited to top_n.
        Returns empty list if author not found or no valid predictions.
    
    Requirements: 1.1, 2.2
    """
    if prediction_data is None:
        return []
    
    data_store = prediction_data.get('data_store', {})
    sbm_data = data_store.get('sbm', {})
    id_map = prediction_data.get('id_map', {})
    
    # Get SBM threshold for normalization
    top_3_meta = prediction_data.get('top_3_meta', [])
    sbm_threshold = 0.0
    for meta in top_3_meta:
        if meta.get('model_name') == 'SBM':
            sbm_threshold = meta.get('threshold', 0.0)
            break
    
    # Get global max score from SBM probability matrix for proper normalization
    # This prevents all scores being 100% when candidates have same block
    probs = sbm_data.get('probs')
    global_max_score = 1.0  # Default fallback
    if probs is not None:
        try:
            import numpy as np
            # Get max probability excluding 1.0 values (which are self-loops in small blocks)
            probs_array = np.array(probs)
            non_one_probs = probs_array[probs_array < 1.0]
            if len(non_one_probs) > 0:
                global_max_score = float(np.max(non_one_probs))
        except Exception:
            global_max_score = 1.0
    
    # Check if author exists in the prediction data
    node_block = sbm_data.get('node_block', {})
    if author_id not in node_block:
        return []
    
    # Get existing neighbors to exclude
    # Note: Graph may use string IDs while PKL uses int IDs
    existing_neighbors = set()
    if existing_graph is not None:
        # Try both int and string versions of author_id
        author_id_str = str(author_id)
        if author_id in existing_graph:
            existing_neighbors = set(int(n) for n in existing_graph.neighbors(author_id))
        elif author_id_str in existing_graph:
            existing_neighbors = set(int(n) for n in existing_graph.neighbors(author_id_str))
    
    # Calculate scores for all candidate nodes (not already connected)
    candidates: List[tuple] = []
    
    for candidate_id in node_block.keys():
        # Skip self
        if candidate_id == author_id:
            continue
        
        # Skip existing connections (Requirement 1.1)
        if candidate_id in existing_neighbors:
            continue
        
        # Calculate SBM score
        score = calculate_sbm_score(author_id, candidate_id, sbm_data)
        if score is not None and score > 0:
            candidates.append((candidate_id, score))
    
    # Sort by score descending (Requirement 2.2)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N (Requirement 2.2)
    top_candidates = candidates[:top_n]
    
    # Build results with normalized scores using global max
    results: List[PredictionResult] = []
    for candidate_id, raw_score in top_candidates:
        # Normalize score to 0-100% range using global max from SBM matrix
        normalized = normalize_score(raw_score, sbm_threshold, global_max_score)
        
        # Get author name from id_map
        author_name = id_map.get(candidate_id, str(candidate_id))
        
        results.append(PredictionResult(
            author_id=candidate_id,
            author_name=author_name,
            raw_score=raw_score,
            normalized_score=normalized
        ))
    
    return results
