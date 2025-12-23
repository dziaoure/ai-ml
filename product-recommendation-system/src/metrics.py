from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def precision_at_k(recommended:Sequence[int], relevant:Sequence[int], k:int) -> float:
    '''
    Precision@K = (# relevant recommended in top K) / K
    '''

    if k <= 0:
        raise ValueError('k must be positive integer')
    
    if not recommended:
        return 0.0
    
    top_k = recommended[:k]
    rel_set = set(relevant)
    hits = sum(1 for item in top_k if item in rel_set)

    return hits / float(k)


def recall_at_k(recommended:Sequence[int], relevant:Sequence[int], k:int) -> float:
    '''
    Recall@K = (# relevant recommended in top K) / (# relevant)
    '''

    if k <= 0:
        raise ValueError('k must be positive integer')
    
    if not relevant:
        return 0.0
    
    if not recommended:
        return 0.0
    
    top_k = recommended[:k]
    rel_set = set(relevant)
    hits = sum(1 for item in top_k if item in rel_set)

    return hits / float(len(rel_set))


def average_precision_at_k(
    recommended:Sequence[int],
    relevant:Sequence[int],
    k:int
) -> float:
    '''
    AP@K (Average Precision at K)
     Average of precisions computed at the ranks where a relevant item is found, up to rank K.

     If no relevant items are found in the top K, returns 0.0.
    '''

    if k <= 0:
        raise ValueError('k must be positive integer')
    
    if not relevant or not recommended:
        return 0.0
    
    top_k = recommended[:k]
    rel_set = set(relevant)

    precisions = []
    hits = 0

    for i, item in enumerate(top_k, start=1):
        if item in rel_set:
            hits += 1
            precisions.append(hits / float(i))

    if not precisions:
        return 0.0
    
    #Normalize by min(#relevant, k) to keep AP bounded in [0, 1]
    denom = min(len(rel_set), k)
    
    return float(np.sum(precisions) / denom)


def mean_metric(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else 0.0


def evaluate_topK(
    user_to_recommended: Dict[int, List[int]],
    user_to_relevant: Dict[int, List[int]],
    k: int
) -> Dict[str, float]:
    '''
    Compute mean Precision@K, Recall@K, and MAP@K across users present in both dictionaries.

    Parameters:
    - user_to_recommended: Dict mapping userId to list of recommended itemIds
    - user_to_relevant: Dict mapping userId to list of relevant itemIds
    - k: Cutoff rank K

    Returns:
    - metrics: Dict with keys 'Precision@K', 'Recall@K', 'MAP@K'
    '''

    users = sorted(set(user_to_recommended.keys()) & set(user_to_relevant.keys()))

    if not users:
        return {'precision@K': 0.0, 'recall@K': 0.0, 'map@K': 0.0}
    
    p_list, r_list, ap_list = [], [], []

    for u in users:
        recs = user_to_recommended[u]
        rel = user_to_relevant[u]
        p_list.append(precision_at_k(recs, rel, k))
        r_list.append(recall_at_k(recs, rel, k))
        ap_list.append(average_precision_at_k(recs, rel, k))

    return {
        'precision@K': mean_metric(p_list),
        'recall@K': mean_metric(r_list),
        'map@K': mean_metric(ap_list),
        'num_users': float(len(users))
    }