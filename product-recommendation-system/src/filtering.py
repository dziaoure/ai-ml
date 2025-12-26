from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

@dataclass(frozen=True)
class ItemCFModel:
    '''
    Lightweight container for item-based collaborative filtering artifacts
    '''
    movie_ids: np.ndarray               # index -> movieId
    movie_id_to_index: Dict[int, int]   # movieId -> index
    user_ids: np.ndarray                # index -> userId
    user_id_to_index: Dict[int, int]    # userId -> index
    user_item_matrix: csr_matrix
    item_item_sim: np.ndarray


def build_user_item_matrix(
    ratings: pd.DataFrame,
    use_implicit: bool = True
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    '''
    Builds a sparse user-item matrix from ratings

    if use_implicity = True:
        - treat any rating as an interaction of value 1.0
    Else:
        - use the rating value itself (explicit feedback)

    Returns:
        (matrix, user_ids, movie_ids, user_id_to_index, movie_id_to_index)
    '''
    users = np.sort(ratings['userId'].unique())
    movies = np.sort(ratings['movieId'].unique())

    user_id_to_index = { uid: i for i, uid in enumerate(users) }
    movie_id_to_index = { mid: j for j, mid in enumerate(movies) }

    row_idx = ratings['userId'].map(user_id_to_index).to_numpy()
    col_idx = ratings['movieId'].map(movie_id_to_index).to_numpy()

    if use_implicit:
        data = np.ones(len(ratings), dtype = np.float32)
    else:
        data = ratings['rating'].to_numpy(dtype = np.float32)

    mat = csr_matrix((data, (row_idx, col_idx)), shape = (len(users), len(movies)))

    return mat, users, movies, user_id_to_index, movie_id_to_index

def fit_item_cf(
    train_ratings: pd.DataFrame,
    use_implicit: bool = True,
    shrinkage: float = 0.0
) -> ItemCFModel:
    '''
    Fit item-based CF by computing item-based cosine similarity

    shrinkage:
        - optional diagonal shrinkage to reduce overconfidence in similarity
        - implemented as sim = sim  (1 + shrinkage) (simple, stable)
    '''
    uim, user_ids, movie_ids, u_map, m_map = build_user_item_matrix(
        train_ratings, use_implicit=use_implicit
    )

    # Cosine similarity between solumns (items)
    # Easiest is to compute on transposed matrix (items x users)
    item_user = uim.T   # shape: (num_items, num_users)

    sim = cosine_similarity(item_user, dense_output=True)   # (num_items, num_items)

    # Optional shrinkage
    if shrinkage and shrinkage > 0:
        sim = sim / (1.0 + float(shrinkage))

    # Avoid self=recommending by zeroing diagonal (we'll alo filter 'seen' array)
    np.fill_diagonal(sim, 0.0)

    return ItemCFModel(
        movie_ids = movie_ids,
        movie_id_to_index = m_map,
        user_ids = user_ids,
        user_id_to_index = u_map,
        user_item_matrix = uim,
        item_item_sim = sim
    )


def recommend_for_user_item_cf(
    model: ItemCFModel,
    user_id: int,
    train_ratings: pd.DataFrame,
    k: int = 10,
    candidate_pool: int = 200
) -> List[int]:
    '''
    Recomend top-k items for a user using item-item CF

    Method:
    - user profile = items they've interacted with in training
    - score(candidate item) = sum(sim(candidate, item_seen))
    - return top-k unseen items
    '''
    if user_id not in model.user_id_to_index:
        # Cold-start user: no training interactions in matrix
        return []
    
    seen = train_ratings.loc[train_ratings['userId'] == user_id, 'movieId'].tolist()
    seen_set = set(int(x) for x in seen)

    if not seen_set:
        return []
    
    # Aggregate scores in a dict (movieId -> score)
    scores: Dict[int, float] = {}

    for mid in seen_set:
        if mid not in model.movie_id_to_index:
            continue

        j = model.movie_id_to_index[mid]
        sim_row = model.item_item_sim[j]    # Similarity of 'mid' to all items

        # Limit to strongest candidates to reduce work
        if candidate_pool and candidate_pool > 0:
            top_idx = np.argpartition(sim_row, -candidate_pool)[-candidate_pool:]
        else:
            top_idx = np.arange(len(sim_row))

        for idx in top_idx:
            cand_movie_id = int(model.movie_ids[idx])

            if cand_movie_id in seen_set:
                continue

            s = float(sim_row[idx])

            if s <= 0:
                continue

            scores[cand_movie_id] = scores.get(cand_movie_id, 0.0) + s

    if not scores:
        return []
    
    ranked = sorted(scores.items(), key = (lambda x: x[1]), reverse = True)

    return [mid for mid, _ in ranked[:k]]

