from __future__ import annotations

from typing import Dict, List

import pandas as pd

class PopularityRecommender:
    '''
    Popularity-based recommender:
    - Ranks items by rating count (then mean rating)
    - Excluded items already seen by the user
    '''

    def __init__(self, min_ratings: int = 50):
        self.min_ratings = min_ratings
        self.ranking_: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> 'PopularityRecommender':
        agg = (
            train_df.groupby('movieId')['rating']
            .agg(rating_count = 'count', rating_mean = 'mean')
            .reset_index()
        )

        agg = agg[agg['rating_count'] >= self.min_ratings].copy()
        agg.sort_values(
            ['rating_count', 'rating_mean'],
            ascending = [False, False],
            inplace = True 
        )

        self.ranking_ = agg.reset_index(drop = True)

        return self
    
    def recommend(
            self,
            user_id: int,
            train_df: pd.DataFrame,
            k: int = 10
    ) -> List[int]:
        if self.ranking_ is None:
            raise RuntimeError('PopularityRecommender must be fit() first')
        
        seen = set(
            train_df.loc[train_df['userId'] == user_id, 'movieId'].tolist()
        )

        recommended: List[int] = []

        for mid in self.ranking_['movieId']:
            if mid not in seen:
                recommended.append(int(mid))

            if len(recommended) >= k:
                break

        return recommended