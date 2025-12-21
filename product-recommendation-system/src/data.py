from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class MovieLensPaths():
    ''' Convenience container for MovieLens file paths '''
    data_dir:Path

    @property
    def ratings_path(self) -> Path:
        return self.data_dir / 'ratings.csv'
    
    @property
    def movies_path(self) -> Path:
        return self.data_dir / 'movies.csv'
    
REQUIRED_RATINGS_COLS = {'userId', 'movieId', 'rating', 'timestamp'}
REQUIRED_MOVIES_COLS = {'movieId', 'title', 'genres'}


def load_ratings(path:str | Path) -> pd.DataFrame:
    '''
    Load MovieLens ratings.csv
    Expected columns: userId, movieId, rating, timestamp
    '''
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_RATINGS_COLS - set(df.columns)

    if missing:
        raise ValueError(f'ratinbs.csv missing required columns: {sorted(missing)}')
    
    # Enfore dtypes
    df = df.copy()
    df['userId'] = df['userId'].astype('int')
    df['movieId'] = df['movieId'].astype('int')
    df['rating'] = df['rating'].astype('float')
    df['timestamp'] = df['timestamp'].astype('int')
    
    # Basic Validation
    if df.empty:
        raise ValueError('ratings dataframe is empty')
    
    if (df['rating'] <= 0).any():
        # MovieLens ratings are between 0.5 and 5.0
        # We'll allow 0.0 in case of alternate versions, but negative ratings are invalid
        if (df['rating'] < 0).any():
            raise ValueError('ratings dataframe contains negative ratings')
    
    return df


def load_movies(path:str | Path) -> pd.DataFrame:
    '''
    Load MovieLens movies.csv
    Expected columns: movieId, title, genres
    '''
    path = Path(path)
    df = pd.read_csv(path)  
    
    missing = REQUIRED_MOVIES_COLS - set(df.columns)

    if missing:
        raise ValueError(f'movies.csv missing required columns: {sorted(missing)}')
    
    df = df.copy()
    df['movieId'] = df['movieId'].astype('int')
    
    if df.empty:
        raise ValueError('movies dataframe is empty')
    
    return df


def load_movielens(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Load (ratings, movies) from a MovieLens folder containing ratings.csv and movies.csv
    '''
    paths = MovieLensPaths(Path(data_dir))

    if not paths.ratings_path.exists():
        raise FileNotFoundError(f'Missing file: {paths.ratings_path}')
    
    if not paths.movies_path.exists():
        raise FileNotFoundError(f'Missing file: {paths.movies_path}')
    
    ratings = load_ratings(paths.ratings_path)
    movies = load_movies(paths.movies_path)

    return ratings, movies


def time_based_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_ratings_per_user: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Time-based split per user:
    - For each user, sort by timestamp
    - Put the most recent `test_ratio` proportion of interactions into the test set
    - Keep earlier interactions in the training set
    
    Why per user?
    - Prevents leakage from users with many ratings dominating the test set
    - Ensures every user can be evaluated (if enough history)
    
    Users with fewer than `min_ratings_per_user` ratings are removed from both splits.

    Parameters:
    - ratings: DataFrame with columns ['userId', 'movieId', 'rating', 'timestamp']
    - test_ratio: Proportion of each user's ratings to assign to the test set
    - min_ratings_per_user: Minimum number of ratings a user must have to be included in the split
    
    Returns:
    - train: Training set DataFrame
    - test: Test set DataFrame
    '''

    if not 0.0 < test_ratio < 1.0:
        raise ValueError('test_ratio must be between 0.0 and 1.0')

    # Filter users with enough interactions
    counts = ratings['userId'].value_counts()
    keep_users = counts[counts >= min_ratings_per_user].index
    filtered = ratings[ratings['userId'].isin(keep_users)].copy()

    if filtered.empty:
        raise ValueError('No users left after filtering; lower min_ratings_per_user or provide more data.')
    
    # Sort once for stable grouping
    filtered.sort_values(['userId', 'timestamp'], inplace=True)

    def _split_group(g: pd.DataFrame) -> pd.DataFrame:
        n = len(g)
        n_test = max(1, int(round(n * test_ratio)))
        
        # Last n_test rows -> test
        mask = [False] * (n - n_test) + [True] * n_test
        g = g.copy()
        g['_is_test'] = mask

        return g
    
    tmp = filtered.groupby('userId', group_keys=False).apply(_split_group)
    test = tmp[tmp['_is_test']].drop(columns=['_is_test'])
    train = tmp[~tmp['_is_test']].drop(columns=['_is_test'])

    # Final sanity check
    if train.empty or test.empty:
        raise ValueError('Resulting train or test set is empty; adjust test_ratio or min_ratings_per_user.')
    
    return train, test
