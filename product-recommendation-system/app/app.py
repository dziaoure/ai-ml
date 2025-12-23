from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# --- Fix imports when runing `streamlit` from `/app`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_movielens, time_based_split
from src.baselines import PopularityRecommender
from src.filtering import fit_item_cf, recommend_for_user_item_cf


#----------------------------------------
#       Caching/Loading
#----------------------------------------

MOVIE_COLUMNS = ['movieId', 'title', 'genres']

@st.cache_data(show_spinner = False)
def load_Data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings, movies = load_movielens(data_dir)
    return ratings, movies

@st.cache_data(show_spinner = False)
def split_data(ratings: pd.DataFrame, test_ratio: float, min_ratings_per_user: int):
    train, test = time_based_split(
        ratings,
        test_ratio = test_ratio,
        min_ratings_per_user = min_ratings_per_user
    )

    return train, test

@st.cache_resource(show_spinner = False)
def build_pop_model(train: pd.DataFrame, min_ratings: int) -> PopularityRecommender:
    return PopularityRecommender(min_ratings=min_ratings).fit(train)

@st.cache_resource(show_spinner = False)
def build_item_cf_model(train: pd.DataFrame, use_implicit: bool, shrinkage: float):
    return fit_item_cf(train_ratings = train, use_implicit = use_implicit, shrinkage = shrinkage)

def _movies_df(movies: pd.DataFrame, movie_ids: List[int]) -> pd.DataFrame:
    if not movie_ids:
        return pd.DataFrame(columns = MOVIE_COLUMNS)
    
    return (
        movies[movies['movieId'].isin(movie_ids)][MOVIE_COLUMNS]
        .drop_duplicates()
        .set_index('movieId')
        .loc[movie_ids] # Keep recommendation order
        .reset_index()
    )


def _recent_history(train: pd.DataFrame, movies: pd.DataFrame, user_id: int, n: int = 15) -> pd.DataFrame:
    hist = (
        train[train['userId'] == user_id]
        .merge(movies[MOVIE_COLUMNS], on = 'movieId', how = 'left')
        .sort_values('timestamp')
    )

    if hist.empty:
        return hist
    
    return hist.tail(n)[['movieId', 'title', 'rating', 'timestamp']]

def _simple_explanations(item_cf_model, train: pd.DataFrame, recs: List[int], user_id: int, top_n: int = 2):
    '''
    Lightweight 'Because you watched...' explanation
    For each recommended item, find the most similar watched items
    '''
    seen = train.loc[train['userId'] == user_id, 'movieId'].tolist()
    seen_set = [int(x) for x in seen if int(x) in item_cf_model.movie_id_to_index]

    explanations = {}

    for rec_mid in recs:
        if rec_mid not in item_cf_model.movie_id_to_index:
            continue

        rec_idx = item_cf_model.movie_id_to_index[rec_mid]
        sims = []

        for seen_mid in seen_set:
            sidx = item_cf_model.movie_id_to_index[seen_mid]
            sim_val = float(item_cf_model.item_item_sim[sidx, rec_idx])
            sims.append((seen_mid, sim_val))
        
        sims.sort(key =(lambda x: x[1]), reverse=True)
        best = [mid for mid, val in sims[:top_n] if val > 0]
        explanations[rec_mid] = best

    return explanations


#-----------------------------------------------
#           Streamlit UI
#-----------------------------------------------
st.set_page_config(page_title = 'Movie Recommender - MovieLens', layout = 'wide')

st.title('🎬 Movie Recommender - MovieLens')
st.caption('Popularity baseline vs Item-based collaborative filtering (cosine similarity)')

data_dir = PROJECT_ROOT / 'data'

if not (data_dir / 'ratings.csv').exists() or not (data_dir / 'movies.csv').exists():
    st.error(
        'Missing MovieLens films. Expected:]n'
        '- data/ratings.csv\n'
        '- data/movies.csv\n'
        'Download MovieLens (`ml-latest-small`) and place them CSVs in the `data/` folder'
    )
    st.stop()

ratings, movies = load_movielens(data_dir)

with st.sidebar:
    st.header('Settings')

    test_ratio = st.slider('Test ratio (time-based split)', min_value = 0.1, max_value = 0.4, value = 0.2, step = 0.05)
    min_ratings_per_user = st.slider('Min ratings per user', min_value = 3, max_value = 20, value = 5, step = 1)

    st.divider()
    model_choice = st.selectbox('Model', ['Popularity (baseline)', 'FIltering (item-based CF)'])
    k = st.slider('Top-K recommendations', min_value = 5, max_value = 25, value = 10, step = 5)

    st.divider()
    pop_min_ratings = st.slider('Popularity min ratings per movie', min_value = 10, max_value = 200, value = 50, step = 10)

    st.divider()
    use_implicit = st.toggle('Use implicit interactions (recommended)', value = True)
    shrinkage = st.slider('Similarity Shrinkage', min_value = 0.0, max_value = 2.0, value = 0.0, step = 0.25)
    candidate_pool = st.slider('Candidate pool per watched items', min_value = 50, max_value = 1000, value = 200, step = 10)

train, test = split_data(ratings, test_ratio=test_ratio, min_ratings_per_user=min_ratings_per_user)

st.write(
    f'**Train:** {len(train):,} ratings • **Test:** {len(test):,} ratings • '
    f'**Users:** {train["userId"].nunique():,} • **Movies:** {train["movieId"].nunique():,}'
)

# Choose a user
user_ids = sorted(train['userId'].unique().tolist())
default_user = user_ids[min(0, len(user_ids) - 1)]
user_id = st.selectbox('Pick a user', user_ids, default_user)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('Recent User History (train)')
    hist = _recent_history(train, movies, user_id = user_id, n = 15)

    if hist.empty:
        st.info('No history found for this user during training')
    else:
        st.dataframe(hist, use_container_width = True)

with col2:
    st.subheader('Recommendations')

    # Build models (cached)
    pop_model = build_pop_model(train, min_ratings=pop_min_ratings)
    item_cf_model = build_item_cf_model(train,use_implicit=use_implicit, shrinkage=shrinkage)

    #Recommended
    if model_choice.startswith('Popularity'):
        recommended = pop_model.recommend(user_id=user_id, train_df=train, k=k)
        rec_df = _movies_df(movies, recommended)
        st.dataframe(rec_df, use_container_width=True)
        st.caption('Baseline: Top popular movies, excluding those already seen during training')
    else:
        recommended = recommend_for_user_item_cf(
            model=item_cf_model,
            user_id = user_id,
            train_ratings=train,
            k=k,
            candidate_pool=candidate_pool
        )

        if not recommended:
            st.warning('No recommendations found (user may be a cold-start or may have sparse overlap)')
        else:
            rec_df = _movies_df(movies, recommended)
            st.dataframe(rec_df, use_container_width=True)

            # Simple explanations
            expl = _simple_explanations(item_cf_model, train, recommended, user_id=user_id, top_n=2)
            st.markdown('**Explanation (lightweight):**')

            for mid in recommended[:min(10, len(recommended))]:
                because = expl.get(mid, [])

                if because:
                    because_titles = movies[movies['movieId'].isin(because)][['movieId', 'title']]
                    because_list = ', '.join(because_titles['title'].tolist())
                    st.write(f'- **{rec_df[rec_df['movieId'] == mid]['title'].iloc[0]}** - because you watched {because_list}')
                else:
                    st.write(f'- **{rec_df[rec_df['movieId'] == mid]['title'].iloc[0]}**')
            st.caption('Filtering: item-based collaborative filtering using cosine similarity.')