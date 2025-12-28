from __future__ import annotations
import sys

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Fix imports when runing `streamlit` from `/app`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.experiment import run_many
from src.env import PricingBanditEnv
from src.plotting import summarize_runs


LABELS = ['$9', '$12', '$15', '$28', '$22']

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class BanditConfig:
    prices: list[float]
    probs: list[float]
    steps: int
    runs: int
    agent: str
    epsilon: float
    base_seed: int

def plot_mean_std(
    summary: pd.DataFrame,
    y_mean: str,
    y_std: str,
    title: str,
    ylabel: str
):
    t = summary['t'].to_numpy()
    mean = summary[y_mean].to_numpy()
    std = summary[y_std].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(t, mean, label='mean')
    ax.fill_between(t, mean - std, mean + std, alpha=0.2, label='± std')
    ax.set_title(title)
    ax.set_xlabel('t')
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)


def plot_action_counts_avg(runs: list[pd.DataFrame], prices: list[float], title: str):
    n_arms = len(prices)

    # Build counts per run, then average
    counts_mat = []

    for df in runs:
        counts = df['action'].value_counts().sort_index()
        counts = counts.reindex(range(n_arms), fill_value=0).to_numpy()
        counts_mat.append(counts)

    counts_mat = np.vstack(counts_mat)
    avg_counts = counts_mat.mean(axis=0)
    std_counts = counts_mat.std(axis=0)

    fig, ax = plt.subplots()
    x = np.arange(n_arms)
    ax.bar(x, avg_counts, yerr=std_counts, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'${p:g}' for p in prices])
    ax.set_title(title)
    ax.set_xlabel('Price (action)')
    ax.set_ylabel('Times selected (avg ± std)')
    st.pyplot(fig)


def compute_expected_value_table(prices: list[float], probs: list[float]) -> pd.DataFrame:
    ev = np.array(prices) * np.array(probs)
    df = pd.DataFrame({
        'Price': [f'${p:g}' for p in prices],
        'Conversion Prob (hidden)': probs,
        'Expected Value (Price * Prob)': ev
    })
    df['Rank (EV)'] = df['Expected Value (Price * Prob)'].rank(ascending=False, method='dense').astype(int)
    df = df.sort_values('Rank (EV)')

    return df

def load_css():
    css_path = Path(__file__).resolve().parent / "styles.css"

    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

def main():
    st.set_page_config('Reinforcement Learning - Dynamic Pricing Bandit', layout='wide')

    load_css()

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        image_path = project_root() / 'images/ai-icon.png'
        st.image(image_path, width=120)

    st.title('Reinforcement Learning')
    subtitle_html = """
        <p style="text-align: center;color:#fff; opacity:0.65; margin: 0 40px;">
        This app compares <strong>Episilon-Greedy</strong> and <strong>UCB</strong> agents on a
        simulated dynamic pricing task.
        The agents repeatedly select a price and receive a reward (revenue). 
        We track <strong>cumulative reward</strong> and <strong>cumulative regret</strong>.
        </p>
    """
    st.markdown(subtitle_html, unsafe_allow_html=True)

    with st.sidebar:
        st.header('Experiment Settings')

        agent = st.selectbox(
            'Agent',
            options=['epsilon_greedy', 'ucb'],
            format_func=(lambda x: 'Expsilon-Greedy' if x == 'epsilon_greedy' else 'UCB1')
        )

        steps = st.slider('Steps (t)', min_value=500, max_value=20000, value=5000, step=500)
        runs = st.slider('Number of runs', min_value=1, max_value=30, value=8, step=1)

        epsilon = st.slider('Epsilon (only for Epsilon-Greedy)', min_value=0.0, max_value=0.5, value=0.1, step=0.01)

        base_seed = st.number_input('Base seed', min_value=0, max_value=1_000, value=42, step=1)

        st.markdown('---')

        st.subheader('Bandit Setup')

        # dDefault setup
        prices = [9, 12, 15, 18, 22]
        probs = [0.30, 0.24, 0.18, 0.12, 0.07]

        st.caption('Prices and conversion probabilities are fixes in this demo.')
        st.write('Prices: ', prices)
        st.write('Conversrion probs: ', probs)

    cfg = BanditConfig(
        prices=prices,
        probs=probs,
        steps=int(steps),
        runs=int(runs),
        agent=agent,
        epsilon=float(epsilon),
        base_seed=int(base_seed)
    )

    # Show the "ground truth" expected values for this simulation (great for explantion)
    st.subheader('Experiment Ground Truth (for this simulation')
    ev_df = compute_expected_value_table(cfg.prices, cfg.probs)
    st.dataframe(ev_df, width='stretch')

    best_row = ev_df.iloc[0]
    st.info(f'Optimal price by expected value: **{best_row["Price"]}**')

    if st.button('Run Simulation', type='primary'):
        with st.spinner('Running Simulation...'):
            runs_df = run_many(
                cfg.prices,
                cfg.probs,
                agent_name=cfg.agent,
                steps=cfg.steps,
                n_runs=cfg.runs,
                base_seed=cfg.base_seed,
                epsilon=cfg.epsilon
            )

            summary = summarize_runs(runs_df)

        # Sunmary metrics
        final_reward = float(summary['cum_reward_mean'].iloc[-1])
        final_regret = float(summary['cum_regret_mean'].iloc[-1])

        st.write('---')

        c1, c2, c3 = st.columns([3, 2, 2])
        c1.metric('Agent', 'Epsilon-Greedy' if cfg.agent == 'epsilon_greedy' else 'UCB1')
        c2.metric('Final cumulatie reward (mean)', f'{final_reward:.0f}')
        c3.metric('Final cumulative regret: (mean)', f'{final_regret:.0f}')

        st.markdown('---')

        left, right = st.columns(2)

        with left:
            st.subheader('Cumulative Reward (mean ± std)')
            plot_mean_std(
                summary,
                y_mean='cum_reward_mean',
                y_std='cum_reward_std',
                title='Cumulative Reward',
                ylabel='Cumulative Reward'
            )

        with right:
            st.subheader('Cumulative Regret (mean ± std)')
            plot_mean_std(
                summary,
                y_mean='cum_regret_mean',
                y_std='cum_regret_std',
                title='Cumulative Regret',
                ylabel='Cumulative Regret'
            )

        st.subheader('Action Selection Distribution (avg ± std across runs)')
        plot_action_counts_avg(runs_df, cfg.prices, title='Action Selection Counts')

        # Optional: show one run's tail to debug learning behavior)
        with st.expander('Show last 15 ros of run 0'):
            st.dataframe(runs_df[0].tail(15), width='stretch')


if __name__ == '__main__':
    main()
