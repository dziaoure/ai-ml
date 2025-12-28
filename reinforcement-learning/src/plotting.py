from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_cumulative_reward(dfs: dict[str, pd.DataFrame], title: str = 'Cumulative Reward', image_name: str = 'cumulative-reward'):
    plt.figure()

    for name, df in dfs.items():
        plt.plot(df['t'], df['cum_reward'], label=name)
    
    plt.xlabel('t')
    plt.ylabel('Cumulative Reward')
    plt.title(title)
    plt.legend()
    plt.savefig(f'../images/{image_name}.png')
    plt.show()


def plot_cumulative_regret(dfs: dict[str, pd.DataFrame], title: str = 'Cumulative Regret', image_name: str = 'cumulative-regret'):
    plt.figure()

    for name, df in dfs.items():
        plt.plot(df['t'], df['cum_regret'], label=name)
   
    plt.xlabel('t')
    plt.ylabel('Cumulative Regret')
    plt.title(title)
    plt.legend()
    plt.savefig(f'../images/{image_name}.png')
    plt.show()


def plot_action_counts(dfs: dict[str, pd.DataFrame], prices: list[float], title: str = 'Action Selection Counts'):
    n_agents = len(dfs)
    plt.figure()

    for name, df in dfs.items():
        counts = df['action'].value_counts().sort_index()

        # Remove all arms included
        counts = counts.reindex(range(len(prices)), fill_value=0)
        plt.plot(range(len(prices)), counts.values, marker='o', label=name)

    plt.xticks(range(len(prices)), [f'${p:g}' for p in prices])
    plt.xlabel('Price (action)')
    plt.ylabel('TImes selected')
    plt.title(title)
    plt.legend()
    plt.show()


def summarize_runs(runs: list[pd.DataFrame]) -> pd.DataFrame:
    '''
    Given multiple run DataFrames with columns:
    t, cum_reward, cum_regret
    return a single DF with mean/std over time
    '''
    base = runs[0][['t']].copy()
    cum_reward_mat = np.vstack([df['cum_reward'].to_numpy() for df in runs])
    cum_regret_mat =  np.vstack([df['cum_regret'].to_numpy() for df in runs])

    base['cum_reward_mean'] = cum_reward_mat.mean(axis=0)
    base['cum_reward_std'] = cum_reward_mat.std(axis=0)

    base['cum_regret_mean'] = cum_regret_mat.mean(axis=0)
    base['cum_regret_std'] = cum_regret_mat.std(axis=0)

    return base


def plot_mean_with_std(
        summary: pd.DataFrame, 
        y_mean_col: str, 
        y_std_col: str, 
        title: str, 
        ylabel: str, 
        label: str,
        image_name: str):
    plt.figure()
    t = summary['t'].to_numpy()
    y = summary[y_mean_col].to_numpy()
    s = summary[y_std_col].to_numpy()

    plt.plot(t, y, label=label)
    plt.fill_between(t, y - s, y + s, alpha=0.2)
    plt.xlabel('t')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f'../images/{image_name}.png')
    plt.show()

