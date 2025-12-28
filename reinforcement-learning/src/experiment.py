from __future__ import annotations

import numpy as np
import pandas as pd

from src.env import PricingBanditEnv
from src.agents import EpsilonGreedyAgent, UCBAgent
from src.simulate import run_simulation


def make_env(prices: list[float], probs: list[float], seed: int) -> PricingBanditEnv:
    return PricingBanditEnv(prices, probs, seed=seed)


def run_many(
    prices: list[float],
    probs: list[float],
    agent_name: str,
    steps: int = 5000,
    n_runs: int = 10,
    base_seed: int = 42,
    epsilon: float = 0.1
) -> list[pd.DataFrame]:
    runs = []

    for i in range(n_runs):
        seed = base_seed + i
        env = make_env(prices, probs, seed=seed)

        if agent_name == 'epsilon_greedy':
            agent = EpsilonGreedyAgent(env.n_arms, epsilon=epsilon, seed=seed)
        elif agent_name == 'ucb':
            agent = UCBAgent(env.n_arms)
        else:
            raise ValueError('`agent_name` must be "epsilon_greey" or "ucb"')
        
        df = run_simulation(env, agent, steps=steps)
        df['run'] = i
        runs.append(df)
    
    return runs
