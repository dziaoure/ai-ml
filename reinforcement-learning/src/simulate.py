from __future__ import annotations
import numpy as np
import pandas as pd

from src.env import PricingBanditEnv

def run_simulation(env: PricingBanditEnv, agent, steps: int = 5000) -> pd.DataFrame:
    exp_rewards = env.expected_rewards()
    optimal = float(np.max(exp_rewards))

    rows = []
    cum_reward = 0.0
    cum_regret = 0.0

    for t in range(1, steps + 1):
        action = agent.select_action()
        reward = env.step(action)
        agent.update(action, reward)

        cum_reward += reward
        regret = optimal - exp_rewards[action]
        cum_regret += regret

        rows.append({
            't': t,
            'action': action,
            'cum_reward': cum_reward,
            'instant_regreet': regret,
            'cum_regret': cum_regret 
        })

    return pd.DataFrame(rows)