from __future__ import annotations
import numpy as np

class PricingBanditEnv:
    '''
    K-armed bandit for dynamic pricing.
    Each arm is a price with a hidden conversion probability p_i.
    Reward = price if conversion occurs else 0
    '''
    def __init__(self, prices: list[float], conversion_probs: list[float], seed: int = 42):
        assert len(prices) == len(conversion_probs)

        self.prices = np.array(prices, dtype=float)
        self.p = np.array(conversion_probs, dtype=float)
        self.rng = np.random.default_rng(seed)

    @property
    def n_arms(self) -> int:
        return len(self.prices)
    
    def step(self, action: int) -> float:
        # Bernoulli purchase with prob p[action]
        buy = self.rng.random() < self.p[action]
        return float(self.prices[action] if buy else 0.0)
    
    def expected_rewards(self) -> np.ndarray:
        return self.prices * self.p
        