from __future__ import annotations
import numpy as np

class EpsilonGreedyAgent:

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 42):
        self.n_arms = n_arms
        self.epsilon = float(epsilon)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        
        return int(np.argmax(self.values))
    

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]

        # Increment mean update
        self.values[action] += (reward - self.values[action]) / n


class UCBAgent:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1

        # Force each arm to be tried once
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
            
        ucb = self.values + np.sqrt((2 * np.log(self.t)) / self.counts)
        return int(np.argmax(ucb))
    
    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
