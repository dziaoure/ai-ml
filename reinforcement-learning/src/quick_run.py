from src.env import PricingBanditEnv
from src.agents import EpsilonGreedyAgent, UCBAgent
from src.simulate import run_simulation

def main():
    prices = [9, 12, 15, 18, 22]
    probs = [0.30, 0.24, 0.18, 0.12, 0.07]  # hidden truth
    
    env = PricingBanditEnv(prices, probs, seed=42)
    eg = EpsilonGreedyAgent(env.n_arms, epsilon=0.1, seed=42)
    df1 = run_simulation(env, eg, steps=5000)
    print('Epsilon-Greedy final cumulative reward:', df1['cum_reward'].iloc[-1])
    print('Epsilon-Greedy final cuulative regret:', df1['cum_regret'].iloc[-1])

    env2 = PricingBanditEnv(prices, probs, seed=42)
    ucb = UCBAgent(env2.n_arms)
    df2 = run_simulation(env2, ucb, steps=5000)
    print('UCB final cumulative reward:', df2['cum_reward'].iloc[-1])
    print('UCB final cuulative regret:', df2['cum_regret'].iloc[-1])


if __name__ == '__main__':
    main()