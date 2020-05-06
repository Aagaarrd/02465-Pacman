"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import numpy as np
from collections import defaultdict
from irlc.agent import Agent, train
import gym
from irlc.irlc_plot import main_plot
import matplotlib.pyplot as plt
import gym_windy_gridworlds


# from irlc import savepdf


class QAgent(Agent):
    """
    Implement the Q-learning agent here. Note that the Q-datastructure already exist
    (see agent class for more information)
    """

    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)

    def pi(self, s):
        """
        Return current action using epsilon-greedy exploration. Look at the Agent class
        for ideas.
        """
        return self.pi_eps(s)

    def train(self, s, a, r, sp, done=False):
        delta = r + self.gamma * np.max(self.Q[sp][a]) - self.Q[s][a]
        self.Q[s][a] += self.alpha * delta

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"


def experiment():
    envn = 'StochWindyGridWorld-v0'
    env = gym.make(envn)
    agent = QAgent(env, epsilon=0.1, alpha=0.5)
    exp = f"experiments/{str(agent)}"
    train(env, agent, exp, num_episodes=200, max_runs=10)
    return env, exp


if __name__ == "__main__":
    env, q_exp = experiment()
    main_plot(q_exp, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q-learning on " + env.spec._env_name)
    plt.show()
