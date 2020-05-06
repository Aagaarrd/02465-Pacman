"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import matplotlib.pyplot as plt
import numpy as np
from q_agent import QAgent
from irlc.irlc_plot import main_plot
from irlc.agent import train
from q_agent import experiment as q_agent_exp
import gym
from irlc.common import defaultdict2


class ExpSarsaAgent(QAgent):
    def __init__(self, env, gamma=0.99, alpha=0.5, epsilon=0.1):
        self.t = 0  # indicate we are at the beginning of the episode
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    def pi_probs(self, s):
        a = np.argmax(self.Q[s])
        pi_probs = np.ones(self.env.nA) * self.epsilon / self.env.nA
        pi_probs[a] += (1 - self.epsilon)
        return pi_probs

    def pi(self, s):
        if self.t == 0:  # !f
            """ we are at the beginning of the episode. Generate a by being epsilon-greedy"""
            return self.pi_eps(s)
        else:  # !f
            """ Return the action self.a you generated during the train where you know s_{t+1} """
            return self.a

    def train(self, s, a, r, sp, done=False):
        """
        generate A' as self.a by being epsilon-greedy. Re-use code from the Agent class.
        """
        self.a = self.pi_eps(sp) if not done else -1  # !b #!b self.a = ....
        pi_probs = self.pi_probs(sp)
        exp_sarsa_target = np.dot(pi_probs, self.Q[sp])
        """ now that you know A' = self.a, perform the update to self.Q[s][a] here """
        delta = r + (self.gamma * exp_sarsa_target if not done else 0) - self.Q[s][a]  # !b
        self.Q[s][a] += self.alpha * delta  # !b
        self.t = 0 if done else self.t + 1  # update current iteration number

    def __str__(self):
        return f"ExpSarsa($\\gamma={self.gamma},\\epsilon={self.epsilon},\\alpha={self.alpha}$)"


def experiment():
    envn = 'StochWindyGridWorld-v0'
    env = gym.make(envn)
    agent = ExpSarsaAgent(env, epsilon=0.1, alpha=0.5)
    exp = f"experiments/{str(agent)}"
    train(env, agent, exp, num_episodes=200, max_runs=10)
    return env, exp


if __name__ == "__main__":
    env, q_experiment = q_agent_exp()  # get results from Q-learning
    env, sarsa_exp = experiment()
    main_plot([q_experiment, sarsa_exp], smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q and Sarsa learning on " + env.spec._env_name)
    plt.show()
