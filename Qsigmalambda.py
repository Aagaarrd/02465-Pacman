import gym
import matplotlib.pyplot as plt
from irlc.common import defaultdict2
from irlc.irlc_plot import main_plot
from irlc.agent import Agent, train
import numpy as np


class Qsigmalambda(Agent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9):
        self.alpha = alpha
        self.lamb = lamb
        super().__init__(env, gamma=gamma, epsilon=epsilon)
        self.e = defaultdict2(self.Q.default_factory)

    def pi_probs(self, s):
        a = np.argmax(self.Q[s][:])
        pi_probs = np.ones(self.env.nA) * self.epsilon / self.env.nA
        pi_probs[a] += (1 - self.epsilon)
        return pi_probs

    def pi(self, s):
        return np.random.choice(range(self.env.nA), p=self.pi_probs(s))

    def train(self, s, a, r, sp, done=False):
        pi_probs = self.pi_probs(sp)
        ap = self.pi(sp)
        sigma = self.get_sigma(a)
        sarsa_target = self.Q[sp][ap]
        exp_sarsa_target = np.dot(pi_probs, self.Q[sp])
        td_target = r + self.gamma * (sigma * sarsa_target + (1 - sigma) * exp_sarsa_target)
        delta = td_target - self.Q[s][a]
        self.e[s][a] += 1
        self.Q[s][a] += self.alpha*delta*self.e[s][a]
        self.e[s][a] = self.gamma*self.lamb*self.e[s][a]*(sigma+(1-sigma)*self.pi_probs(sp)[ap])

    def get_sigma(self, a):
        return np.random.randint(2, size=self.env.nA)[a]


if __name__ == "__main__":
    envn = 'CliffWalking-v0'
    env = gym.make(envn)
    agent = Qsigmalambda(env, gamma=0.9, epsilon=0.1, alpha=0.5)
    agent_name = "Q_sigmalambda"
    exp = f"experiments/{envn}_{agent_name}"
    train(env, agent, exp, num_episodes=200, max_runs=5)
    main_plot(exp, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.show()
