# based on Algorithm 1 from https://arxiv.org/pdf/1711.01569.pdf
import gym
import matplotlib.pyplot as plt
from irlc.common import defaultdict2
from irlc.irlc_plot import main_plot
from irlc.agent import Agent, train
from sarsa_agent import SarsaAgent
from exp_sarsa_agent import ExpSarsaAgent
import gym_windy_gridworlds
import numpy as np


class QslAgent(SarsaAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, sigma=1, sigma_strat='dynamic', alpha=0.5, lamb=0.9):
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        self.lamb = lamb
        self.e = defaultdict2(self.Q.default_factory)
        self.sigma = sigma
        self.sigma_strat = sigma_strat


    def pi_probs(self, s):
        a = np.argmax(self.Q[s])
        pi_probs = np.ones(self.env.nA) * self.epsilon / self.env.nA
        pi_probs[a] += (1 - self.epsilon)
        return pi_probs

    def pi(self, s):
        if self.t == 0:
            return self.pi_eps(s)
        else:
            p = self.pi_probs(s)
            return np.random.choice(np.arange(0, self.env.nA), p=p)

    def train(self, s, a, r, sp, done=False):
        pi_probs = self.pi_probs(sp)
        ap = self.pi_eps(sp)
        sigma = self.sigma*0.9 if self.sigma_strat == 'dynamic' else self.sigma
        sarsa_target = self.Q[sp][ap]
        exp_sarsa_target = np.dot(pi_probs, self.Q[sp])
        td_target = r + self.gamma * (sigma * sarsa_target + (1 - sigma) * exp_sarsa_target if not done else 0)
        td_error = td_target - self.Q[s][a]
        self.e[s][a] += 1
        for s, es in self.e.items():
            for a, e_sa in enumerate(es):
                self.Q[s][a] += self.alpha * td_error * self.e[s][a]
                self.e[s][a] *= self.gamma * self.lamb * (sigma + (1 - sigma) * pi_probs[ap])

        if self.t > 1000:
            done = True

        if done:
            self.e.clear()
            self.sigma = 1 if self.sigma_strat == 'dynamic' else sigma
        else:
            self.a = ap
            self.sigma = sigma
            self.t += 1

    def __str__(self):
        agent = f"Q($\\sigma={self.sigma_strat}-{self.sigma},\\lambda={self.lamb}$)"
        return f"{agent}($\\gamma={self.gamma},\\epsilon={self.epsilon},\\alpha={self.alpha}$)"


def run_exp(env, num_episodes=50, epsilon=0.1, alpha=0.6, gamma=0.90):
    for _ in range(50):
        agents = [SarsaAgent(env, epsilon=epsilon, alpha=alpha, gamma=gamma),
                  ExpSarsaAgent(env, epsilon=epsilon, alpha=alpha, gamma=gamma),
                  QslAgent(env, epsilon=epsilon, alpha=alpha, gamma=gamma, sigma_strat='static', sigma=0.5, lamb=1),
                  QslAgent(env, epsilon=epsilon, alpha=alpha, gamma=gamma, lamb=0.8)]

        experiments = []
        for agent in agents:
            expn = f"experiments/{str(agent)}"
            train(env, agent, expn, num_episodes=num_episodes, max_runs=100)
            experiments.append(expn)
    return experiments


if __name__ == "__main__":
    envn = 'StochWindyGridWorld-v0'
    env = gym.make(envn)
    experiments = run_exp(env, num_episodes=200)
    main_plot(experiments, smoothing_window=15)
    plt.ylim([-100, -30])
    plt.savefig('plot.png')
    plt.show()
