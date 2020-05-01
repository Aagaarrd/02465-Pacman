# based on Algorithm 1 from https://arxiv.org/pdf/1711.01569.pdf
import gym
import matplotlib.pyplot as plt
from irlc.common import defaultdict2
from irlc.irlc_plot import main_plot
from irlc.agent import Agent, train
from sarsa_agent import SarsaAgent
import gym_windy_gridworlds
import numpy as np
# np.seterr('raise')


class Qsigmalambda(SarsaAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9):
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        self.lamb = lamb
        self.e = defaultdict2(self.Q.default_factory)

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
        sigma = self.get_sigma(a)
        sarsa_target = self.Q[sp][ap]
        exp_sarsa_target = np.dot(pi_probs, self.Q[sp])
        td_target = r + self.gamma * (sigma * sarsa_target + (1 - sigma) * exp_sarsa_target)
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
        else:
            self.a = ap
            self.t += 1

    def get_sigma(self, a):
        return np.random.randint(2, size=self.env.nA)[a]

    def __str__(self):
        return f"Q(\sigma, \lambda={self.lamb})_{self.gamma}_{self.epsilon}_{self.alpha}"



if __name__ == "__main__":
    from q_agent import experiment as q_exp
    from sarsa_agent import experiment as sarsa_exp
    env, q_exp = q_exp()
    env, sarsa_exp = sarsa_exp()

    envn = 'StochWindyGridWorld-v0'
    env = gym.make(envn)
    agent = Qsigmalambda(env, gamma=0.9, epsilon=0.1, alpha=0.5, lamb=0.7)
    exp = f"experiments/{envn}_{str(agent)}"
    train(env, agent, exp, num_episodes=200, max_runs=10)
    main_plot([exp, sarsa_exp, q_exp], smoothing_window=10)
    plt.ylim([-100, 0])
    plt.show()
