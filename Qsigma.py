from irlc.agent import Agent, train
import numpy as np
from irlc.irlc_plot import main_plot
import matplotlib.pyplot as plt
import gym


class Qsigma(Agent):
    def __init__(self, env, gamma, alpha, epsilon, n):
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.R, self.S, self.A, self.rho, self.sigma = [None] * (self.n + 1), [None] * (self.n + 1), [None] * (
                self.n + 1), [None] * (self.n + 1), [None] * (self.n + 1)
        self.t = 0
        super().__init__(env, gamma, epsilon)

    def pi(self, s):
        if self.t == 0:
            self.A[self.t] = self.pi_eps(s)
        return self.A[self.t % (self.n + 1)]

    def train(self, s, a, r, sp, done=False):
        t, n = self.t, self.n
        if t == 0:  # We are in the initial state. Reset buffer.
            self.S[0], self.A[0] = s, a
        behavior_policy = self.get_policy(s, epsilon=0.3)
        target_policy = self.get_policy(s, epsilon=self.epsilon)

        self.rho[t % (n + 1)] = target_policy['probs'][s % (n + 1)] / behavior_policy['probs'][s % (n + 1)]
        self.A[t % (n + 1)] = behavior_policy['action']
        self.R[(t + 1) % (n + 1)] = r
        self.S[(t + 1) % (n + 1)] = sp
        self.sigma[t % (n + 1)] = self.get_sigma(a)

        if done:
            T = t + 1
            tau_steps_to_train = range(t - n + 1, T)
        else:
            T = 1e10
            tau_steps_to_train = [t - n + 1]
            ap = self.get_policy(sp, epsilon=0.3)['action']
            self.A[(t + 1) % (n + 1)] = ap
            self.sigma[(t + 1) % (n + 1)] = self.get_sigma(ap)
            self.rho[(t + 1) % (n + 1)] = target_policy['probs'][sp % (n + 1)] / behavior_policy['probs'][sp % (n + 1)]

        for tau in tau_steps_to_train:
            if tau >= 0:
                if t + 1 < T:
                    G = self.Q[self.S[(t + 1) % (n + 1)]][self.A[(t + 1) % (n + 1)]]
                for k in range(min(t + 1, T), tau + 1, -1):
                    k_idx = k % (n + 1)
                    if k == T:
                        G = self.R[T % (n + 1)]
                    else:
                        V = sum(self.pi(s % k_idx) * self.Q[s % k_idx])
                        d = (self.sigma[k_idx] * self.rho[k_idx] + (1 - self.sigma[k_idx]) * self.pi(
                            self.S[k_idx]))
                        G = self.R[k_idx] + self.gamma * d * (
                                G - self.Q[self.S[k_idx]][self.A[k_idx]]) + self.gamma * V

                S_tau, A_tau = self.S[tau % (n + 1)], self.A[tau % (n + 1)]
                delta = (G - self.Q[S_tau][A_tau])
                self.Q[S_tau][A_tau] += self.alpha * delta

        self.t += 1
        if done:
            self.t = 0

    def get_policy(self, s, epsilon):
        a = np.argmax(self.Q[s][:])
        pi_probs = np.ones(self.env.nA) * epsilon / self.env.nA
        pi_probs[a] += (1 - epsilon)
        return {'action': np.random.choice(range(self.env.nA), p=pi_probs), 'probs': pi_probs}

    def get_sigma(self, a):
        return np.random.randint(2, size=self.env.nA)[a]


if __name__ == "__main__":
    envn = 'CliffWalking-v0'
    env = gym.make(envn)
    agent = Qsigma(env, n=3, gamma=0.9, epsilon=0.1, alpha=0.5)
    exp = f"experiments/{envn}_{agent}"
    train(env, agent, exp, num_episodes=200, max_runs=5)
    main_plot(exp, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.show()
