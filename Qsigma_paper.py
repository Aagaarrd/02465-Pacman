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
        self.R, self.S, self.A, self.rho, self.sigma, self.delta = [None] * (self.n + 1), [None] * (self.n + 1), [None] * (
                self.n + 1), [None] * (self.n + 1), [None] * (self.n + 1), [None] * (self.n + 1)
        self.behavior_policy = lambda s: self.get_policy(s, epsilon=0.3)
        self.target_policy = lambda s: self.get_policy(s, epsilon=self.epsilon)
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
        self.rho[t % (n + 1)] = self.target_policy(s)['probs'][a] / self.behavior_policy(s)['probs'][a]
        self.A[t % (n + 1)] = self.behavior_policy(s)['action']
        self.R[t % (n + 1)] = r
        self.S[(t + 1) % (n + 1)] = sp
        self.sigma[t % (n + 1)] = self.get_sigma(a)

        if done:
            T = t + 1
            self.delta[t % (n+1)] = r - self.Q[self.S[t % (n+1)]][self.A[t % (n+1)]]
        else:
            T = 1e10
            ap = self.behavior_policy(sp)['action']
            self.A[(t + 1) % (n + 1)] = ap
            Qp = self.Q[sp % (n+1)][ap % (n+1)]
            self.Q[self.S[(t + 1) % (n + 1)]][self.A[(t + 1) % (n + 1)]] = Qp
            sigmap = self.get_sigma(ap)
            self.sigma[(t + 1) % (n + 1)] = sigmap
            V = np.dot(self.behavior_policy(sp)['probs'], self.Q[sp])
            self.delta[t % (n + 1)] = r + self.gamma*(sigmap*Qp+(1-sigmap)*V) - self.Q[s % (n + 1)][a % (n + 1)]
            self.rho[(t + 1) % (n + 1)] = self.target_policy(sp)['probs'][sp % (n + 1)] / self.behavior_policy(sp)['probs'][sp % (n + 1)]

        tau = t - n + 1
        if tau >= 0:
            rho, E = 1, 1
            S_tau, A_tau = self.S[tau % (n + 1)], self.A[tau % (n + 1)]
            G = self.Q[S_tau][A_tau]
            for k in range(tau, min(tau+n-1, T-1)):
                k_id = k % (n + 1)
                G += E*self.delta[k_id]
                E = self.gamma*E*((1-self.sigma[k_id])*self.target_policy(self.S[(k + 1) % (n + 1)])['probs'][self.A[(k + 1) % (n + 1)]]+self.sigma[(k + 1) % (n + 1)])
                rho *= (1 - self.sigma[k_id] + self.sigma[k_id]*self.rho[k_id])

            delta = rho*(G-self.Q[S_tau][A_tau])
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
