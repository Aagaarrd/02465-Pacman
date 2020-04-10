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
        self.nS, self.nA = env.nS, env.nA
        self.Q = np.random.random((env.nS, env.nA))

        self.stored = {v: {i: 0 for i in range(1, n)} for v in ['actions', 'states','Qs','sigmas','rhos']}
        super().__init__(env, gamma, epsilon)


    def pi(self, s):
        return np.argmax(self.Q[s])


    def train(self, s, a, r, sp, done=False):
        T, t, tau, n = np.inf, -1, 0, self.n

        behavior_policy = self.get_policy(s, epsilon=0.3)
        target_policy = self.get_policy(s, epsilon=self.epsilon)

        self.stored['actions'][0] = behavior_policy['action']
        self.stored['states'][0] = s
        self.stored['Qs'][0] = self.Q[s, self.stored['actions'][0]]
        self.stored['sigmas'][0] = self.get_sigma(a)
        self.stored['rhos'][0] = target_policy['probs'][self.stored['actions'][0]] / behavior_policy['probs'][self.stored['actions'][0]]

        while tau < (T-1):
            t += 1
            if t < T:
                s_, _, done, _ = self.env.step(a)
                self.stored['states'][(t+1)%n] = s_
                if done:
                    T = t+1
                else:
                    a_ = self.get_policy(s, epsilon=0.3)['action']
                    self.stored['actions'][(t+1)%n] = a_
                    self.stored['sigmas'][(t+1)%n] = self.get_sigma(a_)
                    behavior_policy_, target_policy_ = self.get_policy(s_, epsilon=0.3), self.get_policy(s_, epsilon=self.epsilon)
                    self.stored['rhos'][(t+1)%n] = target_policy_['probs'] / behavior_policy_['probs']
            tau = t-n+1
            if tau >= 0:
                if t+1 < T:
                    G = self.stored['Qs'][(t+1)%n]
                for k in range(min(t+1, T), tau+1):
                    Ak = self.stored['actions'][k%n]
                    Sk, Rk, done, _ = self.env.step(Ak)
                    if k == T:
                        G = Rk
                    else:
                        V = sum(target_policy[Sk]['action']*self.stored['Qs'][Sk%n])
                        sigma_k, rho_k = self.stored['sigmas'][k%n], self.stored['rhos'][k%n]
                        G = Rk + self.gamma*(sigma_k*rho_k + (1-sigma_k)*target_policy[Sk]['action'])*(G-self.stored[Sk, Ak])+self.gamma*V
                s_tau, a_tau = self.stored['states'][tau%n], self.stored['actions'][tau%n]
                self.Q[s_tau, a_tau] += self.alpha*(G - self.Q[s_tau, a_tau])


    def get_policy(self, s, epsilon):
        a = np.argmax(self.Q[s][:])
        pi_probs = np.ones(self.nA)*epsilon/self.nA
        pi_probs[a] += (1 - epsilon)

        return {'action': np.random.choice(range(self.nA), p=pi_probs), 'probs': pi_probs}
    
    
    def get_sigma(self, a):
        return np.random.randint(2, size=self.nA)[a]

def cliffwalk():
    env = gym.make('CliffWalking-v0')
    exp = f"experiments/cliffwalk_Q"
    agent = Qsigma(env, gamma=0.9, epsilon=0.1, alpha=0.5, n=20)
    train(env, agent, exp, num_episodes=200, max_runs=5)
    return env, exp


if __name__ == "__main__":
    env, exp_name = cliffwalk()
    main_plot(exp_name, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q-learning on " + env.spec._env_name)
    # savepdf("Q_learning_cliff")
    plt.show()