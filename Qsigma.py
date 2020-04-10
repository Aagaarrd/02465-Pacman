from irlc.agent import Agent, train
import numpy as np

class Qsigma(Agent):
    def __init__(self, env, gamma, alpha, epsilon, n):
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.nS, self.nA = env.nS, env.nA
        self.Q = np.random.random((env.nS, env.nA))

        self.stored = {v: {} for v in ['actions', 'states','deltas','Qs', 'bp', 'sigmas','rhos']}
        super().__init__(env, gamma, epsilon)


    def pi(self, s):
        return np.argmax(self.Q[s, :])


    def train(self, s, a, r, sp, done=False):
        T, tau, n = np.inf, 0, self.n
        self.stored['actions'][0] = self.get_bp(s)[0]
        self.stored['states'][0] = s
        self.stored['Qs'][0] = self.Q[s, self.stored['actions'][0]]
        self.stored['bp'][0] = self.get_bp(s)[1]
        self.stored['sigmas'][0] = self.select_sigma(s)
        self.stored['rhos'][0] = # \pi(A_t+1 | S_t+1) / \mu(S_t+1|A_t+1)

        for t in range(T+self.n-1):
            if t < T:
                sp, R, done, _ = self.env.step(a)
                self.stored['states'][t+1] = sp
                if done:
                    T = t+1
                else:
                    A_tp = np.random.choice(self.behaviour_pi[s])
                    sigma_tp = self.select_sigma()
            if tau >= 0:
                #rho = 0
                E = 1
                G = self.Q[self.stored['Qs'][tau % (n+1)]]

                for k in range(tau, min(tau+n-1, T-1)):
                    G += self.stored['deltas'][k%n]
                    E = self.gamma * (1 - self.stored['sigma'][k%n]) + self.stored['bp'][(k+1)%n] + self.stored['sigmas'][(k+1)%n]
                    rho *= (1 - self.stored['sigmas'][k%n] + self.stored['sigmas'])*self.stored['rhos'][k%n]

                a_tau, s_tau = self.stored['actions'][tau], self.stored['states'][tau]
                self.Q[a_tau, s_tau] += self.alpha * rho * (G - Q[a_tau, s_tau])


    def get_bp(self, s):
        a = np.argmax(self.Q[s][:])
        bp_probs = np.ones(self.nA)*self.epsilon/self.nA
        bp_probs[a] += (1 - self.epsilon)

        return np.random.choice(range(self.nA), p = bp_probs), bp_probs
    
    def get_tp(self, s):
        a = np.argmax(self.Q[s][:])
        tp_probs = np.ones(self.nA)*self.epsilon/self.nA
        tp_probs[a] += (1 - self.epsilon)

        return np.random.choice(range(self.nA), p = bp_probs), bp_probs
    
    
    def get_sigma(s):
        pass

if __name__ == "__main__":
    pass