"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import sys
import itertools
import numpy as np
from irlc import log_time_series
from tqdm import tqdm
from irlc.common import defaultdict2
from gym.envs.toy_text.discrete import DiscreteEnv
from irlc.irlc_plot import existing_runs
import warnings
from collections import OrderedDict
import os
import glob
import csv


class Agent():
    """
    Main agent class. See description of Ex09 for further details on how to use it with the environment, train and main_plot functions.
    """

    def __init__(self, env, gamma=0.99, epsilon=0):
        self.env, self.gamma, self.epsilon = env, gamma, epsilon
        """
        The self.Q variable is a custom datastructure to save the Q(s,a)-values during training. 
        There are multiple ways to implement the Q-values differently than here, most of which will us in hot water
        down the line. For instance, Q-values could be stored like a states x actions numpy table; this is simpler
        than what we have below, but it has the disadvantage it uses a lot of memory and that the states and actions
        has to be integers indexed from zero (i.e. to index self.Q[s,a] ). Another idea is to use nested dictionaries, i.e. 
        env.p[s] is a dictionary with keys a, this use less space, but it makes the max_a Q(s,a) operation difficult. 
        Finally we want the action-space to depend on s. 
        
        We solve this using a custom datastructure: A dictionary such that if we index self.Q[s] for an (unknown) s, 
        it calls the function we provide to defaultdict2, i.e. defaultdict2(myfun) and inserts that value in the dictionary:
        note this is an extension of the defaultdict-class (google to learn more). 
        
        >>> self.Q[s] = defaultdict2(myfun)
        >>> self.Q[s] = myfun(s) # when we index self.Q[s] where s is not in Q[s]
        """
        self.Q = defaultdict2(
            lambda s: np.zeros(len(env.P[s]) if hasattr(env, 'P') and s in env.P else env.action_space.n))

    def pi(self, s):
        """ Should return the Agent's action in state s (i.e. an element contained in env.action_space)"""
        raise NotImplementedError("return action")

    def train(self, s, a, r, sp, done=False):
        """ Called at each step of the simulation.
        The agent was in state s, took action a, ended up in state sp (with reward r).
        'done' is a bool which indicates if the environment terminated when transitioning to sp. """
        raise NotImplementedError()

    def __str__(self):
        """ A unique name for this agent. Used for plotting. """
        return super().__str__()

    def random_pi(self, s):
        """ Generates a random action given s.

        It might seem strange why this is useful, however many policies requires us to to random exploration, and it is
        possible to get the method wrong.
        We will implement the method depending on whether self.env defines an MDP or just contains an action space.
        """
        if isinstance(self.env, DiscreteEnv):
            return np.random.choice(list(self.env.P[s].keys()))
        else:
            return self.env.action_space.sample()

    def pi_eps(self, s):
        """ Implement epsilon-greedy exploration. Return random action with probability self.epsilon,
        else be greedy wrt. the Q-values. """
        return self.random_pi(s) if np.random.rand() < self.epsilon else np.argmax(
            self.Q[s] + np.random.rand(len(self.Q[s])) * 1e-8)

    def value(self, s):
        return np.max(self.Q[s])


"""
This is a simple wrapper class around the Agent class above. It fixes the policy and is therefore useful for doing 
value estimation.
"""


class ValueAgent(Agent):
    def __init__(self, env, gamma=0.95, policy=None, v_init_fun=None):
        self.env = env
        self.policy = policy  # policy to evaluate
        """ Value estimates. 
        Initially v[s] = 0 unless v_init_fun is given in which case v[s] = v_init_fun(s). """
        self.v = defaultdict2(float if v_init_fun is None else v_init_fun)
        super().__init__(env, gamma)

    def pi(self, s):
        return self.random_pi(s) if self.policy is None else self.policy(s)

    def value(self, s):
        return self.v[s]


def load_time_series(experiment_name, exclude_empty=True):
    """
    Load most recent non-empty time series (we load non-empty since lazylog creates a new dir immediately)
    """
    files = list(filter(os.path.isdir, glob.glob(experiment_name + "/*")))
    if exclude_empty:
        files = [f for f in files if
                 os.path.exists(os.path.join(f, "log.txt")) and os.stat(os.path.join(f, "log.txt")).st_size > 0]

    recent = sorted(files, key=lambda file: os.path.basename(file))[-1]
    stats = []
    with open(recent + '/log.txt', 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i == 0:
                head = row
            else:
                stats.append({k: float(v) for k, v in zip(head, row)})
    return stats, recent


def train(env, agent, experiment_name=None, num_episodes=None, verbose=True, reset=True, max_steps=1e10,
          max_runs=None, saveload_model=False, save_stats=True):
    if max_runs is not None and existing_runs(experiment_name) >= max_runs:
        return experiment_name, None, True
    stats = []
    steps = 0
    ep_start = 0
    if saveload_model:  # Code for loading/saving models
        did_load = agent.load(os.path.join(experiment_name))
        if did_load:
            stats, recent = load_time_series(experiment_name=experiment_name)
            ep_start, steps = stats[-1]['Episode'] + 1, stats[-1]['Steps']

    done = False
    with tqdm(total=num_episodes, disable=not verbose) as tq:
        for i_episode in range(num_episodes):
            s = env.reset() if reset else (env.s if hasattr(env, "s") else env.env.s)
            reward = []
            for _ in itertools.count():
                a = agent.pi(s)
                sp, r, done, _ = env.step(a)
                agent.train(s, a, r, sp, done)
                reward.append(r)
                steps += 1
                if done or steps > max_steps:
                    break
                s = sp

            stats.append({"Episode": i_episode + ep_start,
                          "Accumulated Reward": sum(reward),
                          "Average Reward": np.mean(reward),
                          "Length": len(reward),
                          "Steps": steps})
            tq.set_postfix(ordered_dict=OrderedDict(stats[-1]))
            tq.update()
    sys.stderr.flush()
    if saveload_model:
        agent.save(experiment_name)
        if did_load and save_stats:
            os.rename(recent + "/log.txt", recent + "/log2.txt")  # Shuffle old logs

    if experiment_name is not None and save_stats:
        log_time_series(experiment=experiment_name, list_obs=stats)
        print(f"Training completed. Logging: '{', '.join(stats[0].keys())}' to {experiment_name}")
    return experiment_name, stats, done
