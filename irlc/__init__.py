import shutil
import inspect
import os
import compress_pickle
import numpy as np


def is_o_mode():
    return False


def bmatrix(a):
    if is_o_mode():
        return a.__str__()
    else:
        """Returns a LaTeX bmatrix
        :a: numpy array
        :returns: LaTeX bmatrix as a string
        """
        if len(a.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        lines = str(a).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv += [r'\end{bmatrix}']
        return '\n'.join(rv)


# from utils.lazylog import LazyLog

def log_time_series(experiment, list_obs, max_xticks_to_log=None, run_name=None):
    logdir = f"{experiment}/"

    if max_xticks_to_log is not None and len(list_obs) > max_xticks_to_log:
        I = np.round(np.linspace(0, len(list_obs) - 1, max_xticks_to_log))
        list_obs = [o for i, o in enumerate(list_obs) if i in I.astype(np.int).tolist()]

    with LazyLog(logdir) as logz:
        for n, l in enumerate(list_obs):
            for k, v in l.items():
                logz.log_tabular(k, v)
            if "Steps" not in l:
                logz.log_tabular("Steps", n)
            if "Episode" not in l:
                logz.log_tabular("Episode", n)
            logz.dump_tabular(verbose=False)


# from utils.irlc_plot import main_plot as main_plot


# def cn_(file_name):
#     return "cache/"+file_name

def is_this_my_computer():
    CDIR = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    return os.path.exists(CDIR + "/../../Exercises")


def cache_write(object, file_name, only_on_professors_computer=False):
    if only_on_professors_computer and not is_this_my_computer():
        """ Probably for your own good :-). """
        return
    # file_name = cn_(file_name) if cache_prefix else file_name
    dn = os.path.dirname(file_name)
    if not os.path.exists(dn):
        os.mkdir(dn)
    print("Writing cache...", file_name)
    with open(file_name, 'wb', ) as f:
        compress_pickle.dump(object, f, compression="lzma")
    print("Done!")


def cache_exists(file_name, cache_prefix=True):
    # file_name = cn_(file_name) if cache_prefix else file_name
    return os.path.exists(file_name)


def cache_read(file_name, cache_prefix=True):
    # file_name = cn_(file_name) if cache_prefix else file_name
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return compress_pickle.load(f, compression="lzma")
            # return pickle.load(f)
    else:
        return None
