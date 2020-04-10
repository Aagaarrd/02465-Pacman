"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import pandas as pd
import os
import numpy as np

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""


def plot_data(data, y="accumulated_reward", x="Episode", ci=95, estimator='mean', **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    if isinstance(data, list):  # is this correct even?
        data = pd.concat(data, ignore_index=True, axis=0)
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid", font_scale=1.5)
    lp = sns.lineplot(data=data, x=x, y=y, hue="Condition", ci=ci, estimator=estimator, **kwargs)
    plt.legend(loc='best')  # .set_draggable(True)


def existing_runs(experiment):
    nex = 0
    for root, dir, files in os.walk(experiment):
        if 'log.txt' in files:
            nex += 1
    return nex


def get_datasets(fpath, x, condition=None, smoothing_window=None, resample_key=None, resample_ticks=None):
    unit = 0
    if condition is None:
        condition = fpath

    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            json = os.path.join(root, 'params.json')
            if os.path.exists(json):
                with open(json) as f:
                    param_path = open(json)
                    params = json.load(param_path)
                    # exp_name = params['exp_name']

            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)
            # raise Exception("Group by ehre.0")
            if smoothing_window:
                ed_x = experiment_data[x]
                experiment_data = experiment_data.rolling(smoothing_window, min_periods=1).mean()
                experiment_data[x] = ed_x

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition)

            datasets.append(experiment_data)
            # print(experiment_data.columns)
            # if len(experiment_data.columns) > 7:
            #     a = 234
            unit += 1

    nc = f"({unit}x)" + condition[condition.rfind("/") + 1:]
    for i, d in enumerate(datasets):
        datasets[i] = d.assign(Condition=lambda x: nc)
        # d.rename(columns={'Condition': nc}, inplace=True)
        # gapminder.rename(columns={'pop': 'population',
        #                           'lifeExp': 'life_exp',
        #                           'gdpPercap': 'gdp_per_cap'},
        #                  inplace=True)

    if resample_key is not None:
        nmax = 0
        vmax = -np.inf
        vmin = np.inf
        for d in datasets:
            nmax = max(d.shape[0], nmax)
            vmax = max(d[resample_key].max(), vmax)
            vmin = min(d[resample_key].min(), vmin)
        if resample_ticks is not None:
            nmax = min(resample_ticks, nmax)

        new_datasets = []
        tnew = np.linspace(vmin + 1e-6, vmax - 1e-6, nmax)
        for d in datasets:
            nd = {}
            cols = d.columns.tolist()
            for c in cols:
                if c == resample_key:
                    y = tnew
                elif d[c].dtype == 'O':
                    # it is an object. cannot interpolate
                    y = [d[c][0]] * len(tnew)
                else:
                    y = np.interp(tnew, d[resample_key].tolist(), d[c], left=np.nan, right=np.nan)
                    y = y.astype(d[c].dtype)
                nd[c] = y

            ndata = pd.DataFrame(nd)
            ndata = ndata.dropna()
            new_datasets.append(ndata)
        datasets = new_datasets

    return datasets


def main_plot(experiments, legends=None, smoothing_window=10, resample_ticks=None,
              x_key="Episode",
              y_key='Accumulated Reward', **kwargs
              ):
    """
    Plot an experiment. To plot invidual lines (i.e. no averaging) use
    > units="Unit", estimator=None,

    """
    ensure_list = lambda x: x if isinstance(x, list) else [x]
    experiments = ensure_list(experiments)

    if legends is None:
        legends = experiments
    legends = ensure_list(legends)

    data = []
    for logdir, legend_title in zip(experiments, legends):
        resample_key = x_key if resample_ticks is not None else None
        data += get_datasets(logdir, x=x_key, condition=legend_title, smoothing_window=smoothing_window,
                             resample_key=resample_key, resample_ticks=resample_ticks)

    plot_data(data, y=y_key, x=x_key, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    # parser.add_argument('--save_pdf', action="store_true")
    # parser.add_argument('--pdf_dir', default="pdf")
    parser.add_argument('--title', default="please specify title", help="The title to show")
    parser.add_argument('--pdf_name', default=None, help="Name of pdf")

    args = parser.parse_args()
    main_plot(args.logdir, args.legend, args.value, title=args.title)


if __name__ == "__main__":
    main()
