import re
import numpy as np
import pandas as pd
from os.path import join, curdir
from os import listdir
from sklearn.linear_model import LogisticRegression
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import polyfit
from scipy import optimize as opt


params = {'backend': 'ps',
#           'text.latex.preamble': [r'\usepackage{gensymb}'],
          'axes.labelsize': 8, # fontsize for x and y labels (was 10)
          'axes.titlesize': 8,
          'font.size': 8, # was 10
          'legend.fontsize': 8, # was 10
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'font.family': 'serif',
}

matplotlib.rcParams.update(params)
plt.rcParams["font.family"] = "serif"
figsize=(4, 2.5)

filter_legend = ['default',
                    '1/2 filter size',
                    '1/4 filter size',
                    '1/8 filter size',
                    '1/16 filter size']

network_legend = [Line2D([0], [0], marker='o', label='VGG11',
                         markerfacecolor='black'),
                 Line2D([0], [0], marker='^',  label='VGG13',
                         markerfacecolor='black'),
                 Line2D([0], [0], marker='s', label='VGG16',
                         markerfacecolor='black'),
                 Line2D([0], [0], marker='p', label='VGG19',
                         markerfacecolor='black'),]

cmap = plt.get_cmap('viridis')
depth_legend  = [Line2D([0], [0], color=cmap(0.0)),
                Line2D([0], [0], color=cmap(0.25)),
                Line2D([0], [0], color=cmap(0.5)),
                Line2D([0], [0], color=cmap(0.75)),
                Line2D([0], [0], color=cmap(1.0))]

legend_prop = None #{'size': 11}
filter_legend_loc = 4

def get_all_csv_files():
    files = listdir(curdir)
    csv_files = [file for file in files  if file.endswith('.csv') ]
    return csv_files

csv_files = get_all_csv_files()
csv_files = [file for file in csv_files if '_VGG' in file and not 'nbn' in file and not 'cd' in file and not 'BIG' in file and not '100' in file]# or 'frog' in file)]

regex_ft = 'train_features\d'
regex_clf = 'train_classifier\d'
regex_cnv = 'train_conv\d\d'
regex_fc = 'train_fc.*'


def filter_columns(columns):
    res = []
    highest_clf = higher_clf_0
    for col in columns:
        if re.match(regex_ft, col) or re.match(regex_cnv, col) or re.match(regex_fc):
            res.append(col)
        elif(re.match(regex_clf, col)):
            if int(highest-clf[-1]) < int(col[-1]):
                highest_clf = col
            res.append(col)
    col = col.remove(highest_clf)
    return col

def compute_average_saturation(epoch_df):
    cols = list(epoch_df.columns)
    val = 0
    c = 0
    for col in cols:
        if 'eval' in col or 'accuracy' in col or 'loss' in col or 'epoch' in col or 'time_per_step' in col or 'train_classifier6' in col:
            continue
        c += 1
        val += epoch_df[col].values
    return val / c


def get_final_epoch_accuracies(files):
    result = []
    for csv_file in files:

        res = {}
        file = pd.DataFrame.from_csv(csv_file, sep=';')
        try:

            divisor = 2 if 'catdog' in csv_file else 10
            divisor = 1

            last_epoch = file[file['epoch'] == 20]

            res['test_acc'] = last_epoch['test_accuracy'].values[0]
            res['test_loss'] = last_epoch['test_loss'].values[0] / divisor
            res['train_acc'] = last_epoch['train_accuracy'].values[0]
            res['train_loss'] = last_epoch['train_loss'].values[0] / divisor
            res['average_sat'] = compute_average_saturation(last_epoch)[0]
            res['name'] = csv_file.split('.csv')[0]
        except:
            continue
        result.append(res)
    return result

def get_all_epoch_data(files):
    result = []
    for csv_file in files:
        res = {}
        file = pd.DataFrame.from_csv(csv_file, sep=';')
        try:

            divisor = 2 if 'catdog' in csv_file else 10
            if '100' in csv_file:
                divisor = 100
            divisor = 1

            last_epoch = file

            res['test_acc'] = last_epoch['test_accuracy'].values
            res['test_loss'] = last_epoch['test_loss'].values / divisor
            res['train_acc'] = last_epoch['train_accuracy'].values
            res['train_loss'] = last_epoch['train_loss'].values
            res['average_sat'] = [compute_average_saturation(last_epoch[last_epoch['epoch']==i]) for i in range(20)]
            res['name'] = csv_file.split('.csv')
        except:
            continue
        result.append(res)
    return result

result = get_final_epoch_accuracies(csv_files)

all_test_acc = np.array([res['test_acc'] for res in result])
all_train_acc = np.array([res['train_acc'] for res in result])

all_test_loss = np.array([res['test_loss'] for res in result])
all_train_loss = np.array([res['train_loss'] for res in result])

all_average_sat = np.array([res['average_sat'] for res in result])*.01

all_names = np.array([res['name'] for res in result])

all_problem = np.array([name.split('_')[0] for name in all_names])
all_problem[all_problem == '10'] = 'CIFAR10'
all_problem[all_problem == '10imbalanced'] = 'Imb. CIFAR10'
all_problem[all_problem == 'catdog'] = 'cat-vs-dog'
all_problem[all_problem == '100'] = 'CIFAR100'

all_networks = [name.split('_')[1] for name in all_names if 'VGG' in name.split('_')[1]]
all_networks_filtered  = []
filter_sizes = []
depth = []
for i in range(len(all_names)):
    if 'VGG11_A' in all_names[i]:
        all_networks_filtered.append('VGG11')
        filter_sizes.append(filter_legend[0])
    elif 'VGG13_A' in all_names[i]:
        all_networks_filtered.append('VGG13')
        filter_sizes.append(filter_legend[0])
    elif 'VGG16_A' in all_names[i]:
        all_networks_filtered.append('VGG16')
        filter_sizes.append(filter_legend[0])
    elif 'VGG19_A' in all_names[i]:
        all_networks_filtered.append('VGG19')
        filter_sizes.append(filter_legend[0])
    elif 'VGG11_S' in all_names[i]:
        all_networks_filtered.append('VGG11')
        filter_sizes.append(filter_legend[1])
    elif 'VGG11_XS' in all_names[i]:
        all_networks_filtered.append('VGG11')
        filter_sizes.append(filter_legend[2])
    elif 'VGG11_XXS' in all_names[i]:
        all_networks_filtered.append('VGG11')
        filter_sizes.append(filter_legend[3])
    elif 'VGG11_XXXS' in all_names[i]:
        all_networks_filtered.append('VGG11')
        filter_sizes.append(filter_legend[4])
    elif 'VGG13_S' in all_names[i]:
        all_networks_filtered.append('VGG13')
        filter_sizes.append(filter_legend[1])
    elif 'VGG13_XS' in all_names[i]:
        all_networks_filtered.append('VGG13')
        filter_sizes.append(filter_legend[2])
    elif 'VGG13_XXS' in all_names[i]:
        all_networks_filtered.append('VGG13')
        filter_sizes.append(filter_legend[3])
    elif 'VGG13_XXXS' in all_names[i]:
        all_networks_filtered.append('VGG13')
        filter_sizes.append(filter_legend[4])
    elif 'VGG16_S' in all_names[i]:
        all_networks_filtered.append('VGG16')
        filter_sizes.append(filter_legend[1])
    elif 'VGG16_XS' in all_names[i]:
        all_networks_filtered.append('VGG16')
        filter_sizes.append(filter_legend[2])
    elif 'VGG16_XXS' in all_names[i]:
        all_networks_filtered.append('VGG16')
        filter_sizes.append(filter_legend[3])
    elif 'VGG16_XXXS' in all_names[i]:
        all_networks_filtered.append('VGG16')
        filter_sizes.append(filter_legend[4])
    elif 'VGG19_S' in all_names[i]:
        all_networks_filtered.append('VGG19')
        filter_sizes.append(filter_legend[1])
    elif 'VGG19_XS' in all_names[i]:
        all_networks_filtered.append('VGG19')
        filter_sizes.append(filter_legend[2])
    elif 'VGG19_XXS' in all_names[i]:
        all_networks_filtered.append('VGG19')
        filter_sizes.append(filter_legend[3])
    elif 'VGG19_XXXS' in all_names[i]:
        all_networks_filtered.append('VGG19')
        filter_sizes.append(filter_legend[4])
    else:
        all_networks_filtered.append('LEL')

all_networks_filtered = np.array(all_networks_filtered)
unique_networks_filtered = list(set(all_networks_filtered))
filter_sizes = np.array(filter_sizes)

unique_problem = np.array(list(set(all_problem)))
unique_problem = unique_problem[unique_problem != 'BIG10']

all_acc_gap = all_train_acc - all_test_acc
all_loss_gap = (all_test_loss - all_train_loss)

b10, m10, n10 = polyfit(all_average_sat[all_problem == 'CIFAR10'], all_test_acc[all_problem == 'CIFAR10'], 2)
b2, m2, n2 = polyfit(all_average_sat[all_problem == 'cat-vs-dog'], all_test_acc[all_problem == 'cat-vs-dog'], 2)

x = np.linspace(0, 1, 100)

colors = {
    filter_legend[0]: cmap(0.0),
    filter_legend[1] : cmap(0.25),
    filter_legend[2] : cmap(0.5),
    filter_legend[3]: cmap(0.75),
    filter_legend[4]: cmap(1.0)
}

network_depth = {
    'VGG11': 'o',
    'VGG13': '^',
    'VGG16': 's',
    'VGG19': 'p'
}

for classes in [2,10]:
    fig, ax = plt.subplots(figsize=figsize)

    if classes == 2:
        ax.plot(x, n2*(x**2)+m2*x+b2, c='grey')
    else:
        ax.plot(x, n10*(x**2)+m10*x+b10, c='grey')

    for i in range(len(all_average_sat)):
        if classes == 2 and 'CIFAR' in all_problem[i]:
                continue
        elif classes == 10 and 'cat-vs-dog' in all_problem[i]:
                continue
        ax.scatter(x=all_average_sat[i], y=all_test_acc[i], color=colors[filter_sizes[i]], marker=network_depth[all_networks_filtered[i]])

    l1 = plt.legend(network_legend, ['VGG11', 'VGG13', 'VGG16', 'VGG19'], loc=0, prop=legend_prop)
    ax.add_artist(l1)
    fig.subplots_adjust(right=0.75)
    plt.legend(depth_legend, filter_legend, loc=filter_legend_loc, prop=legend_prop)
    plt.xlim((0,1))
    plt.grid()

    plt.xlabel('Saturation')
    plt.ylabel('Test accuracy')
    plt.rcParams["font.family"] = "serif"
    plt.show()
    fig.savefig(
        f'generalization_test_acc_{classes}.eps',
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )

###

b10, m10, n10 = polyfit(all_average_sat[all_problem == 'CIFAR10'], all_test_loss[all_problem == 'CIFAR10'], 2)
b2, m2, n2 = polyfit(all_average_sat[all_problem == 'cat-vs-dog'], all_test_loss[all_problem == 'cat-vs-dog'], 2)

for classes in [2, 10]:
    fig, ax = plt.subplots(figsize=figsize)

    if classes == 2:
        ax.plot(x, n2*(x**2)+m2*x+b2, c='grey')
    else:
        ax.plot(x, n10*(x**2)+m10*x+b10, c='grey')

    for i in range(len(all_average_sat)):
        if classes == 2 and 'CIFAR' in all_problem[i]:
                continue
        elif classes == 10 and 'cat-vs-dog' in all_problem[i]:
                continue

        ax.scatter(x=all_average_sat[i], y=all_test_loss[i], color=colors[filter_sizes[i]], marker=network_depth[all_networks_filtered[i]])


    l1 = plt.legend(network_legend, ['VGG11', 'VGG13', 'VGG16', 'VGG19'], loc=2, prop=legend_prop)
    ax.add_artist(l1)
    plt.legend(depth_legend, filter_legend, loc=1 , prop=legend_prop)

    # plt.xlim((0,1))
    if classes == 2:
        plt.ylim((0,.015))
    else:
        plt.ylim((0, .02))
    plt.grid()
    plt.xlabel('Saturation')
    plt.ylabel('Test loss')
    fig.savefig(
        f"generalization_test_loss_{classes}.eps",
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )

###

b10, m10, n10 = polyfit(all_average_sat[all_problem == 'CIFAR10'], all_train_loss[all_problem == 'CIFAR10'], 2)
b2, m2, n2 = polyfit(all_average_sat[all_problem == 'cat-vs-dog'], all_train_loss[all_problem == 'cat-vs-dog'], 2)

for classes in [2, 10]:
    fig, ax = plt.subplots(figsize=figsize)

    if classes == 2:
        ax.plot(x, n2*(x**2)+m2*x+b2, c='grey')
    else:
        ax.plot(x, n10*(x**2)+m10*x+b10, c='grey')

    for i in range(len(all_average_sat)):
        if classes == 2 and 'CIFAR' in all_problem[i]:
                continue
        elif classes == 10 and 'cat-vs-dog' in all_problem[i]:
                continue
        ax.scatter(x=all_average_sat[i], y=all_train_loss[i], color=colors[filter_sizes[i]], marker=network_depth[all_networks_filtered[i]])

    l1 = plt.legend(network_legend, ['VGG11', 'VGG13', 'VGG16', 'VGG19'], loc=1, prop=legend_prop)
    ax.add_artist(l1)
    plt.legend(depth_legend, filter_legend, loc=2, prop=legend_prop)

    plt.xlim((0,1))
    plt.ylim((0,.01))
    plt.grid()
    plt.xlabel('Saturation')
    plt.ylabel('Train loss')
    fig.savefig(
        f"generalization_train_loss_{classes}.eps",
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )

plt.show()
