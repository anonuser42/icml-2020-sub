import pandas as pd
from os.path import join, curdir
from os import listdir
from sklearn.linear_model import LogisticRegression
import matplotlib

params = {'backend': 'ps',
#           'text.latex.preamble': [r'\usepackage{gensymb}'],
          'axes.labelsize': 8, # fontsize for x and y labels (was 10)
          'axes.titlesize': 8,
          'font.size': 8, # was 10
          'legend.fontsize': 6, # was 10
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'font.family': 'serif',
}

matplotlib.rcParams.update(params)

from matplotlib import pyplot as plt



def get_all_csv_files():
    files = listdir(curdir)
    csv_files = [file for file in files if file.endswith('.csv')]
    return csv_files

csv_files = get_all_csv_files()
csv_files = [file for file in csv_files if ('_dense' in file and not '100' in file)]# or 'frog' in file)]
print(csv_files)

import re

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
        elif (re.match(regex_clf, col)):
            if int(highest - clf[-1]) < int(col[-1]):
                highest_clf = col
            res.append(col)
    col = col.remove(highest_clf)
    return col


def compute_average_saturation(epoch_df):
    cols = list(epoch_df.columns)
    val = 0
    # print(cols)
    c = 0
    for col in cols:
        if 'eval' in col or 'accuracy' in col or 'loss' in col or 'epoch' in col or 'time_per_step' in col or not '1' in col:
            continue
        c += 1
        print(col)
        val += epoch_df[col].values
    return val / c


def get_final_epoch_accuracies(files):
    result = []
    for csv_file in files:
        print(csv_file)
        res = {}
        file = pd.DataFrame.from_csv(csv_file, sep=';')
        try:

            divisor = 2 if 'catdog' in csv_file else 10
            divisor = 1

            if 'catdog' in csv_file.split('.csv')[0]:
                last_epoch = file[file['epoch'] == 20]
            else:
                last_epoch = file[file['epoch'] == 7]

            res['test_acc'] = last_epoch['test_accuracy'].values[0]
            res['test_loss'] = last_epoch['test_loss'].values[0] / divisor
            res['train_acc'] = last_epoch['train_accuracy'].values[0]
            res['train_loss'] = last_epoch['train_loss'].values[0] / divisor
            res['average_sat'] = compute_average_saturation(last_epoch)[0]
            res['name'] = csv_file.split('.csv')[0]
        except:
            continue
        result.append(res)
        print()
        print()
    return result


def get_all_epoch_data(files):
    result = []
    for csv_file in files:
        print(csv_file)
        res = {}
        file = pd.DataFrame.from_csv(csv_file, sep=';')
        try:
            # if True:

            divisor = 2 if 'catdog' in csv_file else 10
            if '100' in csv_file:
                divisor = 100
            divisor = 1

            last_epoch = file

            res['test_acc'] = last_epoch['test_accuracy'].values
            res['test_loss'] = last_epoch['test_loss'].values / divisor
            res['train_acc'] = last_epoch['train_accuracy'].values
            res['train_loss'] = last_epoch['train_loss'].values
            res['average_sat'] = [compute_average_saturation(last_epoch[last_epoch['epoch'] == i]) for i in range(20)]
            res['name'] = csv_file.split('.csv')
        except:
            continue
        result.append(res)
        print()
        print()
    return result


plt.rcParams["font.family"] = "serif"
figsize=(4, 3)

result = get_final_epoch_accuracies(csv_files)
all_epoch_results = get_all_epoch_data(csv_files)

from matplotlib import pyplot as plt
import numpy as np
print(all_epoch_results[0].keys())

fig, ax = plt.subplots(figsize=figsize)
#plt.title('Train loss given different number of units during training')
color = ['blue', 'red', 'yellow', 'green', 'violet']
for k, i in enumerate(['8', '16', '32', '64', '128']):
    for res in all_epoch_results:
       # try:
        if True:
            if res['name'][0].split('_')[-3] == i and res['name'][0].split('_')[0] == '10':
                print(res['name'])
                plt.plot(list(range(21)),np.array(res['train_loss']), label='{} units'.format(res['name'][0].split('_')[-3]), color=color[k])
                plt.grid()
                break

       # except:
       #     continue
plt.xticks(list(range(20)), np.arange(1,21))
plt.xlim((0,19))
#plt.ylim((0,100))
#plt.ylim((0.0105,0.0127))


plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Training Loss', fontsize=16)

plt.legend(prop={'size':11})
plt.show()
fig.savefig(
    f'train_layer_sat_per_epoch_20.eps',
    format="eps",
    dpi=1000,
    bbox_inches="tight",
)


fig, ax = plt.subplots(figsize=figsize)
#plt.title('Train loss given different number of units during training')
color = ['blue', 'red', 'yellow', 'green', 'violet']
for k, i in enumerate(['8', '16', '32', '64', '128']):
    for res in all_epoch_results:
       # try:
        if True:
            if res['name'][0].split('_')[-3] == i and res['name'][0].split('_')[0] == '10':
                print(res['name'])
                plt.plot(list(range(21)),np.array(res['test_loss']), label='{} units'.format(res['name'][0].split('_')[-3]), color=color[k])
                plt.grid()
                break

       # except:
       #     continue
plt.xticks(list(range(20)), np.arange(1,21))
plt.xlim((0,7))
#plt.ylim((0,100))
#plt.ylim((0.0105,0.0127))


plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Loss', fontsize=16)
plt.legend(prop={'size':11})
plt.show()
fig.savefig(
    f'val_loss_per_epoch_7.eps',
    format="eps",
    dpi=1000,
    bbox_inches="tight",
)


fig, ax = plt.subplots(figsize=figsize)
#plt.title('Train loss given different number of units during training')
color = ['blue', 'red', 'yellow', 'green', 'violet']
for k, i in enumerate(['8', '16', '32', '64', '128']):
    for res in all_epoch_results:
       # try:
        if True:
            if res['name'][0].split('_')[-3] == i and res['name'][0].split('_')[0] == '10':
                print(res['name'])
                plt.plot(list(range(21)),np.array(res['test_loss']), label='{} units'.format(res['name'][0].split('_')[-3]), color=color[k])
                plt.grid()
                break

       # except:
       #     continue
plt.xticks(list(range(21)), np.arange(0,21))
plt.xlim((1,20))
#plt.ylim((0,100))
#plt.ylim((0.0105,0.0127))


plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Loss', fontsize=16)
plt.legend(prop={'size':11})
plt.show()
fig.savefig(
    f'val_loss_per_epoch_20.eps',
    format="eps",
    dpi=1000,
    bbox_inches="tight",
)

fig, ax = plt.subplots(figsize=figsize)
#plt.title('Train loss given different number of units during training')
color = ['blue', 'red', 'yellow', 'green', 'violet']
for k, i in enumerate(['8', '16', '32', '64', '128']):
    for res in all_epoch_results:
       # try:
        if True:
            if res['name'][0].split('_')[-3] == i and res['name'][0].split('_')[0] == '10':
                print(res['name'])
                plt.plot(list(range(20)),np.array(res['average_sat'])/100, label='{} units'.format(res['name'][0].split('_')[-3]), color=color[k])
                plt.grid()
                break

       # except:
       #     continue
plt.xticks(list(range(1, 21)), np.arange(1,21))
plt.xlim((1,20))
plt.ylim((0,1))
#plt.ylim((0.0105,0.0127))


plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Saturation', fontsize=16)
plt.legend(prop={'size':11})
plt.show()
fig.savefig(
    f'layer_sat_per_epoch_20.eps',
    format="eps",
    dpi=1000,
    bbox_inches="tight",
)

fig, ax = plt.subplots(figsize=figsize)
#plt.title('Train loss given different number of units during training')
color = ['blue', 'red', 'yellow', 'green', 'violet']
for k, i in enumerate(['8', '16', '32', '64', '128']):
    for res in all_epoch_results:
       # try:
        if True:
            if res['name'][0].split('_')[-3] == i and res['name'][0].split('_')[0] == '10':
                print(res['name'])
                plt.plot(list(range(20)),np.array(res['average_sat'])/100, label='{} units'.format(res['name'][0].split('_')[-3]), color=color[k])
                plt.grid()
                break

       # except:
       #     continue
plt.xticks(list(range(0, 21)), np.arange(1,22))
plt.xlim((0,7))
plt.ylim((0,1))
#plt.ylim((0.0105,0.0127))


plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Saturation', fontsize=16)

plt.legend(prop={'size':11})
plt.show()
fig.savefig(
    f'layer_sat_per_epoch_7.eps',
    format="eps",
    dpi=1000,
    bbox_inches="tight",
)