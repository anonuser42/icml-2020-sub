import pandas as pd
from os.path import join, curdir
from os import listdir
from sklearn.linear_model import LogisticRegression
import matplotlib
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit

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


def get_final_epoch_accuracies(files, last_epoch=7):
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

for last_epoch in [7]:

    result = get_final_epoch_accuracies(csv_files, last_epoch)
    all_epoch_results = get_all_epoch_data(csv_files)

    from matplotlib import pyplot as plt
    import numpy as np
    print(all_epoch_results[0].keys())

    all_test_acc = np.array([res['test_acc'] for res in result]) * 100
    all_train_acc = np.array([res['train_acc'] for res in result]) * 100

    all_test_loss = np.array([res['test_loss'] for res in result])
    all_train_loss = np.array([res['train_loss'] for res in result])

    all_average_sat = np.array([res['average_sat'] for res in result])
    print(all_average_sat)
    all_names = np.array([res['name'] for res in result])
    color = np.zeros(len(all_names))
    for i in range(len(color)):
        color[i] = int(all_names[i].split('_')[-3])
        print(color[i])

    problem = np.zeros(len(all_names))
    for i in range(len(all_names)):
        problem[i] = 10 if all_names[i].split('_')[0] == '10' else 2
        print(problem)

    all_acc_gap = all_train_acc - all_test_acc
    all_loss_gap = (all_test_loss - all_train_loss)

    b10, m10 = polyfit(all_average_sat[problem == 10], all_test_acc[problem == 10] / 100, 1)
    b2, m2 = polyfit(all_average_sat[problem == 2], all_test_acc[problem == 2] / 100, 1)
    fig = plt.figure(figsize=figsize)
    plt.grid()

    x = np.arange(0, 100, 1)
    plt.scatter(x=all_average_sat[problem == 10], y=all_test_acc[problem == 10] / 100, c=color[problem == 10])
    plt.scatter(x=all_average_sat[problem == 2], y=all_test_acc[problem == 2] / 100, c=color[problem == 2])

    plt.plot(x, m10 * x + b10, label=f'CIFAR10 Regression Line')
    plt.plot(x, m2 * x + b2, label=f'cat-vs-dog Regression Line')
    # plt.plot(x, )
    c_bar = plt.colorbar()
    c_bar.set_label('Number of units in second layer')
    # for i, xy in enumerate(zip(all_average_sat, all_test_acc)):
    #    plt.annotate(all_names[i], xy)

    plt.xticks([0, 20, 40, 60, 80, 100], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.ylim((40/100,70/100))
    plt.xlabel('Saturation')
    plt.ylabel('Test Accuracy')

    plt.legend()

    plt.show()

    fig.savefig(
        f'test_accuracy_dense_no_overfit.eps',
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )

    #########

    result = get_final_epoch_accuracies(csv_files, last_epoch)
    all_epoch_results = get_all_epoch_data(csv_files)

    from matplotlib import pyplot as plt
    import numpy as np
    print(all_epoch_results[0].keys())

    all_test_acc = np.array([res['test_acc'] for res in result]) * 100
    all_train_acc = np.array([res['train_acc'] for res in result]) * 100

    all_test_loss = np.array([res['test_loss'] for res in result])
    all_train_loss = np.array([res['train_loss'] for res in result])

    all_average_sat = np.array([res['average_sat'] for res in result])
    print(all_average_sat)
    all_names = np.array([res['name'] for res in result])
    color = np.zeros(len(all_names))
    for i in range(len(color)):
        color[i] = int(all_names[i].split('_')[-3])
        print(color[i])

    problem = np.zeros(len(all_names))
    for i in range(len(all_names)):
        problem[i] = 10 if all_names[i].split('_')[0] == '10' else 2
        print(problem)

    all_acc_gap = all_train_acc - all_test_acc
    all_loss_gap = (all_test_loss - all_train_loss)

    b10, m10 = polyfit(all_average_sat[problem == 10], all_test_loss[problem == 10], 1)
    b2, m2 = polyfit(all_average_sat[problem == 2], all_test_loss[problem == 2] , 1)
    fig = plt.figure(figsize=figsize)
    plt.grid()

    x = np.arange(0, 100, 1)
    plt.scatter(x=all_average_sat[problem == 10], y=all_test_loss[problem == 10], c=color[problem == 10])
    plt.scatter(x=all_average_sat[problem == 2], y=all_test_loss[problem == 2], c=color[problem == 2])

    plt.plot(x, m10 * x + b10, label=f'CIFAR10 Regression Line')
    plt.plot(x, m2 * x + b2, label=f'cat-vs-dog Regression Line')
    # plt.plot(x, )
    c_bar = plt.colorbar()
    c_bar.set_label('Number of units in second layer')
    # for i, xy in enumerate(zip(all_average_sat, all_test_acc)):
    #    plt.annotate(all_names[i], xy)

    plt.xticks([0, 20, 40, 60, 80, 100], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.ylim((0.0, 0.0150))
    plt.xlabel('Saturation')
    plt.ylabel('Test Loss')

    plt.legend()

    plt.show()

    fig.savefig(
        f'test_loss_dense_no_overfit.eps',
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )


    #########

    result = get_final_epoch_accuracies(csv_files, last_epoch)
    all_epoch_results = get_all_epoch_data(csv_files)

    from matplotlib import pyplot as plt
    import numpy as np
    print(all_epoch_results[0].keys())

    all_test_acc = np.array([res['test_acc'] for res in result]) * 100
    all_train_acc = np.array([res['train_acc'] for res in result]) * 100

    all_test_loss = np.array([res['test_loss'] for res in result])
    all_train_loss = np.array([res['train_loss'] for res in result])

    all_average_sat = np.array([res['average_sat'] for res in result])
    print(all_average_sat)
    all_names = np.array([res['name'] for res in result])
    color = np.zeros(len(all_names))
    for i in range(len(color)):
        color[i] = int(all_names[i].split('_')[-3])
        print(color[i])

    problem = np.zeros(len(all_names))
    for i in range(len(all_names)):
        problem[i] = 10 if all_names[i].split('_')[0] == '10' else 2
        print(problem)

    all_acc_gap = all_train_acc - all_test_acc
    all_loss_gap = (all_test_loss - all_train_loss)

    b10, m10 = polyfit(all_average_sat[problem == 10], all_train_loss[problem == 10], 1)
    b2, m2 = polyfit(all_average_sat[problem == 2], all_train_loss[problem == 2] , 1)
    fig = plt.figure(figsize=figsize)
    plt.grid()

    x = np.arange(0, 100, 1)
    plt.scatter(x=all_average_sat[problem == 10], y=all_train_loss[problem == 10], c=color[problem == 10])
    plt.scatter(x=all_average_sat[problem == 2], y=all_train_loss[problem == 2], c=color[problem == 2])

    plt.plot(x, m10 * x + b10, label=f'CIFAR10 Regression Line')
    plt.plot(x, m2 * x + b2, label=f'cat-vs-dog Regression Line')
    # plt.plot(x, )
    c_bar = plt.colorbar()
    c_bar.set_label('Number of units in second layer')
    # for i, xy in enumerate(zip(all_average_sat, all_test_acc)):
    #    plt.annotate(all_names[i], xy)

    plt.xticks([0, 20, 40, 60, 80, 100], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.ylim((0.0, 0.0150))
    plt.xlabel('Saturation')
    plt.ylabel('Train Loss')

    plt.legend()

    plt.show()

    fig.savefig(
        f'train_loss_dense_no_overfit.eps',
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )

    ########

def get_final_epoch_accuracies(files, last_epoch=7):
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
        print()
        print()
    return result


for last_epoch in [20]:

    result = get_final_epoch_accuracies(csv_files, last_epoch)
    all_epoch_results = get_all_epoch_data(csv_files)

    from matplotlib import pyplot as plt
    import numpy as np
    print(all_epoch_results[0].keys())

    all_test_acc = np.array([res['test_acc'] for res in result]) * 100
    all_train_acc = np.array([res['train_acc'] for res in result]) * 100

    all_test_loss = np.array([res['test_loss'] for res in result])
    all_train_loss = np.array([res['train_loss'] for res in result])

    all_average_sat = np.array([res['average_sat'] for res in result])
    print(all_average_sat)
    all_names = np.array([res['name'] for res in result])
    color = np.zeros(len(all_names))
    for i in range(len(color)):
        color[i] = int(all_names[i].split('_')[-3])
        print(color[i])

    problem = np.zeros(len(all_names))
    for i in range(len(all_names)):
        problem[i] = 10 if all_names[i].split('_')[0] == '10' else 2
        print(problem)

    all_acc_gap = all_train_acc - all_test_acc
    all_loss_gap = (all_test_loss - all_train_loss)

    b10, m10 = polyfit(all_average_sat[problem == 10], all_test_acc[problem == 10] / 100, 1)
    b2, m2 = polyfit(all_average_sat[problem == 2], all_test_acc[problem == 2] / 100, 1)
    fig = plt.figure(figsize=figsize)
    plt.grid()

    x = np.arange(0, 100, 1)
    plt.scatter(x=all_average_sat[problem == 10], y=all_test_acc[problem == 10] / 100, c=color[problem == 10])
    #plt.scatter(x=all_average_sat[problem == 2], y=all_test_acc[problem == 2] / 100, c=color[problem == 2])

    plt.plot(x, m10 * x + b10, label=f'CIFAR10 Regression Line')
    #plt.plot(x, m2 * x + b2, label=f'cat-vs-dog Regression Line')
    # plt.plot(x, )
    c_bar = plt.colorbar()
    c_bar.set_label('Number of units in second layer')
    # for i, xy in enumerate(zip(all_average_sat, all_test_acc)):
    #    plt.annotate(all_names[i], xy)

    plt.ylim((40/100,70/100))
    plt.xticks([0, 20, 40, 60, 80, 100], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.xlabel('Saturation Train')
    plt.ylabel('Test Accuracy')

    plt.legend()

    plt.show()

    fig.savefig(
        f'test_accuracy_dense_overfit.eps',
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )