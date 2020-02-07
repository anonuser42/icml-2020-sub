from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import join, exists
import matplotlib

params = {'backend': 'ps',
#           'text.latex.preamble': [r'\usepackage{gensymb}'],
          'axes.labelsize': 10, # fontsize for x and y labels (was 10)
          'axes.titlesize': 14,
          'font.size': 10, # was 10
          'legend.fontsize': 10, # was 10
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'font.family': 'serif',
}

matplotlib.rcParams.update(params)

shallow = pd.read_csv('10_VGG19_XXXS_1.csv', sep=';')
deep = pd.read_csv('10_VGG19_A2.csv', sep=';')
medium = pd.read_csv('10_VGG13_S_2.csv', sep=';')

def extract_layer_saturation(df, excluded = 'classifier6', epoch=20):
    cols = list(df.columns)
    train_cols = [col for col in cols if 'train' in col and not excluded in col and not 'accuracy' in col and not 'loss' in col]
    epoch_df = df[df['epoch'] == epoch]
    epoch_df = epoch_df[train_cols]
    return epoch_df

shallow_sats = extract_layer_saturation(shallow)
deep_sats = extract_layer_saturation(deep)
medium_sats = extract_layer_saturation(medium)

def plot_saturation_level(df, title, name):
    cols = list(df.columns)
    col_names = ['conv{}'.format(i+1) for i in range(len(df.columns))]
    col_names[-1] = "fc1"
    plt.figure(figsize=(7,5))
    plt.grid()
    plt.bar(list(range(len(cols))), df.values[0]/100)
    plt.xticks(list(range(len(cols))), col_names, rotation=60, fontsize=12)
    plt.ylim((0,1))
    plt.yticks(fontsize=16)
    plt.xlabel('layer', fontsize=16)
    plt.title(title, fontsize=16)
    plt.ylabel('saturation', rotation='vertical', fontsize=16)
    plt.savefig(
        f"sat_{name}.eps", format="eps", dpi=1000, bbox_inches="tight", pad_inches=0.01
    )

def plot_saturation_level_ax(df, ax):
    cols = list(df.columns)
    col_names = [i+1 for i in range(len(df.columns))]
    ax.grid()
    ax.bar(list(range(len(cols))), df.values[0])
    ax.set_xticks([])
    ax.set_ylim((0,100))

plot_saturation_level(shallow_sats, '60.57% Test Accuracy', name='shallow')
plot_saturation_level(deep_sats, '87.52% Test Accuracy', name='deep')
plot_saturation_level(medium_sats, '88.74% Test Accuracy', name='medium')
