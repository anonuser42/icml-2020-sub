from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import join, exists


def extract_layer_saturation(df, excluded="classifier-7", epoch=20):
    epoch= 1 if epoch == 0 else epoch
    cols = list(df.columns)
    train_cols = [col for col in cols if "train-" in col and not excluded in col]
    epoch_df = df[df.index == (epoch)]
    epoch_df = epoch_df[train_cols]
    return epoch_df 


cfg = {
    "V": [64, "M"],
    "VS": [32, "M"],
    "W": [64, "M", 128, "M"],
    "WS": [32, "M", 64, "M"],
    "X": [64, "M", 128, "M", 256, "M"],
    "XS": [32, "M", 64, "M", 128, "M"],
    "Y": [64, "M", 128, "M", 256, "M", 512, "M"],
    "YS": [32, "M", 64, "M", 128, "M", 256, "M"],
    "YXS": [16, "M", 32, "M", 64, "M", 128, "M"],
    "Z": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "ZXS": [16, "M", 32, "M", 64, "M", 128, "M", 128, "M"],
    "ZXXS": [8, "M", 16, "M", 32, "M", 64, "M", 64, "M"],
    "ZXXXS": [4, "M", 8, "M", 16, "M", 32, "M", 32, "M"],
    "ZS": [32, "M", 64, "M", 128, "M", 256, "M", 256, "M"],
    "AS": [32, "M", 64, "M", 126, 126, "M", 256, 256, "M", 256, 256, "M"],
    "AXS": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M"],
    "AXXS": [8, "M", 16, "M", 32, 32, "M", 64, 64, "M", 64, 64, "M"],
    "AXXXS": [4, "M", 8, "M", 16, 16, "M", 32, 32, "M", 32, 32, "M"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "AL": [128, "M", 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 1024, "M"],
    "BS": [32, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"],
    "BXS": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M"],
    "BXXS": [8, 8, "M", 16, 16, "M", 32, 32, "M", 64, 64, "M", 64, 64, "M"],
    "BXXXS": [4, 4, "M", 8, 8, "M", 16, 16, "M", 32, 32, "M", 32, 32, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "DS": [
        32,
        32,
        "M",
        64,
        64,
        "M",
        128,
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        256,
        256,
        256,
        "M",
    ],
    "DXXXS": [4, 4, "M", 8, 8, "M", 16, 16, 16, "M", 32, 32, 32, "M", 32, 32, 32, "M"],
    "DXXS": [8, 8, "M", 16, 16, "M", 32, 32, 32, "M", 64, 64, 64, "M", 64, 64, 64, "M"],
    "DXS": [
        16,
        16,
        "M",
        32,
        32,
        "M",
        64,
        64,
        64,
        "M",
        128,
        128,
        128,
        "M",
        128,
        128,
        128,
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "DL": [
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        1024,
        1024,
        1024,
        "M",
        1024,
        1024,
        1024,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
    "ES": [
        32,
        32,
        "M",
        64,
        64,
        "M",
        128,
        128,
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        256,
        256,
        256,
        256,
        "M",
    ],
    "EXS": [
        16,
        16,
        "M",
        32,
        32,
        "M",
        64,
        64,
        64,
        64,
        "M",
        128,
        128,
        128,
        128,
        "M",
        128,
        128,
        128,
        128,
        "M",
    ],
    "EXXS": [
        8,
        8,
        "M",
        16,
        16,
        "M",
        32,
        32,
        32,
        32,
        "M",
        64,
        64,
        64,
        64,
        "M",
        64,
        64,
        64,
        64,
        "M",
    ],
    "EXXXS": [
        4,
        4,
        "M",
        8,
        8,
        "M",
        16,
        16,
        16,
        16,
        "M",
        32,
        32,
        32,
        32,
        "M",
        32,
        32,
        32,
        32,
        "M",
    ],
}
from os.path import join

model_lookup = {
    "10_VGG13_S_2.csv": cfg["BS"],
    "10_VGG11_XXXXS_0.csv": cfg["AXXXS"],
    "10_VGG19_A2.csv": cfg["E"],
    "10_VGG19_XXXXS_1.csv": cfg["EXXXS"],
}

model_title = {
    r"VGG11_bs128_e50_t10000_idhistorics.csv": "VGG11",
    r"VGG11_XXS_bs128_e50_t10000_idhistorics.csv": "VGG11 (sparse)",
    r"VGG19_bs128_e50_t10000_idhistorics.csv": "VGG19",
    r"VGG19_XXS_bs128_e50_t10000_idhistorics.csv": "VGG19 (sparse)",
}

# ============== Main plot ================

import os
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import (
    MaxNLocator,
    MultipleLocator,
    FormatStrFormatter,
    AutoMinorLocator,
)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

# latexify(columns=2)


def plot_progress_saturation(deep, title, ax, file, scale_x, outer, plot_ind):
    num_epochs = 50
    top = []
    for i in range(num_epochs):
        deep_sats = extract_layer_saturation(deep, epoch=i)
        deep_sat_vals = deep_sats.filter(regex="features|classifier").values
        top.append(deep_sat_vals.squeeze())

    num_layers = len(top[0])
    test_acc = np.repeat(deep.test_accuracy[:num_epochs], num_layers)
    _x = np.arange(num_layers)
    _y = np.arange(num_epochs)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = np.array(top).flatten() / 100
    bottom = np.zeros_like(top)
    width = depth = 1
    Jet = plt.get_cmap("jet")
    colors = Jet(test_acc)

    cax = ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)
    plt.title(model_title[os.path.basename(file)])

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    counter = {'conv': 0, 'fc': 0}

    def replace(layer_name):
        if "features" in layer_name:
            counter['conv'] += 1
            return f"Conv-{counter['conv'] - 1}"
        elif "classifier" in layer_name:
            counter['fc'] += 1
            return f"Linear-{counter['fc']-1}"
        else:
            print(x, layer_name)
            raise ("Not implemented")

    labels = [
        x for x in deep_sats.columns if x.split("_")[-1][:5] in ("class", "featu")
    ]
    labels = [replace(x) for x in labels]

    ax.set_xticks(np.arange(len(labels)), minor=True)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(rotation=45, labelsize=5, pad=-5)  # rotation=70
    ax.yaxis.set_tick_params(pad=-4, size=15, labelsize=6)  # rotation=70
    ax.zaxis.set_tick_params(pad=-0.8, size=15, labelsize=6)  # rotation=70
    ax.set_xlabel("layer")
    ax.set_ylabel("epoch")
    if True:  # `outer`
        ax.zaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_zlabel("saturation", labelpad=0.1)
    else:
        ax.zaxis.set_major_formatter(plt.NullFormatter())

    ax.set_zlim(0, 1.0001)

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, 1, 0.8, 1]))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.xaxis.set_minor_formatter(FormatStrFormatter("%s"))


params = {
    "backend": "ps",
    "axes.labelsize": 8,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 8,
    "font.size": 20,  # was 10
    "legend.fontsize": 8,  # was 10
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
}

matplotlib.rcParams.update(params)

figsize = (8.1, 7)
fig = plt.figure(figsize=figsize)

# runs = sorted(glob.glob("10_VGG*_*.csv"))
runs = (
    ".\\logs\\VGG11\\Cifar10\\VGG11_bs128_e50_t10000_idhistorics.csv",
    ".\\logs\\VGG11_XXS\\Cifar10\\VGG11_XXS_bs128_e50_t10000_idhistorics.csv",
    ".\\logs\\VGG19\\Cifar10\\VGG19_bs128_e50_t10000_idhistorics.csv",
    ".\\logs\\VGG19_XXS\\Cifar10\\VGG19_XXS_bs128_e50_t10000_idhistorics.csv",
)

cols = 2
rows = round(len(runs) / cols)

max_x = max([pd.read_csv(file, sep=";").shape[1] for file in runs])

for col, file in enumerate(runs):
    r, c = divmod(col, cols)
    deep = pd.read_csv(file, sep=";")
    ax = fig.add_subplot(rows, cols, col + 1, projection="3d")
    fig.subplots_adjust(hspace=0, wspace=0)
    ax.view_init(elev=15)
    scale_x = deep.shape[1] / max_x

    outer = c == (cols - 1)
    sm = plot_progress_saturation(deep, file, ax, file, scale_x, outer, col)
    ax.annotate(
        "saturation",
        xy=(2, 1),
        xytext=(3, 1.5),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

ax.text2D(0.05, 0.95, " ", transform=plt.gca().transAxes)
fig.subplots_adjust(hspace=0.2, wspace=0.1)

Jet = plt.get_cmap("jet")
sm = ScalarMappable(cmap=Jet, norm=plt.Normalize(0, 1.0))
sm.set_array([])

cbar = plt.colorbar(sm, ax=fig.axes, pad=0.1, shrink=0.5, orientation="horizontal")

cbar.ax.tick_params(bottom=False)
cbar.ax.set_ylabel("Test accuracy", rotation=0)
cbar.ax.yaxis.set_label_coords(-0.2, 0.2)  # x, y
cbar.outline.set_visible(False)
#plt.tight_layout()
fig.tight_layout(pad=0, h_pad=0, w_pad=0)
fig.set_size_inches(8.1, 7)
plt.show()
#fig.savefig(
#    "sat_3dc.eps", format="eps", dpi=1000, bbox_inches="tight", pad_inches=0.01
#)
