# coding: utf-8
""" Plot TensorBoard summary data for paper. """

import re
import os
import json
import glob

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

sns.reset_orig()
sns.set_context("paper")

params = {
    "backend": "ps",
    "axes.labelsize": 8,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 8,
    "font.size": 8,  # was 10
    "legend.fontsize": 8,  # was 10
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
}

matplotlib.rcParams.update(params)
sns.set_style("white", params)

def set_size(fig):
    fig.set_size_inches(3.4, 2.2)
    plt.tight_layout()

def plot_summary(
    summaries, title="", ylabel="", y_lim=None, x_lim=None, window=1, name="default"
):
    fig, ax = plt.subplots(1)
    set_size(fig)

    for key in sorted(summaries.keys(), key=lambda x: float(x)):
        color = sns.color_palette()
        try:
            x, y = zip(*summaries[key])
        except Exception as e:
            print(f"Exception: {e}")
            continue
        # smooth curve
        x_ = []
        y_ = []

        for i in range(0, len(x), window):
            x_.append(x[i])
            y_.append(sum(y[i : i + window]) / float(window))
        base_line, = ax.plot(x_, y_, linewidth=2.0, label=int(key))
        ax.plot(x, y, alpha=0.3, color=base_line.get_color())
        if y_lim:
            plt.ylim(y_lim[0], y_lim[1])
        if x_lim:
            plt.xlim(x_lim[0], x_lim[1])
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.legend()

    plt.grid()
    plt.savefig(
        f"autoencoder_{name}.eps",
        format="eps",
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0.01,
    )


def extract_events(key="loss"):
    vals = {}
    for z_dim in [8, 16, 32, 64, 128]:
        vals[z_dim] = []
    folders = glob.glob("data_compare_covariance/z*/batchsize*/samplesNone")
    for folder in folders:
        events = glob.glob(os.path.join(folder, "epochs600", "events.out.*"))
        for event in events:
            try:
                for e in tf.train.summary_iterator(event):
                    m = re.search("\/z\d*", event)
                    z_dim = int(m.group(0).split("z")[-1])
                    for value in e.summary.value:
                        if value.tag == key:
                            vals[z_dim].append((e.step, value.simple_value))
            except:
                continue
    return vals


# Get loss from history
filepath = "autoencoder_loss.json"
if os.path.exists(filepath):
    loss = json.load(open(filepath))
else:
    loss = extract_events("loss")
    # Save to file
    json.dump(loss, open(filepath, "w"))
plot_summary(
    loss,
    "Bottleneck layer size and loss",
    ylabel="Loss",
    y_lim=(0, 0.08),
    x_lim=(0, 6000),
    window=100,
    name="loss",
)

# Get explained variance from history
filepath = "autoencoder_explained_var.json"
if os.path.exists(filepath):
    explained_var = json.load(open(filepath))
else:
    explained_var = extract_events(key="explained_variance")
    json.dump(explained_var, open(filepath, "w"))
plot_summary(
    explained_var,
    "Bottleneck layer size and intrinsic dimensionality",
    ylabel="Eigenvalues needed to explain variance",
    x_lim=(0, 6000),
    # y_lim=(0, 0.075),
    # window=100,
    name="explained_var",
)

def plot_saturation(
    summaries, title="", x_lim=None, y_lim=None, window=10, name="default"
):
    sns.set_context("paper")
    fig, ax = plt.subplots()
    set_size(fig)

    for key in sorted(summaries.keys(), key=lambda x: int(x)):
        color = sns.color_palette()
        x, y = zip(*summaries[key])

        # smooth curve
        x_ = []
        y_ = []

        for i in range(0, len(x), window):
            x_.append(x[i])
            y_.append(sum(y[i : i + window]) / float(window))
        base_line, = ax.plot(x_, [y / int(key) for y in y_], linewidth=2.0, label=key)
        if y_lim:
            plt.ylim(y_lim[0], y_lim[1])
        if x_lim:
            plt.xlim(x_lim[0], x_lim[1])
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel("Bottleneck saturation")
        plt.legend()

    plt.grid()
    plt.savefig(
        f"autoencoder_{name}.eps",
        format="eps",
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0.01,
    )


plot_saturation(
    explained_var,
    "Bottleneck layer size and saturation",
    x_lim=(0, 6000),
    y_lim=(0, 1.0),
    # window=100,
    name="saturation",
)


# Get validation loss from history
filepath = "autoencoder_val_loss.json"
if os.path.exists(filepath):
    val_loss = json.load(open(filepath))
else:
    val_loss = extract_events("val_loss")
    # Save to file
    json.dump(val_loss, open(filepath, "w"))
plot_summary(
    val_loss,
    "Bottleneck layer size and validation loss",
    x_lim=(val_loss["8"][0][0], 6000),
    # y_lim=(0, 0.075),
    ylabel="Validation Loss",
    # window=100,
    name="val_loss",
)

plt.show()
