import os

import igraph
import seaborn
import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def plot_heatmap(matrix, project, method, embedding, out):
    mask = np.zeros_like(matrix)
    mask[np.triu_indices_from(mask)] = True
    seaborn.heatmap(matrix, mask=mask)
    out = os.path.join(out, f"heatmap_{project}_{method}_{embedding}.pdf")
    plt.savefig(out)
    plt.clf()


def plot_seaborns(df, technique, project, method, embedding, out):
    colors = {}
    classes = set(df['y'].tolist())

    for i, color in zip(classes, igraph.drawing.colors.ClusterColoringPalette(len(classes))):
        colors[str(i)] = matplotlib.colors.to_hex(color)

    fig = seaborn.scatterplot(data=df, x="C1", y="C2", hue="y", palette=colors)
    if len(colors) > 12:
        fig.legend_.remove()

    out = os.path.join(out, f"{technique}_{project}_{method}_{embedding}.pdf")
    plt.savefig(out)
    plt.clf()
