import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns
import sklearn
from sklearn.manifold import TSNE

matplotlib.matplotlib_fname()
import numpy

sns.set_theme(style="ticks")


def load_embeddings(path, project):
    x = os.path.join(path, "**", "*.vec")
    files = glob.glob(x, recursive=True)

    communities = []
    features = []
    shapes = []
    size = []
    means = []
    avgs = []
    for file in files:
        i = str(re.search("comm_(\d+).vec", file, re.IGNORECASE).group(1))
        doc_emb = []

        with open(file, "rt", encoding="utf8") as inf:
            lines = inf.readlines()
            # if len(lines) < 5:
            #    continue
            for line in lines:
                node_features = line.split(" ")[1:]
                doc_emb.append(node_features)
                features.append(node_features)
                communities.append(i)
                shapes.append("sample")
                size.append(15)

        similarities = sklearn.metrics.pairwise.cosine_similarity(numpy.array(doc_emb))
        #plot_heatmap(similarities, f"{project} - Comm {i}")
        iterate_indices = numpy.tril_indices(similarities.shape[0])

        tot = 0
        n = 0
        for z, v in zip(*iterate_indices):
            tot += similarities[z, v]
            n += 1
        avg = tot / n

        print(avg)
        avgs.append(avg)

        mean = numpy.array(doc_emb).astype(numpy.float).mean(axis=0).tolist()
        means.append(mean)
        features.append(mean)
        communities.append(i)
        shapes.append("mean")
        size.append(50)

    print("Avgs", numpy.mean(avgs), "Std", numpy.std(avgs))
    features = numpy.array(features).astype(numpy.float)
    return features, communities, shapes, size, means


# import plotly.express as px


def plot_heatmap(maxtrix, project):
    mask = numpy.zeros_like(maxtrix)
    mask[numpy.triu_indices_from(mask)] = True

    ax = plt.axes()

    seaborn.heatmap(maxtrix, mask=mask, vmin=0.70, vmax=0.99)
    ax.set_title(f"{project} - Package")
    plt.show()


# def plot_plotly(df):
#    fig = px.scatter(df, x='PC1', y='PC2', color='y', symbol="type")
#    fig.show()

def plot_seaborns(df, project, method):

    fig = seaborn.scatterplot(df["PC1"], df["PC2"], hue=df["y"],
                              markers=df['type'],
                              palette=sns.color_palette("bright", len(set(df["y"])))).set_title(f"{project} - {method}")
    plt.legend([], [], frameon=False)
    plt.show()


def visualize(embeddings, classes, shapes, size, project, method):
    points = TSNE(n_components=2).fit_transform(embeddings)

    df = pd.DataFrame(points, columns=["PC1", "PC2"])
    df["y"] = classes
    df['type'] = shapes
    df["size"] = size
    plot_seaborns(df, project, method)


def main():
    data = []
    for project in ["antlr4", "avro", "openj9"]:
        print("-" * 20)
        print(project)
        pdata = []
        method = "infomap"
        # project = "elasticsearch"
        path = f"//data/graphs/{method}/raw/{project}"
        embeddings, classes, shapes, size, means = load_embeddings(path, project)

        visualize(embeddings, classes, shapes, size, project, method)

        similarities = sklearn.metrics.pairwise.cosine_similarity(numpy.array(means))

        # similarities = numpy.array(similarities)
        # norm_similarities = similarities / similarities.sum().sum()
        # similarities = norm_similarities
        # print(similarities)
        path = f"//data/graphs/projects/{project}/comm_dependencies_{method}.csv"

        dependencies = numpy.loadtxt(path, dtype=int, delimiter=",")
        norm_dependencies = dependencies / dependencies.sum().sum()

        dependencies = norm_dependencies
        assert dependencies.shape == similarities.shape, print(dependencies.shape, similarities.shape)

        iterate_indices = numpy.tril_indices(dependencies.shape[0])

        #plot_heatmap(similarities, project)
        col_skip = dependencies.any(axis=0)
        rows_skip = dependencies.any(axis=1)

        sims = []

        for i, j, c, r in zip(*iterate_indices, col_skip, rows_skip):
            if c and r:
                simil = similarities[i, j]
                sumx = dependencies[i, j] + dependencies[j, i]
                data.append((sumx, simil))
                pdata.append((sumx, simil))
                sims.append(similarities[i, j])

        print("Project", numpy.mean(sims), "Std", numpy.std(sims))
        df = pd.DataFrame(pdata, columns=["similarity", "dependency"])

        for method in ["pearson"]:
            corr = df.corr(method=method)
            print(corr)

        silhouette = sklearn.metrics.silhouette_score(embeddings, classes)

        print("Euclidean Silhouette", silhouette)

        similarities = sklearn.metrics.pairwise.cosine_distances(numpy.array(embeddings))


        silhouette = sklearn.metrics.silhouette_score(similarities, classes, metric="precomputed")

        print("Similarity Silhouette", silhouette)

    df = pd.DataFrame(data, columns=["similarity", "dependency"])
    print("=" * 40)
    for method in ["pearson"]:
        corr = df.corr(method=method)
        print(corr)


if __name__ == '__main__':
    main()
