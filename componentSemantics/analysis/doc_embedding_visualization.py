import glob
import os
import re

import igraph
import pandas as pd
import seaborn
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import seaborn as sns

matplotlib.matplotlib_fname()
import numpy

sns.set_theme(style="ticks")


def load_embeddings(path):
    embeddings = {}
    with open(path, "rt", encoding="utf8") as inf:
        for line in inf:
            splitLines = line.split()
            word = splitLines[0]
            embedding = numpy.array([float(value) for value in splitLines[1:]])
            embeddings[word] = embedding

    return embeddings


# import plotly.express as px


def plot_heatmap(maxtrix, project):
    mask = numpy.zeros_like(maxtrix)
    mask[numpy.triu_indices_from(mask)] = True

    ax = plt.axes()

    seaborn.heatmap(maxtrix, mask=mask, vmin=0.70, vmax=0.99)
    ax.set_title(f"{project} - Doc")
    plt.show()


# def plot_plotly(df):
#    fig = px.scatter(df, x='PC1', y='PC2', color='y', symbol="type")
#    fig.show()


def plot_seaborns(df):
    fig = seaborn.scatterplot(df["PC1"], df["PC2"], hue=df["y"], markers=df['type'])
    plt.show()


def visualize(embeddings, classes, shapes, size):
    points = PCA(n_components=2).fit_transform(embeddings)

    df = pd.DataFrame(points, columns=["PC1", "PC2"])
    df["y"] = classes
    df['type'] = shapes
    df["size"] = size
    plot_seaborns(df)


def split_embedding_community(path, embeddings, project):
    x = os.path.join(path, "**", "*.graphml")
    files = glob.glob(x, recursive=True)

    communities = []
    features = []
    shapes = []
    size = []
    means = []
    skipped = []
    avgs = []
    for file in files:
        i = str(re.search("comm_(\d+).graphml", file, re.IGNORECASE).group(1))
        doc_emb = []

        subcommunity = igraph.Graph.Read_GraphML(file)

        for node in subcommunity.vs:
            if "." == node["filePath"]:
                continue

            node_features = embeddings[node["filePath"]]
            doc_emb.append(node_features)
            features.append(node_features)
            communities.append(i)
            shapes.append("sample")
            size.append(15)

        if doc_emb:
            tot = 0
            n = 0
            similarities = sklearn.metrics.pairwise.cosine_similarity(numpy.array(doc_emb))
            iterate_indices = numpy.tril_indices(similarities.shape[0])
            for z, v in zip(*iterate_indices):
                tot += similarities[z, v]
                n += 1
            avg = tot / n

            print(avg)
            avgs.append(avg)

            similarities = sklearn.metrics.pairwise.cosine_similarity(numpy.array(doc_emb))
            plot_heatmap(similarities, f"{project} - Comm {i}")
            mean = numpy.array(doc_emb).astype(numpy.float).mean(axis=0).tolist()
            means.append(mean)
            features.append(mean)
            communities.append(i)
            shapes.append("mean")
            size.append(50)
        else:
            skipped.append(i)

    print("Avgs", numpy.mean(avgs), "Std", numpy.std(avgs))
    features = numpy.array(features).astype(numpy.float)
    return features, communities, shapes, size, means, skipped


def main():
    data = []
    for project in ["antlr4", "avro", "openj9"]:
        print("-" * 20)
        print(project)
        pdata = []
        method = "infomap"
        # project = "elasticsearch"
        path = f"../../data/embeddings/{project}.vec"
        embeddings = load_embeddings(path)
        graph = f"/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/data/graphs/{method}/raw/{project}/"
        embeddings, classes, shapes, size, means, skipped = split_embedding_community(graph, embeddings, project)

        visualize(embeddings, classes, shapes, size)

        similarities = sklearn.metrics.pairwise.cosine_similarity(numpy.array(means))
        # similarities = numpy.array(similarities)
        # norm_similarities = similarities / similarities.sum().sum()
        # similarities = norm_similarities
        # print(similarities)
        path = f"../../data/graphs/projects/{project}/comm_dependencies_{method}.csv"

        dependencies = numpy.loadtxt(path, dtype=int, delimiter=",")
        norm_dependencies = dependencies / dependencies.sum().sum()

        dependencies = norm_dependencies
        dependencies = numpy.delete(dependencies, skipped, axis=0)
        dependencies = numpy.delete(dependencies, skipped, axis=1)
        assert dependencies.shape == similarities.shape, print(dependencies.shape, similarities.shape)

        iterate_indices = numpy.tril_indices(dependencies.shape[0])

        sims = []
        plot_heatmap(similarities, project)
        col_skip = dependencies.any(axis=0)
        rows_skip = dependencies.any(axis=1)
        for z, x, c, r in zip(*iterate_indices, col_skip, rows_skip):
            if c and r:
                simil = similarities[z, x]
                sumx = dependencies[z, x] + dependencies[x, z]
                data.append((sumx, simil))
                pdata.append((sumx, simil))
                sims.append(similarities[z, x])

        print("Project", numpy.mean(sims), "Std", numpy.std(sims))


        df = pd.DataFrame(pdata, columns=["similarity", "dependency"])

        silhouette = sklearn.metrics.silhouette_score(embeddings, classes)

        print("Euclidean Silhouette", silhouette)

        similarities = sklearn.metrics.pairwise.cosine_similarity(numpy.array(embeddings))
        ones = numpy.ones_like(similarities)
        similarities = numpy.abs(ones - similarities)

        silhouette = sklearn.metrics.silhouette_score(similarities, classes, metric="precomputed")

        print("Similarity Silhouette", silhouette)

        for method in ["pearson"]:
            corr = df.corr(method=method)
            print(corr)

    df = pd.DataFrame(data, columns=["similarity", "dependency"])
    print("=" * 40)
    for method in ["pearson"]:
        corr = df.corr(method=method)
        print(corr)


if __name__ == '__main__':
    main()
