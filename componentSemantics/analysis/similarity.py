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
import numpy as np

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


def get_features_communities(path, embeddings):
    x = os.path.join(path, "**", "*.graphml")
    files = glob.glob(x, recursive=True)

    communities = []
    features = []
    skipped = []
    feature_shape = embeddings[list(embeddings.keys())[0]].shape
    for file in files:
        i = str(re.search("comm_(\d+).graphml", file, re.IGNORECASE).group(1))
        doc_emb = []

        subcommunity = igraph.Graph.Read_GraphML(file)

        for node in subcommunity.vs:
            if "." == node["filePath"]:
                continue

            node_features = numpy.array(embeddings[node["filePath"]]).astype(numpy.float)
            doc_emb.append(node_features)
            features.append(node_features)
            communities.append(i)

        if not doc_emb:
            skipped.append(i)
            # features.append(np.zeros(feature_shape))
            # communities.append(i)

    data = pd.DataFrame(zip(features, communities), columns=["features", "classes"])
    return data, skipped


def aggregate(data, method="mean"):
    if method == "mean":
        data = data.groupby('classes')["features"].apply(np.mean).reset_index(name='features')
    elif method == "sum":
        data = data.groupby('classes')["features"].apply(np.sum).reset_index(name='features')
    else:
        raise ValueError("Method of aggregation not defined.")
    data["classes"] = pd.to_numeric(data["classes"])
    return data.sort_values("classes").set_index("classes", drop=False)


def load_dependencies(path, skipped):
    dependencies = numpy.loadtxt(path, dtype=int, delimiter=",")
    norm_dependencies = dependencies / dependencies.sum().sum()

    dependencies = norm_dependencies
    dependencies = numpy.delete(dependencies, skipped, axis=0)
    dependencies = numpy.delete(dependencies, skipped, axis=1)
    return dependencies


def main(method, embedding):
    for project in ["antlr4", "avro", "openj9"]:
        embedding_path = f"../../data/embeddings/{embedding}/{project}.vec"
        embeddings = load_embeddings(embedding_path)
        graph = f"/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/data/graphs/{method}/raw/{project}/"

        features, skipped = get_features_communities(graph, embeddings)

        # visualize(embeddings, classes, shapes, size)

        aggregated_features = aggregate(features)
        similarities = sklearn.metrics.pairwise.cosine_similarity(
            numpy.array(aggregated_features["features"].tolist()))
        plot_heatmap(similarities, project)

        path = f"../../data/graphs/projects/{project}/comm_dependencies_{method}.csv"

        dependencies = load_dependencies(path, skipped)
        assert dependencies.shape == similarities.shape, print(dependencies.shape, similarities.shape)

        dep_sim, sims = get_depsim(dependencies, similarities)

        print("Project", numpy.mean(sims), "Std", numpy.std(sims))

        df = pd.DataFrame(dep_sim, columns=["similarity", "dependency"])

        cosine_distance = sklearn.metrics.pairwise.cosine_distances(aggregated_features["features"].tolist())

        silhouette = sklearn.metrics.silhouette_score(cosine_distance, aggregated_features['classes'].tolist(),
                                                      metric="precomputed")

        print("Similarity Silhouette", silhouette)

        corr = df.corr()
        print("Correlation", corr)


def get_depsim(dependencies, similarities):
    dep_sim = []
    sims = []
    iterate_indices = numpy.tril_indices(dependencies.shape[0])
    col_skip = dependencies.any(axis=0)
    rows_skip = dependencies.any(axis=1)
    for z, x, c, r in zip(*iterate_indices, col_skip, rows_skip):
        if c and r:
            simil = similarities[z, x]
            sumx = dependencies[z, x] + dependencies[x, z]
            dep_sim.append((sumx, simil))

            sims.append(similarities[z, x])
    return dep_sim, sims


if __name__ == '__main__':
    methods = ["infomap"]
    embeddings = ["document"]
    for method in methods:
        for embedding in embeddings:
            main(method, embedding)
