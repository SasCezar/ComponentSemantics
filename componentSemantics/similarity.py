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


def plot_heatmap(maxtrix, project):
    mask = numpy.zeros_like(maxtrix)
    mask[numpy.triu_indices_from(mask)] = True

    ax = plt.axes()

    seaborn.heatmap(maxtrix, mask=mask, vmin=0.70, vmax=0.99)
    ax.set_title(f"{project} - Doc")
    plt.show()


def plot_seaborns(df):
    fig = seaborn.scatterplot(data=df, x="PC1", y="PC2", hue="y")
    plt.show()


def visualize(embeddings, classes):
    points = PCA(n_components=2).fit_transform(embeddings)

    df = pd.DataFrame(points, columns=["PC1", "PC2"])
    df["y"] = classes
    plot_seaborns(df)


def community_features(path, embeddings, level):
    level = {"package": "name", "document": "filePath"}[level]

    x = os.path.join(path, "**", "*.graphml")
    files = glob.glob(x, recursive=True)

    communities = []
    features = []
    skipped = []

    for file in files:
        i = str(re.search("comm_(\d+).graphml", file, re.IGNORECASE).group(1))
        doc_emb = []

        subcommunity = igraph.Graph.Read_GraphML(file)

        for node in subcommunity.vs:
            if "." == node[level]:
                continue

            node_features = numpy.array(embeddings[node[level]]).astype(numpy.float)
            doc_emb.append(node_features)
            features.append(node_features)
            communities.append(i)

        if not doc_emb:
            skipped.append(i)

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
    skipped = np.array(skipped).astype(int)
    dependencies = norm_dependencies
    dependencies = numpy.delete(dependencies, skipped, axis=0)
    dependencies = numpy.delete(dependencies, skipped, axis=1)
    return dependencies


def communities_similarities(features):
    avgs = []
    sims = []
    for i, community in features.groupby("classes"):
        comm_feat = sklearn.metrics.pairwise.cosine_similarity(community["features"].tolist())
        iterate_indices = numpy.tril_indices(comm_feat.shape[0])
        plot_heatmap(comm_feat, "TEST")
        n = 0
        tot = 0
        for r, c in zip(*iterate_indices):
            tot += comm_feat[r, c]
            n += 1
            sims.append(comm_feat[r, c])

        mean = tot / n
        print("Community", i, "Mean Sim", mean)
        avgs.append(mean)

    print("SIMS AVG", numpy.mean(sims), "MEAN MEANS", numpy.mean(avgs))
    return numpy.mean(avgs)


def main(method, embedding):
    for project in ["antlr4", "avro", "openj9"]:
        embedding_path = f"../../data/embeddings/{embedding}/{project}.vec"
        embeddings = load_embeddings(embedding_path)
        graph = f"../../data/graphs/{method}/raw/{project}/"

        features, skipped = community_features(graph, embeddings, embedding)

        visualize(features['features'].tolist(), features["classes"].tolist())

        aggregated_features = aggregate(features)
        similarities = sklearn.metrics.pairwise.cosine_similarity(
            numpy.array(aggregated_features["features"].tolist()))
        plot_heatmap(similarities, project)

        path = f"../../data/graphs/projects/{project}/comm_dependencies_{method}.csv"

        dependencies = load_dependencies(path, skipped)
        assert dependencies.shape == similarities.shape, print(dependencies.shape, similarities.shape)

        dep_sim, sims = get_depsim(dependencies, similarities)

        communities_sim = communities_similarities(features)

        print("Project", numpy.mean(sims), "Std", numpy.std(sims))
        glob_sims = sklearn.metrics.pairwise.cosine_similarity(features["features"].tolist())
        glob_sims = glob_sims.reshape(-1)
        print("GLOBAL Project", numpy.mean(glob_sims), "Std", numpy.std(glob_sims))

        cosine_distance = sklearn.metrics.pairwise.cosine_distances(features["features"].tolist())

        silhouette = sklearn.metrics.silhouette_score(cosine_distance, features['classes'].tolist(),
                                                      metric="precomputed")

        print("Similarity Silhouette", silhouette)

        df = pd.DataFrame(dep_sim, columns=["similarity", "dependency"])
        corr = df.corr()
        print("Correlation", corr)


def get_depsim(dependencies, similarities):
    dep_sim = []
    sims = []
    iterate_indices = numpy.tril_indices(dependencies.shape[0])
    col_skip = dependencies.any(axis=0)
    rows_skip = dependencies.any(axis=1)
    for i, j, c, r in zip(*iterate_indices, col_skip, rows_skip):
        if c and r:
            simil = similarities[i, j]
            sumx = dependencies[i, j] + dependencies[j, i]
            dep_sim.append((sumx, simil))

            sims.append(similarities[i, j])
    return dep_sim, sims


if __name__ == '__main__':
    methods = ["leiden", "infomap"]
    embeddings = ["package", "document"]
    for method in methods:
        for embedding in embeddings:
            print("Processing", method, embedding)
            main(method, embedding)
