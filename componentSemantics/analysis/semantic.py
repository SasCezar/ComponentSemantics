import glob
import os
import re
from collections import defaultdict, Counter

import igraph
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn import metrics

from utils import load_embeddings, check_dir


class SemanticScores:
    def __init__(self, in_path, out_path, visualize=False):
        self.in_path = in_path
        self.out_path = out_path

        self.graph_path = self.in_path + "/graphs/{community_algorithm}/raw/{project}/"
        self.embeddings_path = self.in_path + "/embeddings/{features_algorithm}/{project}.vec"
        self.dependencies_path = self.in_path + \
                                 "/graphs/projects/{project}/comm_dependencies_weighted_{community_algorithm}.csv"

        self.level = defaultdict(lambda: "filePathReal")
        self.level["package"] = "name"
        self.visualize = visualize
        self.raw_out = f"{self.out_path}/plots/raw_data/"
        check_dir(self.raw_out)

    def analyze(self, project, community_algorithm, features_algorithm):
        project_data, skipped = self.load_project(project, community_algorithm, features_algorithm)
        cohesion = self.cohesion(project_data)
        inter_similarities = self.separation(project_data)

        mean_intersimilarity = np.mean(inter_similarities)
        std_intersimilarity = np.std(inter_similarities)

        dependencies_path = self.dependencies_path.format(project=project,
                                                          community_algorithm=community_algorithm)
        dependencies = self._load_dependencies(dependencies_path, skipped)

        dep_sim_corr = self.dependency_similarity_corr(dependencies, inter_similarities)
        silhouette = self.silhouette(project_data)

        if self.visualize:
            self.raw_plot(project_data, project, community_algorithm, features_algorithm)

        comm_size = [x[1] for x in Counter(project_data["classes"].tolist()).most_common()]
        result = {"cohesion": cohesion[0],
                  "inter_similarity": (mean_intersimilarity, std_intersimilarity)[0],
                  "dep_sim_corr": dep_sim_corr['similarity'][1],
                  "silhouette": silhouette, "comm_size": comm_size}

        return result

    def load_project(self, project, community_algorithm, features_algorithm):
        graph_folder = self.graph_path.format(community_algorithm=community_algorithm,
                                              project=project)

        graph_path = os.path.join(graph_folder, "**", "*.graphml")
        files = glob.glob(graph_path, recursive=True)

        features_file = self.embeddings_path.format(features_algorithm=features_algorithm,
                                                    project=project)

        embeddings = load_embeddings(features_file)
        if len(embeddings) < 30:
            raise Exception("Size of graph to small")
        communities = []
        features = []
        skipped = []

        for file in files:
            i = str(re.search("comm_(\d+).graphml", file, re.IGNORECASE).group(1))
            doc_emb = []

            subcommunity = igraph.Graph.Read_GraphML(file)

            for node in subcommunity.vs:
                name = node[self.level[features_algorithm]]
                if name == ".":
                    continue

                node_features = np.array(embeddings[name]).astype(np.float)
                doc_emb.append(node_features)
                features.append(node_features)
                communities.append(i)

            if not doc_emb or len(subcommunity.vs) < 4:
                skipped.append(i)

        project_data = pd.DataFrame(zip(features, communities), columns=["features", "classes"])

        return project_data, skipped

    def cohesion(self, data):
        avgs = []
        for i, community in data.groupby("classes"):
            comm_feat = metrics.pairwise.cosine_similarity(community["features"].tolist())
            iterate_indices = np.tril_indices(comm_feat.shape[0])
            tot = 0
            for r, c in zip(*iterate_indices):
                tot += comm_feat[r, c]

            mean = tot / len(iterate_indices[0])
            avgs.append(mean)

        return np.mean(avgs), np.std(avgs)

    def separation(self, data):
        aggregated_features = self._aggregate(data)
        intra_similarities = metrics.pairwise.cosine_similarity(np.array(aggregated_features["features"].tolist()))

        return intra_similarities

    @staticmethod
    def silhouette(data):
        cosine_distances = metrics.pairwise.cosine_distances(data["features"].tolist())

        silhouette = metrics.silhouette_score(cosine_distances,
                                              data['classes'].tolist(),
                                              metric="precomputed")

        return silhouette

    def dependency_similarity_corr(self, dependencies, similarities):
        dep_sim_df = self._align_dep_sim(dependencies, similarities)
        corr = dep_sim_df.corr(method="pearson")

        return corr

    @staticmethod
    def _load_dependencies(path, skipped):
        dependencies = np.loadtxt(path, dtype=int, delimiter=",")
        norm_dependencies = dependencies / dependencies.sum().sum()
        skipped = np.array(skipped).astype(int)
        dependencies = norm_dependencies
        dependencies = np.delete(dependencies, skipped, axis=0)
        dependencies = np.delete(dependencies, skipped, axis=1)

        return dependencies

    @staticmethod
    def _aggregate(data, method="mean"):
        """
        Creates a representation of a community given the features of all the members
        """
        if method == "mean":
            data = data.groupby('classes')["features"].apply(np.mean).reset_index(name='features')
        elif method == "sum":
            data = data.groupby('classes')["features"].apply(np.sum).reset_index(name='features')
        else:
            raise ValueError("Method of aggregation not defined.")

        data["classes"] = pd.to_numeric(data["classes"])

        return data.sort_values("classes").set_index("classes", drop=False)

    @staticmethod
    def _align_dep_sim(dependencies, similarities):
        deps = []
        sims = []

        iterate_indices = np.tril_indices(dependencies.shape[0])
        col_skip = np.invert(dependencies.any(axis=0))
        row_skip = np.invert(dependencies.any(axis=1))
        for i, j, in zip(*iterate_indices):
            if i == j and col_skip[i] and row_skip[j]:
                continue

            total_dependency = dependencies[i, j] + dependencies[j, i]
            deps.append(total_dependency)
            sims.append(similarities[i, j])

        df = pd.DataFrame(zip(sims, deps), columns=["similarity", "dependency"])

        return df

    def raw_plot(self, df, project, method, embedding):
        embeddings = df["features"].tolist()
        classes = df["classes"].tolist()
        cosine_distance = metrics.pairwise.cosine_distances(embeddings)
        tsne_points = TSNE(n_components=2, metric="precomputed").fit_transform(cosine_distance)

        for dim_technique, points in [("TSNE", tsne_points)]:
            df = pd.DataFrame(points, columns=["C1", "C2"])
            df["y"] = classes
            df.to_csv(os.path.join(self.raw_out, f"{dim_technique}_{project}_{method}_{embedding}.csv"),
                      encoding="utf8")
