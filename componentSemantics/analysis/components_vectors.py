import glob
import os
import re
from collections import defaultdict

import igraph
import numpy as np
import pandas as pd

from utils import load_embeddings, check_dir


class ComponentVectors:
    def __init__(self, in_path, out_path, visualize=False):
        self.in_path = in_path
        self.out_path = out_path

        self.graph_path = self.in_path + "/graphs/{community_algorithm}/raw/{project}/"
        self.embeddings_path = self.in_path + "/embeddings/{features_algorithm}/{project}.vec"
        self.dependencies_path = self.in_path + \
                                 "/graphs/projects/{project}/comm_dependencies_weighted_{community_algorithm}.csv"

        self.level = defaultdict(lambda: "filePathRelative")
        self.level["package"] = "name"
        self.visualize = visualize
        self.raw_out = f"{self.out_path}/plots/raw_data/"
        check_dir(self.raw_out)

    def analyze(self, project, community_algorithm, features_algorithm):
        project_data, skipped = self.load_project(project, community_algorithm, features_algorithm)

        comm_vectors, project_vector = self.get_community_vector(project_data)

        return comm_vectors, project_vector

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

    def get_community_vector(self, data):
        aggregated_features = self._aggregate_grouped(data)
        single_feature = self._aggregate(data)
        return aggregated_features, single_feature

    @staticmethod
    def _aggregate(data, method="mean"):
        """
        Creates a representation of a community given the features of all the members
        """
        if method == "mean":
            res = np.mean(np.array(data['features'].tolist()), axis=0)
        elif method == "sum":
            res = np.sum(np.array(data['features'].tolist()), axis=0)
        else:
            raise ValueError("Method of aggregation not defined.")

        return res

    @staticmethod
    def _aggregate_grouped(data, method="mean"):
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

        return pd.DataFrame(zip(sims, deps), columns=["similarity", "dependency"])
