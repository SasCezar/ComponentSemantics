from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn import metrics

from utils import load_embeddings, check_dir


class IdentifiersTokens:
    def __init__(self, in_path, out_path, labels=None, visualize=False):
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
        self.labels = pd.DataFrame([{'label': k, "embedding": v} for k, v in labels.items()])
        check_dir(self.raw_out)

    def analyze(self, project, features_algorithm):
        terms, count = self.load_project(project, features_algorithm)

        similarities, weighted_similarities = self.get_similarities(terms, self.labels, count)

        return {
            'similarities': similarities,
            'weighted_similarities': weighted_similarities,
        }

    def load_project(self, project, features_algorithm):
        features_file = self.embeddings_path.format(features_algorithm=features_algorithm,
                                                    project=project)

        vectors = load_embeddings(features_file)
        a = [{'term': k, 'count': v[0]} for k, v in vectors.items()]
        b = [{'term': k, 'embedding': v[1:].tolist()} for k, v in vectors.items()]
        count = pd.DataFrame(a)
        embeddings = pd.DataFrame(b)

        return embeddings, count

    def get_similarities(self, terms, labels, count):
        terms_matrix = np.array(terms['embedding'].tolist())
        labels_matrix = np.array(labels['embedding'].tolist())
        similarities = metrics.pairwise.cosine_distances(terms_matrix, labels_matrix)
        weights = count['count'].to_numpy()

        weighted_similarity = similarities * weights[:, None]

        return similarities, weighted_similarity

