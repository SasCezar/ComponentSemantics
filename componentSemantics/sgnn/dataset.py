import glob
import os

import torch
from torch_geometric.data import Dataset, Data
from igraph import Graph
import numpy


class DependencyCommunityDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(DependencyCommunityDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return self.raw_file_names

    @property
    def raw_file_names(self):
        return glob.glob(os.path.join(self.raw_dir, "**", "*.graphml"), recursive=True)

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return self.processed_file_names

    @property
    def processed_file_names(self):
        return [x.replace(self.raw_dir, self.processed_dir).replace(".graphml", ".pt") for x in self.raw_paths]

    def process(self):
        for i, raw_path in enumerate(self.raw_paths):
            graph = Graph.Read_GraphML(raw_path)
            edge_index = self._get_edge_index(graph)
            feature_path = raw_path.replace(".graphml", ".vec")

            embeddings = self._load_node_features(feature_path)
            data = Data(x=embeddings, edge_index=edge_index)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            project = os.path.basename(os.path.dirname(raw_path))
            file = os.path.basename(os.path.basename(raw_path))

            project_processed = os.path.join(self.processed_dir, project)
            if not os.path.exists(project_processed):
                os.makedirs(project_processed)

            torch.save(data, os.path.join(self.processed_dir, project, file.replace(".graphml", ".pt")))

    @staticmethod
    def _load_node_features(path):
        features = []
        with open(path, "rt", encoding="utf8") as inf:
            for line in inf:
                node_features = line.split(" ")[1:]
                features.append(node_features)

        return torch.tensor(numpy.array(features, dtype=float), dtype=torch.float)

    @staticmethod
    def _get_edge_index(graph):
        source_vertices = []
        target_vertices = []
        for edge in graph.es:
            source_vertices.append(edge.source)
            target_vertices.append(edge.target)

        return torch.tensor([source_vertices, target_vertices], dtype=torch.long)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data
