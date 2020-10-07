import glob
import os

import igraph
import leidenalg
import numpy
from tqdm import tqdm

from feature_extraction import DocumentFeatureExtraction


class CommunityExtraction:
    def __init__(self, features):
        self.algorithms = {
            "leiden": self.leiden,
            "infomap": self.infomap
        }

        self.features = features

        self._clean_edges = ["isChildOf", "isImplementationOf", "nestedTo", "belongsTo", "implementedBy", "definedBy"]

    def extract(self, project_name, graph_path, out_path):
        graph = igraph.Graph.Read_GraphML(graph_path)
        graph = self._clean(graph)
        graph_out = os.path.join(out_path, "graphs", "projects", project_name)
        self._check_dir(graph_out)

        features_out = os.path.join(out_path, "embeddings", self.features.level)
        features = self.features.get_embeddings(graph)
        self._check_dir(features_out)
        features_name = f"{project_name}.vec"
        self.features.save_features(features, features_out, features_name)

        for method in tqdm(self.algorithms, leave=False):
            method_out = os.path.join(out_path, "graphs", method, "raw", project_name)
            self._check_dir(method_out)
            communities = self.algorithms[method](graph)
            graph.vs[method] = communities.membership
            sub_communities = list(self._extract_sub(graph, method))

            connected_components, dependency = self.extract_community_dependency(graph, method)
            name = f"comm_dependencies_{method}.csv"
            numpy.savetxt(os.path.join(graph_out, name), dependency.astype(int), fmt='%i', delimiter=",")

            for i, community in enumerate(sub_communities):
                name = f"comm_{i}.graphml"
                self.save_graph(community, method_out, name)

        name = f"{project_name}.graphml"
        self.save_graph(graph, graph_out, name)

    @staticmethod
    def leiden(graph):
        return leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)

    @staticmethod
    def infomap(graph):
        return graph.community_infomap()

    @staticmethod
    def _extract_sub(graph, method):
        for i in sorted(set(graph.vs[method])):
            ids = [x.index for x in graph.vs if x[method] == i]
            comm = graph.subgraph(ids)
            yield comm

    def _clean(self, graph):
        delete = [x.index for x in graph.vs if "$" in x['name']]
        graph.delete_vertices(delete)
        for edge_label in self._clean_edges:
            graph.es.select(labelE=edge_label).delete()

        graph.vs.select(_degree=0).delete()

        return graph

    @staticmethod
    def _check_dir(path):
        project_path = os.path.join(path)
        if not os.path.exists(project_path):
            os.makedirs(project_path)

    @staticmethod
    def save_graph(graph, path, name):
        out = os.path.join(path, name)
        graph.write_graphml(out)

    @staticmethod
    def extract_community_dependency(graph, method):
        connected = set()
        n = len(set(graph.vs[method]))
        dependency = numpy.zeros((n, n))
        for edge in graph.es:
            source_vertex = graph.vs[edge.source]
            target_vertex = graph.vs[edge.target]
            source_community = source_vertex[method]
            target_community = target_vertex[method]
            if source_community != target_community:
                dependency[source_community, target_community] += 1
                connected.add(source_community)
                connected.add(target_community)

        return connected, dependency


def process(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    feature = DocumentFeatureExtraction()
    extractor = CommunityExtraction(feature)

    for project in tqdm(projects):
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))[0]
        extractor.extract(project, filepath, out_path)


if __name__ == '__main__':
    process("../data/arcanOutput/", "../../data/")
