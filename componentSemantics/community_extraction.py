import glob
import os

import igraph
import leidenalg
import numpy
from tqdm import tqdm

from feature_extraction import TfidfFeatureExtraction
from utils import check_dir, clean_graph


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
        graph = clean_graph(graph)
        graph_out = os.path.join(out_path, "graphs", "projects", project_name)
        check_dir(graph_out)

        for method in tqdm(self.algorithms, leave=False):
            method_out = os.path.join(out_path, "graphs", method, "raw", project_name)
            check_dir(method_out)
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
            source_community = graph.vs[edge.source][method]
            target_community = graph.vs[edge.target][method]
            if source_community != target_community:
                dependency[source_community, target_community] += 1
                connected.add(source_community)
                connected.add(target_community)

        return connected, dependency


def extract_communities(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    feature = TfidfFeatureExtraction()
    extractor = CommunityExtraction(feature)

    for project in tqdm(projects):
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))[0]
        extractor.extract(project, filepath, out_path)


if __name__ == '__main__':
    extract_communities("../data/arcanOutput/", "../data/")
