import glob
import os
import re

import igraph
import leidenalg
import numpy
import spacy
from tqdm import tqdm


class CommunityExtraction:
    def __init__(self):
        self.algorithms = {
            "leiden": self.leiden,
            "infomap": self.infomap
        }

        self.features = FeatureExtraction()

        self._clean_edges = ["isChildOf", "isImplementationOf", "nestedTo", "belongsTo", "implementedBy", "definedBy"]

    def extract(self, project_name, graph_path, out_path):
        graph = igraph.Graph.Read_GraphML(graph_path)
        graph = self._clean(graph)
        graph_out = os.path.join(out_path, "projects", project_name)
        self._check_dir(graph_out)

        for method in tqdm(self.algorithms, leave=False):
            method_out = os.path.join(out_path, method, "raw", project_name)
            self._check_dir(method_out)
            communities = self.algorithms[method](graph)
            graph.vs[method] = communities.membership
            sub_communities = list(self._extract_sub(graph, method))

            connected_components, dependency = self.extract_community_dependency(graph, method)
            name = f"comm_dependencies_{method}.csv"
            numpy.savetxt(os.path.join(graph_out, name), dependency.astype(int), fmt='%i', delimiter=",")

            for i, community in enumerate(sub_communities):
                # if i not in connected_components:
                #    continue
                name = f"comm_{i}.graphml"
                self.save_graph(community, method_out, name)
                features = self.features.get_embeddings(community)
                name = f"comm_{i}.vec"
                self.features.save_features(features, method_out, name)

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

        # delete = []
        # for i in range(n):
        #    if i not in connected:
        #        delete.append(i)

        # dependency = numpy.delete(dependency, delete, axis=0)
        # dependency = numpy.delete(dependency, delete, axis=1)

        return connected, dependency


class FeatureExtraction:
    def __init__(self):
        self.nlp = spacy.load("en_trf_bertbaseuncased_lg")

    def get_embeddings(self, subgraph):
        for node in tqdm(subgraph.vs, leave=False):
            name = node['name']
            name, clean = self.name_to_sentence(name)

            if not clean:
                clean = node['name']

            embedding = self.nlp(clean).vector
            yield name, clean, embedding

    @staticmethod
    def split_camel(name):
        splitted = re.sub('([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
        return splitted

    def name_to_sentence(self, name):
        tokens = name.split(".")[2:]
        clean = []

        for token in tokens:
            clean.extend(self.split_camel(token))

        return name, " ".join(clean).lower()

    @staticmethod
    def save_features(features, path, name):
        out = os.path.join(path, name)

        with open(out, "wt", encoding="utf8") as outf:
            for name, cleanded, embedding in features:
                rep = " ".join([str(x) for x in embedding.tolist()])
                line = name + " " + rep + "\n"
                outf.write(line)


def process(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    extractor = CommunityExtraction()

    for project in tqdm(projects):
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))[0]
        extractor.extract(project, filepath, out_path)


if __name__ == '__main__':
    process("../../data/arcanOutput/", "../../data/graphs")
