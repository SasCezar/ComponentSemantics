import glob
import os
import shutil
from collections import Counter
from typing import Dict

import igraph
import matplotlib.pyplot as plt
import numpy
import seaborn
from tqdm import tqdm

from community.community import Leiden, Infomap, AbstractCommunityDetection
from csio.graph_load import ArcanGraphLoader
from utils import check_dir


class CommunityExtraction:
    def __init__(self, algorithms: Dict[str, AbstractCommunityDetection]):
        self.algorithms = algorithms
        self.visual_style = {"vertex_size": 15, "bbox": (1000, 1000), "margin": 20}

    def extract(self, project_name, graph_path, out_path):
        graph = ArcanGraphLoader().load(graph_path)
        graph_out = os.path.join(out_path, "graphs", "projects", project_name)
        check_dir(graph_out)

        plot_out = os.path.join(out_path, "plots", "graphs")
        check_dir(plot_out)
        print("Project", project_name, "# Nodes", len(graph.vs), "# Edges", len(graph.es))

        size_out = os.path.join(out_path, "plots", "size")
        check_dir(size_out)

        for method in self.algorithms:
            method_out = os.path.join(out_path, "graphs", method, "raw", project_name)
            check_dir(method_out)
            communities = self.algorithms[method].find_community(graph)
            graph.vs[method] = communities

            self.plot(graph, plot_out, f"{project_name}_{method}.pdf", method)

            connected_components, dependency, weighted_dependency = self.extract_community_dependency(graph, method)
            self.save_dependencies(dependency, graph_out, f"comm_dependencies_{method}.csv")
            self.save_dependencies(weighted_dependency, graph_out, f"comm_dependencies_weighted_{method}.csv")

            sub_communities = list(self._extract_sub(graph, method))
            for i, community in enumerate(sub_communities):
                name = f"comm_{i}.graphml"
                self.save_graph(community, method_out, name)

            print("Project", project_name, "Method", method, "# Comm", len(set(communities)))

            self.plot_size_distribution(communities, size_out, f"{project_name}_{method}.pdf")

        name = f"{project_name}.graphml"
        self.save_graph(graph, graph_out, name)

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
    def save_dependencies(dependencies, path, name):
        numpy.savetxt(os.path.join(path, name), dependencies.astype(int), fmt='%i', delimiter=",")

    @staticmethod
    def extract_community_dependency(graph, method):
        connected = set()
        n = len(set(graph.vs[method]))
        dependency = numpy.zeros((n, n))
        weighted_dependency = numpy.zeros((n, n))

        for edge in graph.es:
            source_community = graph.vs[edge.source][method]
            target_community = graph.vs[edge.target][method]
            if source_community != target_community:
                dependency[source_community, target_community] += 1
                weighted_dependency[source_community, target_community] += graph.es[edge.index]["weight"]
                connected.add(source_community)
                connected.add(target_community)

        return connected, dependency, weighted_dependency

    def plot(self, graph, path, name, method):
        pal = igraph.drawing.colors.ClusterColoringPalette(len(set(graph.vs[method])))
        graph.vs['color'] = pal.get_many(graph.vs[method])
        layout = graph.layout_fruchterman_reingold()
        self.visual_style["layout"] = layout
        self.visual_style["vertex_label"] = graph.vs[method]
        igraph.plot(graph, os.path.join(path, name), **self.visual_style)
        del (graph.vs['color'])

    @staticmethod
    def plot_size_distribution(communities, graph_out, name):
        counts = Counter(communities)
        sizes = [x[1] for x in counts.most_common(100)]
        print(sizes)

        seaborn.regplot(x=list(range(len(sizes))),
                        y=[x[1] for x in counts.most_common(100)],
                        scatter_kws={"s": 80},
                        order=3, ci=None)

        plt.savefig(os.path.join(graph_out, name))
        plt.clf()


def extract_communities(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    algorithms = {
        "leiden": Leiden(),
        "infomap": Infomap()
    }

    extractor = CommunityExtraction(algorithms)

    for project in tqdm(projects):
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))[0]
        extractor.extract(project, filepath, out_path)


if __name__ == '__main__':
    shutil.rmtree("../data/graphs")
    extract_communities("../data/arcanOutput/", "../data/")
