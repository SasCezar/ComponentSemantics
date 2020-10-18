import glob
import os
from collections import Counter

import igraph
import leidenalg
import numpy
import seaborn

import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import check_dir, clean_graph


class CommunityExtraction:
    def __init__(self):
        self.algorithms = {
            "leiden": self.leiden,
            "infomap": self.infomap
        }

        self.visual_style = {"vertex_size": 15, "bbox": (1000, 1000), "margin": 20}

    def extract(self, project_name, graph_path, out_path):
        graph = igraph.Graph.Read_GraphML(graph_path)
        graph = clean_graph(graph)
        graph_out = os.path.join(out_path, "graphs", "projects", project_name)
        check_dir(graph_out)

        plot_out = os.path.join(out_path, "plots", "graphs")
        check_dir(plot_out)
        print("Project", project_name, "# Nodes", len(graph.vs), "# Edges", len(graph.es))
        for method in self.algorithms:
            method_out = os.path.join(out_path, "graphs", method, "raw", project_name)
            check_dir(method_out)
            communities = self.algorithms[method](graph)
            graph.vs[method] = communities.membership
            sub_communities = list(self._extract_sub(graph, method))

            pal = igraph.drawing.colors.ClusterColoringPalette(len(set(graph.vs[method])))
            graph.vs['color'] = pal.get_many(graph.vs[method])
            self.plot(graph, plot_out, f"{project_name}_{method}.pdf", method)
            connected_components, dependency = self.extract_community_dependency(graph, method)
            name = f"comm_dependencies_{method}.csv"
            numpy.savetxt(os.path.join(graph_out, name), dependency.astype(int), fmt='%i', delimiter=",")

            for i, community in enumerate(sub_communities):
                name = f"comm_{i}.graphml"
                self.save_graph(community, method_out, name)

            print("Project", project_name, "Method", method, "# Comm", len(set(communities.membership)))

            counts = Counter(communities.membership)
            ax = plt.axes()
            seaborn.regplot(x=list(range(len([x[1] for x in counts.most_common(100)]))),
                            y=[x[1] for x in counts.most_common(100)],
                            scatter_kws={"s": 80},
                            order=3, ci=None)
            ax.set_title(f"{project_name} - {method}")
            plt.show()

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

    def plot(self, graph, path, name, method):
        layout = graph.layout_fruchterman_reingold()
        self.visual_style["layout"] = layout
        self.visual_style["vertex_label"] = graph.vs[method]
        igraph.plot(graph, os.path.join(path, name), **self.visual_style)


def extract_communities(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    extractor = CommunityExtraction()

    for project in tqdm(projects):
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))[0]
        extractor.extract(project, filepath, out_path)


if __name__ == '__main__':
    extract_communities("../data/arcanOutput/", "../data_2/")
