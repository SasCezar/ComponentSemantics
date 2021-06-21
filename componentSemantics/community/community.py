import random
from abc import ABC, abstractmethod

import igraph
import leidenalg
import numpy
from igraph import Graph
import infomap

igraph.set_random_number_generator(random)
random.seed(1337)
numpy.random.seed(1337)


class AbstractCommunityDetection(ABC):
    @abstractmethod
    def find_community(self, graph: Graph, weights="weight"):
        raise not NotImplemented


class Leiden(AbstractCommunityDetection):
    def find_community(self, graph: Graph, weights="weight"):
        return leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition,
                                        weights=weights,
                                        seed=1337).membership


class Infomap(AbstractCommunityDetection):
    def __init__(self, params="--directed --silent --seed 1337 --num-trials 10"):
        self.params = params

    def find_community(self, graph: Graph, weights="weight"):
        im = infomap.Infomap(self.params)

        for edge in graph.es:
            im.add_link(edge.source, edge.target, edge[weights])

        im.run()

        unsorted_communitites = [
            (node.node_id, node.module_id) for node in im.tree if node.is_leaf
        ]

        min_id = min(x[1] for x in unsorted_communitites)
        return [
            x[1] - min_id
            for x in sorted(unsorted_communitites, key=lambda x: x[0])
        ]
