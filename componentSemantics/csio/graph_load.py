from abc import ABC, abstractmethod

import igraph
import numpy
import pandas


class GraphLoader(ABC):
    @abstractmethod
    def load(self, path):
        raise NotImplemented


class ArcanGraphLoader(GraphLoader):
    def __init__(self):
        self.clean_edges = ["isChildOf", "isImplementationOf", "nestedTo",
                            "belongsTo", "implementedBy", "definedBy",
                            "containerIsAfferentOf", "unitIsAfferentOf"]

    def load(self, path):
        graph = igraph.Graph.Read_GraphML(path)
        graph = self._clean_graph(graph)
        return graph

    def _clean_graph(self, graph):
        graph.es['weight'] = graph.es['Weight']
        delete = [x.index for x in graph.vs if "$" in x['name']]
        graph.delete_vertices(delete)
        for edge_label in self.clean_edges:
            graph.es.select(labelE=edge_label).delete()

        graph.vs.select(_degree=0).delete()

        return graph


class UnderstandGraphLoader(GraphLoader):
    def load(self, path):
        df = pandas.read_csv(path)
        graph = igraph.Graph.TupleList(df.itertuples(index=False), directed=True, weights=True)
        return graph
