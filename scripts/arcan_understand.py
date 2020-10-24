import igraph
import pandas

from infomap import Infomap

from componentSemantics import utils


def load_arcan(graph_path):
    graph = igraph.Graph.Read_GraphML(graph_path)
    graph = utils.clean_graph(graph)
    return graph


def load_understand(graph_path):
    df = pandas.read_csv(graph_path)
    graph = igraph.Graph.TupleList(df.itertuples(index=False), directed=True, weights=True)
    return graph


def get_arcan_nodes_edges(graph):
    nodes = set()
    edges = set()
    for edge in graph.es:
        source = graph.vs[edge.source]["name"]
        target = graph.vs[edge.target]["name"]

        nodes.add(source)
        nodes.add(target)
        edges.add((source, target))

    return nodes, edges


def get_understand_edges(graph):
    nodes = set()
    edges = set()
    for edge in graph.es:
        source = graph.vs[edge.source]["name"]
        target = graph.vs[edge.target]["name"]

        nodes.add(source)
        nodes.add(target)
        edges.add((source, target))

    return nodes, edges


if __name__ == '__main__':
    arcan_graph_path = "resources/dep-graph-1-2a4d846e6e1cf3e5e8576e0be9a9698c97b9f606.graphml"
    understand_graph_path = "resources/dependencies_weights-avro.csv"

    arcan_graph = load_arcan(arcan_graph_path)
    understand_graph = load_understand(understand_graph_path)

    arcan_nodes, arcan_edges = get_arcan_nodes_edges(arcan_graph)
    understand_nodes, understand_edges = get_understand_edges(understand_graph)

    nodes_intersection = arcan_nodes.intersection(understand_nodes)
    print(len(nodes_intersection), len(arcan_nodes))

    edges_intersection = arcan_edges.intersection(understand_edges)
    print(len(edges_intersection), len(arcan_edges))

    nodes_intersection = understand_nodes.intersection(arcan_nodes)
    print(len(nodes_intersection), len(understand_nodes))

    edges_intersection = understand_edges.intersection(arcan_edges)
    print(len(edges_intersection), len(understand_edges))

    comm = understand_graph.community_infomap(edge_weights="weight", trials=1)
    print(comm.membership)
    print(len(set(comm.membership)))

    # Command line flags can be added as a string to Infomap
    im = Infomap("--two-level --directed")

    # Add weight as optional third argument
    for edge in arcan_graph.es:
        # print(edge.source)
        im.add_link(edge.source, edge.target, edge["weight"])
    # Run the Infomap search algorithm to find optimal modules
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
