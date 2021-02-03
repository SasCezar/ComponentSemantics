import os

from sklearn.metrics import adjusted_rand_score

from csio.graph_load import ArcanGraphLoader


def get_comm(graph):
    leiden = []
    infomap = []
    for i, node in enumerate(graph.vs):
        leiden.append(node['leiden'])
        infomap.append(node['infomap'])

    return leiden, infomap


def get_intersection(graphs_path):
    projects = [project for project in os.listdir(graphs_path)
                if os.path.isdir(os.path.join(graphs_path, project))]

    mean = []
    with open("rand_index.csv", "wt", encoding="utf8") as outf:
        outf.write(f"project, size, rand\n")
        for project in projects:
            try:
                graph_path = os.path.join(graphs_path, project, f"{project}.graphml")
                graph = ArcanGraphLoader().load(graph_path)
                leiden_comm, infomap_comm = get_comm(graph)

                rand = adjusted_rand_score(leiden_comm, infomap_comm)

                mean.append(rand)
                print(project, ",", rand)
                outf.write(f"{project}, {len(graph.vs)}, {rand}\n")
            except:
                continue

    print(sum(mean) / len(mean))


if __name__ == '__main__':
    graphs_path = "../data/graphs/projects"

    get_intersection(graphs_path)
