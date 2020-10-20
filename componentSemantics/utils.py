import os


def check_dir(path):
    project_path = os.path.join(path)
    if not os.path.exists(project_path):
        os.makedirs(project_path)


clean_edges = ["isChildOf", "isImplementationOf", "nestedTo", "belongsTo", "implementedBy", "definedBy",
               "containerIsAfferentOf", "unitIsAfferentOf"]


def clean_graph(graph):
    graph.es['weight'] = graph.es['Weight']
    delete = [x.index for x in graph.vs if "$" in x['name']]
    graph.delete_vertices(delete)
    for edge_label in clean_edges:
        graph.es.select(labelE=edge_label).delete()

    graph.vs.select(_degree=0).delete()

    return graph
