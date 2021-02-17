import csv
import os
import sys
import traceback

import pandas

from analysis.components_vectors import ComponentVectors
from utils import check_dir

if __name__ == '__main__':
    in_path = "../data/"
    out_path = "../data/"
    methods = ["leiden", "infomap"]
    #embeddings = ["code2vec", "package", "document", "fastText", "TFIDF"]
    embeddings = ["code2vec-multi"]#, "package", "document", "fastText", "TFIDF"]
    analysis = ComponentVectors(in_path, out_path, visualize=False)
    df = pandas.DataFrame()
    embeddings_path = os.path.join(out_path, 'components_embeddings')
    check_dir(embeddings_path)
    arc_in = os.listdir(os.path.join(in_path, "arcanOutput"))
    projects = [project for project in arc_in
                if os.path.isdir(os.path.join(in_path, "arcanOutput", project))]

    skipped = 0
    counted = 0

    sizes = []
    nums = []
    seen = set()
    for method in methods:
        for embedding in embeddings:
            with open(os.path.join(embeddings_path, f'comm_{method}-{embedding}.vec'), 'wt', encoding='utf8') as comm_out, \
                    open(os.path.join(embeddings_path, f'proj_{method}-{embedding}.vec'), 'wt', encoding='utf8') as proj_out:
                comm_writer = csv.writer(comm_out, delimiter=' ')
                proj_writer = csv.writer(proj_out, delimiter=' ')
                print_size = True
                for project in projects:
                    print("Processing", project, method, embedding)
                    try:
                        comm_vectors, proj_vec = analysis.analyze(project, method, embedding)
                        communities = comm_vectors['classes'].tolist()
                        features = comm_vectors['features'].tolist()
                        if print_size:
                            print_size = False
                            size = len(features[0])
                            comm_writer.writerow([0, size])
                            proj_writer.writerow([len(projects), size])
                        for community, feature in zip(communities, features):
                            row = [f'{project}-comm-{community}']
                            row.extend(feature)
                            comm_writer.writerow(row)
                            counted += 1
                        proj_row = [project]
                        proj_row.extend(proj_vec.tolist())
                        proj_writer.writerow(proj_row)
                    except:
                        traceback.print_exc(file=sys.stdout)
                        skipped += 1
                        pass

    print("Counted", counted)
    print("Skipped", skipped)
    print("Total", skipped + counted)
