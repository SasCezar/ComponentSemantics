import os
import sys
import traceback

import pandas

from analysis.result_agg import aggregate
from analysis.semantic import SemanticScores

if __name__ == '__main__':
    in_path = "../data/"
    out_path = "../data_test/"
    methods = ["leiden", "infomap"]
    embeddings = ["code2vec", "package", "document", "fastText", "TFIDF"]
    analysis = SemanticScores(in_path, out_path, visualize=False)
    df = pandas.DataFrame()

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
            for project in projects:
                print("Processing", project, method, embedding)
                try:
                    res = analysis.analyze(project, method, embedding)
                    sz = res["comm_size"]
                    sz = [x for x in sz if x > 3]
                    del res["comm_size"]

                    if (method, project) not in seen:
                        sizes.extend([{"algorithm": method, "size": x} for x in sz])
                        nums.append({"algorithm": method, "num": len(sz)})
                        seen.add((method, project))

                    res.update({"comm_algorithm": method,
                                "feature_algorithm": embedding,
                                "project": project})
                    df = df.append(res, ignore_index=True)
                    counted += 1
                except:
                    traceback.print_exc(file=sys.stdout)
                    skipped += 1
                    pass

    print("Counted", counted)
    print("Skipped", skipped)
    print("Total", skipped + counted)

    sizes = pandas.DataFrame(sizes)
    sizes.to_csv("comm_sizes.csv", index=False)

    nums = pandas.DataFrame(nums)
    nums.to_csv("num_comm.csv", index=False)
    all_df = pandas.DataFrame(df)
    all_df.to_csv("scores.csv")
    values, _ = aggregate(df)

    # print(df)
    # semantic_results_to_latex(df)
    # print(values)
