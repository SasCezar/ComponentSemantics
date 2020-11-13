from collections import defaultdict

import pandas

from analysis.result_agg import aggregate
from analysis.semantic import SemanticScores
from csio.result import semantic_results_to_latex

if __name__ == '__main__':
    in_path = "../data/"
    out_path = "../data/"
    methods = ["leiden", "infomap"]
    embeddings = ["package", "document", "TFIDF"]
    analysis = SemanticScores(in_path, out_path)
    df = pandas.DataFrame()
    for method in methods:
        for embedding in embeddings:
            for project in ["antlr4", "avro", "openj9"]:
                print("Processing", project, method, embedding)
                res = analysis.analyze(project, method, embedding)
                res.update({"comm_algorithm": method,
                            "feature_algorithm": embedding,
                            "project": project})

                df = df.append(res, ignore_index=True)

    aggregate(df)

    # semantic_results_to_latex(df)
