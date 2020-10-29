from analysis.semantic import SemanticScores

if __name__ == '__main__':
    in_path = "../data/"
    out_path = "../data/"
    methods = ["leiden", "infomap"]
    embeddings = ["package", "document", "TFIDF"]
    analysis = SemanticScores(in_path, out_path)
    for embedding in embeddings:
        for method in methods:
            for project in ["antlr4", "avro", "openj9"]:
                print("Processing", project, method, embedding)
                res = analysis.analyze(project, method, embedding)
                print(res)
                print("=" * 60)
