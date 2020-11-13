from collections import defaultdict, Counter
import numpy as np


def best_algorithm(df, metric, level, inverse=False):
    leiden_scores = np.array(df[metric][level]["leiden"])
    infomap_scores = np.array(df[metric][level]["infomap"])

    result = leiden_scores > infomap_scores

    if inverse:
        result = ~result

    best_method = ["leiden" if x else "infomap" for x in result]

    return best_method


def best_algorithm_self(df, algorithm, level, inverse=False):
    leiden_scores = np.array(df["cohesion"][level][algorithm])
    infomap_scores = np.array(df["inter_similarity"][level][algorithm])

    result = leiden_scores > infomap_scores
    distances = leiden_scores - infomap_scores

    if inverse:
        result = ~result

    return result, distances


metric_sign = {
    "cohesion": False,
    "inter_similarity": True,
    "dep_sim_corr": True,
    "silhouette": False
}


def aggregate(df):
    overall_metrics = defaultdict(lambda: defaultdict(lambda: dict()))
    for metric in ["cohesion", "inter_similarity", "dep_sim_corr", "silhouette"]:
        pivot = df.pivot(index="project", columns=["feature_algorithm", "comm_algorithm"], values=metric)
        columns = sorted(pivot.columns.tolist(), key=lambda x: x[0], reverse=True)

        for column in columns:
            overall_metrics[metric][column[0]][column[1]] = pivot[column].tolist()

    for metric in overall_metrics:
        invert = metric_sign[metric]
        for level in overall_metrics[metric]:
            scores = best_algorithm(overall_metrics, metric, level, invert)

            count = Counter(scores)
            print(metric, level, count.most_common())

    for level in ["package", "document", "TFIDF"]:
        for algorithm in ["leiden", "infomap"]:
            scores, distances = best_algorithm_self(overall_metrics, algorithm, level)
            count = Counter(scores)
            print("separation", algorithm, level, count.most_common())
            print("distances", distances, np.mean(distances), np.std(distances))
