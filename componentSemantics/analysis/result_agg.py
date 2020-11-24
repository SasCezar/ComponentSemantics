from collections import defaultdict, Counter
import numpy as np
import pandas as pd


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

    agg_list = []
    best = defaultdict(list)
    for metric in overall_metrics:
        invert = metric_sign[metric]
        for level in overall_metrics[metric]:
            scores = best_algorithm(overall_metrics, metric, level, invert)
            best[metric].append(scores)
            count = Counter(scores)
            count_leiden = count["leiden"]
            count_infomap = count["infomap"]
            values_leiden = {"agg_metric": f"best {metric}", "feature_algorithm": level, "comm_algorithm": "leiden",
                             metric: count_leiden}
            values_infomap = {"agg_metric": f"best {metric}", "feature_algorithm": level, "comm_algorithm": "leiden",
                              metric: count_infomap}

            agg_list.append(values_leiden)

            agg_list.append(values_infomap)

            print(metric, level, count.most_common())

    dist = []
    for level in ["code2vec", "package", "document", "fastText", "TFIDF"]:
        for algorithm in ["leiden", "infomap"]:
            scores, distances = best_algorithm_self(overall_metrics, algorithm, level)
            dist.extend([{"level": level, "algorithm": algorithm, "dist": x} for x in distances])

            count = Counter(scores)
            print("separation", algorithm, level, count.most_common())
            print("Avg distances", np.nanmean(distances), "STD", np.nanstd(distances))

            values = {"agg_metric": "sep_value", "feature_algorithm": level, "comm_algorithm": algorithm,
                      "Separation": np.nanmean(distances)}

            agg_list.append(values)

    all_distances = pd.DataFrame(dist)
    all_distances.to_csv("distances.csv", index=False)

    agg_results = pd.DataFrame(agg_list)  # , columns=["agg_metric", "metric",
    #          "feature_algorithm", "comm_algorithm", "value"])
    scores_agreement = []
    mapping = {
        ('infomap', 'infomap', 'infomap', 'infomap', 'infomap'): "I5-L0",
        ('infomap', 'infomap', 'infomap', 'infomap', 'leiden'): "I4-L1",
        ('infomap', 'infomap', 'infomap', 'leiden', 'leiden'): "I3-L2",
        ('infomap', 'infomap', 'leiden', 'leiden', 'leiden'): "I2-L3",
        ('infomap', 'leiden', 'leiden', 'leiden', 'leiden'): "I1-L4",
        ('leiden', 'leiden', 'leiden', 'leiden', 'leiden'): "I0-L5"

    }
    for metric in overall_metrics:
        levels = Counter([tuple(sorted(t)) for t in zip(*best[metric])])
        res = {"metric": metric}
        for key, num in levels.most_common():
            res[mapping[key]] = num

        scores_agreement.append(res)

        print("agreement", metric, levels.most_common())

    df_agreement = pd.DataFrame(scores_agreement)
    df_agreement.to_csv("agreement.csv", index=False)

    return agg_results, scores_agreement
