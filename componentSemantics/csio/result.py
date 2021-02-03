def semantic_results_to_latex(df):
    for metric in ["cohesion", "inter_similarity", "dep_sim_corr", "silhouette"]:
        pivot = df.pivot(index="project", columns=["feature_algorithm", "comm_algorithm"], values=metric)
        columns = sorted(pivot.columns.tolist(), key=lambda x: x[0], reverse=True)
        print(pivot[columns].to_latex(float_format='%.4f'))
        print("=" * 60)


def agg_results_to_latex(df):
    for metric in ["cohesion", "inter_similarity", "dep_sim_corr", "silhouette"]:
        pivot = df.pivot(index="agg_metric", columns=["feature_algorithm", "comm_algorithm"], values=metric)
        columns = sorted(pivot.columns.tolist(), key=lambda x: x[0], reverse=True)
        print(pivot[columns].to_latex(float_format='%.4f'))
        print("=" * 60)
