import xgboost

from class_informative_words import TFIDF
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier

from datasetEval.dataloader import lact, diSipio

hyperparameters = {'XGB': dict(booster=['gbtree', 'gblinear', 'dart']),
                   'MLP': dict(activation=['relu', 'tanh'], solver=['sgd', 'adam'], alpha=[0.001, 0.003], hidden_layer_sizes=[(512, 256, 128), (512, 128, 64)])}
models = {'XGB': xgboost.XGBClassifier(), 'MLP': MLPClassifier(max_iter=500)}


datasets = [(lact, {'path': '/home/sasce/Downloads/Classifications Dataset/LACT/msr09-data/41-software-systems',
                    'data': 'MUDA'})]
datasets = [(diSipio, {
    'path': '/home/sasce/Downloads/Classifications Dataset/Di Sipio/evaluation/evaluation structure/ten_folder_100/root'})]

args = {'max_features': 1000, 'max_df': 0.8, 'min_df': 0.1, 'token_pattern': r"(?u)\b\w\w{3,}\b"}

dataset = datasets[0][0].read(**datasets[0][1])
X, y, feature_names = TFIDF(dataset, **args)

for model in ['MLP']:
    clf = GridSearchCV(estimator=models[model], param_grid=hyperparameters[model], n_jobs=-1, return_train_score=True)
    clf.fit(X, y)

    print(clf.best_score_)

    print(clf.cv_results_)

    print(clf.best_params_)