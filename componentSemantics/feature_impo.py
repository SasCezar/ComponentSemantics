import numpy as np
import pandas
from docutils.nodes import header
from sklearn.ensemble import GradientBoostingClassifier
from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import LogisticRegression


data = pandas.read_csv('our_tfidf_1000.csv', delimiter=",")
y = list(data['y'])
X = data.drop('y', axis=1)

#model = LogisticRegression(multi_class="auto", solver="liblinear")

viz = FeatureImportances(GradientBoostingClassifier(), stack=True, relative=False, topn=40, labels=None, colormap='tab20c')
viz.fit(X, y)
viz.show()