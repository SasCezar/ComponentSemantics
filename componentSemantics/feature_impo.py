import numpy as np
from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import LogisticRegression


data = np.loadtxt('out_tfidf_1000.csv', skiprows=1)
X = data[:, :1000]
y = data[:,  1001]

model = LogisticRegression(multi_class="auto", solver="liblinear")
viz = FeatureImportances(model, stack=True, relative=False)
viz.fit(X, y)
viz.show()