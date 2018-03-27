# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:37:41 2018

@author: Fish
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import mean_squared_error as mse
#from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import make_scorer

df = pd.read_csv("./bank-additional-full.csv", delimiter = ";")

#convert categorical fields to ordinal
df_dummies = pd.get_dummies(df)

#delete redundant fields
del df_dummies['duration'] #is not known prior to the call, so it is not a predictor
del df_dummies['y_no'] #is redundant as it is equivalent to  y_yes == 0

#plot target class
plt.hist(df_dummies['y_yes'], bins=2)

df_dummies = df_dummies.dropna()
X = df_dummies.copy()
del X['y_yes']
#y = df_dummies['y_yes'].copy()

X = X.values #convert to numpy array
y = df_dummies['y_yes'].values

clf = ExtraTreesClassifier(n_estimators = 2000,max_depth = 4)
scores = cross_val_score(clf, X, y, cv=10, scoring = make_scorer(acc))

print("ACC 10-Kfold stratified CV: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
clf.fit(X, y)
#clf.score(X, y)
print("ACC: %0.2f")% (acc(y,clf.predict(X))) #this gives identical result as above score

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(14):
    print("%d. %s (%f)" % (f + 1, df_dummies.columns[indices[f]],  importances[indices[f]]))
    
fig = plt.figure()
plt.title("Feature importances")
plt.bar(range(14), importances[indices[:14]], color="r", yerr=std[indices[:14]], align="center")
plt.xticks(range(14), df_dummies.columns[indices[:14]])
plt.xlim([-1, 14])
fig.set_size_inches(15,8)
axes = plt.gca()
axes.set_ylim([0,None])
axes.set_xticklabels(df_dummies.columns[indices[:14]], rotation=45)
plt.savefig("importances.png",bbox_inches='tight')
plt.savefig("importances.pdf",bbox_inches='tight')
    
    
    
    
    
