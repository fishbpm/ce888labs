# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:47:27 2018

@author: Fish
"""

import pandas
import numpy as np
import pandas as pd

#from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

#from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

df = pd.read_csv("hUSCensus1990raw50K.csv.bz2",compression = "bz2")

df_demo = pd.DataFrame()


df_demo["AGE"] = df[["AGE"]].copy()
df_demo["INCOME"] = df[["INCOME" + str(i) for i in range(1,8)]].sum(axis = 1)

df_demo["YEARSCH"] = df[["YEARSCH"]].copy()
df_demo["ENGLISH"] = df[["ENGLISH"]].copy()
df_demo["FERTIL"] = df[["FERTIL"]].copy()
df_demo["YRSSERV"] = df[["YRSSERV"]].copy()

df_demo = pd.get_dummies(df_demo, columns = ["ENGLISH", "FERTIL" ])

X = df_demo.values[np.random.choice(df_demo.values.shape[0], 10000)]

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_db = PCA(n_components = 3).fit_transform(sc.fit_transform(X))
X_db = sc.fit_transform(X)

#print('Number of clusters: %d' % n_clusters)
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_db, labels))
scores = []
best = 0
iterations = 10
for clusters in range(2, 11):
    c_score = 0
    for i in range(iterations):
        km = AgglomerativeClustering(n_clusters = clusters).fit(X_db)#KMeans(n_clusters = clusters).fit(X_db)#
        score = metrics.silhouette_score(X_db, km.labels_)
        scores.append([clusters, score])
        c_score += score
    c_score /= iterations #compute the average score for this number clusters
    if best == 0 or c_score < best:
        best = c_score
        n_clusters = clusters

print("best is ",n_clusters, "clusters - with silhouette score of ", best)

scores_df = pd.DataFrame(scores, columns=["clusters", "score"])
plt.figure()
sns.pointplot(x="clusters", y="score", data=scores_df, join=False)

plt.savefig("confidence.png",bbox_inches='tight')
plt.savefig("confidence.pdf",bbox_inches='tight')

#n_clusters = 3
labels = AgglomerativeClustering(n_clusters = n_clusters).fit_predict(X_db)#KMeans(n_clusters = n_clusters).fit_predict(X_db)#
#km = KMeans(n_clusters = n_clusters).fit(X_db)

#X_df = pd.DataFrame(X_db, columns=df_demo.columns)#columns = ["PCA1", "PCA2", "PCA3"])#columns=df_demo.columns)
X_df = pd.DataFrame(X_db[:,:3], columns=['AGE', 'INCOME', 'YEARSCH'])
X_df['label'] = labels
#labels = km.labels_.astype(str)
#centroids = km.cluster_centers_

plt.figure()
#sns.pairplot(data=X[:,:2],hue=labels)
#sns.pairplot(data=X_df[['AGE', 'INCOME', 'YEARSCH', 'label']], hue='label', dropna=True)
#sns.pairplot(data=X_df[['AGE', 'YEARSCH', 'ENGLISH_1', 'label']], hue='label', dropna=True)
sns.pairplot(data=X_df, hue='label', dropna=True)

plt.savefig("cluster_visual.png",bbox_inches='tight')
plt.savefig("cluster_visual.pdf",bbox_inches='tight')






