from sklearn.cluster import KMeans
import pandas as pd


def cluster_features(X, features=[], n_clusters=2):
    X_cluster = X[features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_jobs=-1)
    predict = kmeans.fit_predict(X_cluster)
    return pd.Series(predict, index=X_cluster.index)


def describe_cluster(X, features=[], n_clusters=2):
    for i in range(n_clusters):
        query_str = "quality_cluster==%d" % i
        segment = X.query(query_str)[features]
        print "----For Cluster %d----" % i
        print segment.describe()
        print "--Examples--"
        print segment[:10]