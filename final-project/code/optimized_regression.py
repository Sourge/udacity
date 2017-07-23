from sklearn.linear_model import Ridge
from clustering import cluster_features
from matplotlib import pyplot as plt
import pandas as pd

def cluster_data(X):
    quality_cluster = cluster_features(
        features=["waterfront", "view", "condition", "grade", "yr_built", "yr_renovated"],
        X=X, n_clusters=5)
    space_cluster = cluster_features(features=["bedrooms", "bathrooms", "floors"], X=X, n_clusters=4)
    X = X.assign(quality_cluster=quality_cluster.values)
    X = X.assign(space_cluster=space_cluster.values)
    return X

def train_optimized_regressors(X,y):
    regressors = {}
    print y
    print "Starting to train optimized regressors..."
    for i in range(5):
        loop = {}
        for j in range(4):
            regression = Ridge(random_state=42)
            query_str = "quality_cluster==%d and space_cluster==%d" % (i, j)
            regr_data = X
            regr_data = regr_data.assign(price=y.values)
            regr_data = regr_data.query(query_str)
            print "Training Regressor for Quality Cluster = %d and Space Cluster = %d (Training Set Size: %d)" % (i, j, len(regr_data))
            loop_X = regr_data["sqft_living"].reshape(len(regr_data["sqft_living"]),1)
            clf = regression.fit(loop_X , regr_data["price"])
            loop[j] = clf
            plt.scatter(regr_data["sqft_living"], regr_data["price"], color='blue')
            plt.plot(regr_data["sqft_living"], regression.predict(loop_X), color='red', linewidth=2)
            plt.savefig("img/train/"+str(i)+"_"+str(j)+".png")
            plt.clf()
        regressors[i] = loop
    print "...done training optimized regressors"
    return regressors


def predict_optimized(regressors, X):
    X = cluster_data(X)
    predict_y = pd.Series()
    print "Starting to predict with optimized regressors..."
    for i in range(5):
        for j in range(4):
            query_str = "quality_cluster==%d and space_cluster==%d" % (i, j)
            regr_X = X
            regr_X = regr_X.query(query_str)
            print "Predicting with Regressor for Quality Cluster = %d and Space Cluster = %d (Test Set Size: %d)" % (
            i, j, len(regr_X))
            regressor = regressors[i][j]
            loop_X = regr_X["sqft_living"].reshape(len(regr_X["sqft_living"]),1)
            loop_y = pd.Series(regressor.predict(loop_X),index=regr_X.index)
            predict_y = predict_y.append(loop_y)
            plt.scatter(loop_X, loop_y, color='blue')
            plt.plot(loop_X, regressor.predict(loop_X), color='red', linewidth=2)
            plt.savefig("img/prediction/"+str(i)+"_"+str(j)+".png")
            plt.clf()
    print "...done predicting"
    return predict_y
