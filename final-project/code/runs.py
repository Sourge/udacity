from data import describe_data
from simple_regression import plot_data
from simple_regression import linear_regression
from feature_relevance import check_feature_relevance
from feature_relevance import compare_above_basement
from feature_performance import check_feature_performance
from clustering import describe_cluster
from sklearn.model_selection import train_test_split as tts
from optimized_regression import train_optimized_regressors
from optimized_regression import cluster_data
from optimized_regression import predict_optimized
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR


def cluster_regressors(data):

    X = data.drop(["id", "date", "price", "sqft_above", "sqft_basement"], axis=1)
    y = data["price"]

    ## split into train and test set
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.10, random_state=42)
    X_train_clustered = cluster_data(X_train)
    describe_cluster(features=["waterfront", "view", "condition", "grade", "yr_built", "yr_renovated"],
                     X=X_train_clustered,
                     n_clusters=5)
    describe_cluster(features=["bedrooms", "bathrooms", "floors"], X=X_train_clustered, n_clusters=4)

    ## Train optimized regressors
    regressors = train_optimized_regressors(X_train_clustered, y_train)

    y_predict = predict_optimized(regressors=regressors, X=X_test)

    r2_optimized = r2_score(y_test, y_predict)
    print "r2-score for Clustered Regressors: %.4f" % r2_optimized

def linearSVR(data):
    X = data.drop(["id", "date", "price","long","lat", "zipcode","yr_renovated"], axis=1)
    y = data["price"]
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.10, random_state=42)
    svr = LinearSVR(random_state=42)
    svr.fit(X_train, y_train)
    y_predict = svr.predict(X_test)
    print "r2-score for LinearSVR: %f" % r2_score(y_test, y_predict)

def analyze_data(data):
    describe_data(data)
    X = data.drop(["id", "date", "price"], axis=1)
    y = data["price"]
    ## Feature Relevance
    check_feature_relevance(data=X)
    compare_above_basement(data=data, filename="basement_above.png")
    ## test feature performance
    check_feature_performance(X,y, filename="feature_performance.png")

def simpleLinearRegression(data):
    ## Linear Regression
    plot_data(data=data, x='sqft_living', y='price', filename='price_sqft.png')
    linear_regression(data=data, x='sqft_living', y='price', filename='linear_regression.png')

def ridgeRegression(data):
    from sklearn.linear_model import Ridge
    X = data.drop(["id", "date", "price","long","lat", "zipcode","yr_renovated","sqft_above","sqft_basement"], axis=1)
    y = data["price"]
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.10, random_state=42)
    ridge = Ridge(random_state=42)
    ridge.fit(X_train, y_train)
    y_predict = ridge.predict(X_test)
    print "r2-score for Ridge Regression: %f" % r2_score(y_test, y_predict)

