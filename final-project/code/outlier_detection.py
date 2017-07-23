import numpy as np # linear algebra
import matplotlib.pyplot as plt

def detect_outliers(data):
    # For each feature find the data points with extreme high or low values
    _out = []
    duplicate = []
    check_data = data.drop(labels=["id","date"], axis=1)
    extreme_count = dict.fromkeys(data.index.values,0)
    for feature in check_data.keys():

        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(check_data[feature], 25)

        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(check_data[feature], 75)

        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)
        extreme_step = 3 * (Q3 - Q1)

        # Display the outliers
        print "Calculating Data points considered outliers for the feature '{}'...".format(feature)
        _outliers = check_data[~((check_data[feature] >= Q1 - step) & (check_data[feature] <= Q3 + step))]
        _extreme_outliers = check_data[~((check_data[feature] >= Q1 - extreme_step) & (check_data[feature] <= Q3 + extreme_step))]
        for index in _outliers.index.values:
            if index in _out:
                duplicate.append(index)
            else:
                _out.append(index)
        if feature != "price":
            plot_data(x_outlier=_outliers, x_extreme=_extreme_outliers, x_normal=data, feature=feature)
        for i in _extreme_outliers.index.values:
            extreme_count[i] = extreme_count[i]+1
    # Remove the outliers, if any were specified


def plot_data(x_outlier, x_extreme, x_normal, feature):
    plt.scatter(y=x_normal[feature], x=x_normal["price"], color="green")
    plt.scatter(y=x_outlier[feature], x=x_outlier["price"], color="yellow")
    plt.scatter(y=x_extreme[feature], x=x_extreme["price"], color="red")
    plt.savefig("img/outliers/"+feature+".png" )
    plt.clf()