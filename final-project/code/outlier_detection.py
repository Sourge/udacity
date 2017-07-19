import numpy as np # linear algebra

def detect_outliers(data):
    # For each feature find the data points with extreme high or low values
    _out = []
    duplicate = []
    check_data = data.drop(labels=["price", "date"], axis=1)
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

    # Print the datapoints which are displayed in more than 1 feature
    # print "duplicates: ",duplicate

    print "extreme outliers: ", _extreme_outliers.index.values
    outliers = _extreme_outliers.index.values

    # Remove the outliers, if any were specified
    good_data = data.drop(data.index[outliers]).reset_index(drop=True)
    prices = data["price"].drop(data.index[outliers]).reset_index(drop=True)
    return good_data