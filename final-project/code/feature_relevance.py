from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

def check_feature_relevance(data):
    redundant = []
    significant = []
    for feature in data.keys():
        print 'Checking Feature "', feature, '"'
        # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
        X = data.drop(labels=feature, axis=1)


        # Split the data into training and testing sets using the given feature as the target
        y = data[feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


        # Create a decision tree regressor and fit it to the training set
        regressor = DecisionTreeRegressor(random_state=42)
        regressor.fit(X_train, y_train)

        # Report the score of the prediction using the testing set
        score = regressor.score(X_test, y_test)
        print "The Regressor score is: ", score, "\n"
        if score > 0.95:
            redundant.append(feature)
        if score < 0:
            significant.append(feature)

    print "Redundant features(>0) : ", redundant, "\n"
    print "Significant features(<0) : ", significant, "\n"

def compare_above_basement(data, filename):
    basement = data['sqft_basement']
    no_basement = basement[basement == 0].keys()
    has_basement = data.drop(no_basement).reset_index()
    above = has_basement['sqft_above']
    has_basement['ratio'] = above.div(has_basement['sqft_basement'], fill_value=1)
    has_basement.plot.scatter(x='ratio', y='price')
    plt.savefig("img/"+filename)
    plt.clf()