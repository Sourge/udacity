
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = "kc_house_data.csv"

df = pd.read_csv(data, header = 0)

data = df.drop(labels = ["id","price","date"], axis=1)
for feature in data.keys():
    print 'Checking Feature "',feature,'"'
    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    X = data.drop(labels = feature, axis=1)

    from sklearn.cross_validation import train_test_split
    # Split the data into training and testing sets using the given feature as the target
    y = data[feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    from sklearn.tree import DecisionTreeRegressor
    # Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state = 42)
    regressor.fit(X_train, y_train)

    # Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    print "The Regressor score is: ",score,"\n"