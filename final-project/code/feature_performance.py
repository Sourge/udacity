from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt
import numpy as np

def check_feature_performance(X, y, filename):
    print "checking feature performance... (this might take a while)"
    forest = ExtraTreesClassifier(n_estimators=100,criterion="entropy",min_samples_split=10,n_jobs=-1,random_state=42)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    for f in range(X.shape[1]):
        print("----Feature \"%s\"---- " % (X.keys()[f]))
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig("img/"+filename)
    plt.clf()
    print "...done checking feature performance"