import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
from sklearn.metrics import r2_score
matplotlib.style.use('ggplot')

def linear_regression(data, x, y, filename):
    regression = linear_model.LinearRegression()
    regr_X = data[x].reshape(21613, -1)
    regr_y = data[y]

    regression.fit(regr_X, regr_y)
    plt.scatter(regr_X, regr_y, color='blue')
    plt.plot(regr_X, regression.predict(regr_X), color='red', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("img/"+filename)
    plt.clf()

    print "For the Simple Regression the R2 Score is: %.4f" % r2_score(regr_y, regression.predict(regr_X))