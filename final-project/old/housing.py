
# coding: utf-8

# # Predict House Prices in King Country #
# __Note: gmaps plugin neeeds to be installed:__
# 
# conda install -c conda-forge gmaps
# 

# In[1]:

# get_ipython().magic(u'matplotlib inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ## Loading the dataset ##

# In[2]:

data = "kc_house_data.csv"

df = pd.read_csv(data, header = 0)


# ## Getting first information about the data ##
# To get a first impression about the data, will try display useful information about every attribute.

# In[3]:

df.describe()


# #### Price ####
# The first values I will have a closer look at will be the price.
# 
# We have prices ranging from roughly \$75,000 to  \$7,700,000. 50% of the prices is around \$450,000.
# The standard deviation is \$360,000. The mean of the prices is \$540,000.
# If we look at the quarters we have:
# - first 25% span over \$235,000
# - second 25% span over \$130,000
# - third 25% span over \$195,000
# - top 25% span \$7,125,000
# 
# This clearly shows that the first 3/4 of all houses lie in a pretty similiar range and are at least somewhat evenly distributed in their quarters. The top 25% are really far out regarding their prices which might be caused by a lot of expensive houses mixed with some extremely expensive houses.
# 
# #### Living Sqft. ####
# The second value I will examine is the living space (in square feet). This value spans in between 290 to 13,540 sqft.
# The standard deviation is 918 and the mean around 2080 sqft. This points that most of the houses should be in between 1000 to 3000 sqft. When looking at the quarter distribution you get the following picture:
# - first 25% is 1137 sqft. in distance
# - second 25% spans 483 sqft.
# - third 25% is 640 sqft.
# - the last 25% is 9990 sqft.
# 
# This huge difference proves the previous estimation as at least 75% of the houses are in between 290 to 2550 sqft. in living space. This also kind of resembles the discoveries of looking at the house prices.

# ## Simple Regression model ##
# To create a benchmark model we will first do a simple linear regression on the relation between living space and price of the house.
# ### Relationship between price and living space ###
# First, I will print the relationship between price and living space.

# In[4]:

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

df.plot.scatter(x='sqft_living', y='price')


# ### Linear Regression ###
# As seen here, most of the house are forming some kind of triangle on the graph which looks like it will work good with a linear regression.
# 
# 

# In[5]:

from sklearn import linear_model

regression = linear_model.LinearRegression()
regr_X = df['sqft_living'].reshape(21613,-1)
regr_y = df['price']

regression.fit(regr_X, regr_y)
plt.scatter(regr_X, regr_y,  color='blue')
plt.plot(regr_X, regression.predict(regr_X), color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


# ### R^2 Score of the simple regression ###
# For comparision the R^2 Score of this linear regression will be calculated in the following.

# In[6]:

from sklearn.metrics import r2_score

print "For the Simple Regression the R2 Score is: %.4f" % r2_score(regr_y, regression.predict(regr_X))


# ## Improving the Predicition model ##
# This part is about finding a better metric for predicting future house sales regarding their price.
# 
# First, I will detect outliers and delete them from the dataset if needed.

# ### Detecting Outliers ###
# The first step to improve our learning behaviour is to find outliers and then remove them from the data set if needed.
# To detect outliers I will use the Isolation Forest Algorithm which is good for high-dimensional data sets as we have present here. 

# In[ ]:

from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(df)
y = clf.predict(df)
print y


# ### Location based prices ###
# House prices don't only depend on the size of the house or amount of rooms, but are also really dependant on the location of said house. To get an idea how the position might impact my data I analyse the relationship between location and price in my dataset.

# In[ ]:

import gmaps
gmaps.configure(api_key="AIzaSyDPWAl8lcrK9q-tOkrl64sGkxDnbWz47Ko")

locations = df[["lat", "long"]]
prices = df["price"]

heatmap_layer = gmaps.heatmap_layer(locations, weights=prices)
heatmap_layer.max_intensity = 7200000
heatmap_layer.point_radius = 4

fig = gmaps.figure()
fig.add_layer(heatmap_layer)
fig


# This graph shows that there seems to be a real relationship between location and price. Especially in the center of Seattle the prices are much higher. There is also the town of Snoqualmie which is known for having a lot of highly educated inhabitants. That circumstance leads to the people of Snoqualmie having a substantially higher household income than the average. 
# 
# Therefore, the housing prices are also higher in this area. The same reasons for higher living costs can also be applied to Seattle, which is much more attractive to live in for wealthy people.

# ### Using Support Vector Machines ###
# Because of the higher dimensionality of the feature space SVM might be well suited for this problem. In the following implementation I try to use SVM to improve my predictions.

# In[ ]:

from sklearn.svm import SVR
X = df[['sqft_living']][:10]
y = df['price'][:10]
#X = np.sort(5 * np.random.rand(40, 1), axis=0)
#y = np.sin(X).ravel()

svr_rbf = SVR(kernel='linear', C=1e8, epsilon=0.1)
y_rbf = svr_rbf.fit(X, y).predict(X)

lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# ### Improved Regression Model ###

# In[ ]:




# In[ ]:



