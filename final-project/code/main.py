from data import load_data
from runs import simpleLinearRegression, analyze_data, ridgeRegression, cluster_regressors, linearSVR
from outlier_detection import detect_outliers

## Load Data
df = load_data()

## Analyze Data
analyze_data(data=df)

## Outlier Detection
detect_outliers(data=df)

to_drop = [15870]
good_data = df.drop(df.index[to_drop]).reset_index(drop=True)


## Regression Split using Clusters
cluster_regressors(data=good_data)

## Linear SVR
linearSVR(data=good_data)

## Ridge Regression
ridgeRegression(data=good_data)

## simple Linear Regression
simpleLinearRegression(data=df)