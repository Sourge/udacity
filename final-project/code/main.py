from data import load_data
from runs import simpleLinearRegression, analyze_data, ridgeRegression, cluster_regressors, linearSVR
from outlier_detection import detect_outliers

## Load Data
df = load_data()

## Analyze Data
analyze_data(data=df)

## simple Linear Regression
simpleLinearRegression(data=df)

## Outlier Detection
good_data = detect_outliers(data=df)

## Regression Split using Clusters
cluster_regressors(data=good_data)

## Linear SVR
linearSVR(data=good_data)

## Ridge Regression
ridgeRegression(data=good_data)