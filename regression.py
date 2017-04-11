import math

import numpy as np
import quandl
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

# Obtaining dataset for GOOGLE TICK GOOGL
df = quandl.get("WIKI/GOOGL")

# For testing
# print(df.head())

# Selecting only the columns required
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# Adding High-to-Low percentage change as column in dataframe
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
# Adding Close-to-Open percentage change as column in dataframe
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
# Filtering redundant columns and retaining columns required for regressive methods
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# For testing
# print(df.head())

# y-column in consideration
forecast_col = 'Adj. Close'
# Replacing in-place all NaN/None/Null elements with -99999
df.fillna(value=-99999, inplace=True)
# Defining the size of the estimation --> Estimate until 1% of the len(dataframe). For example if dataframe is for
# 100 days then it will estimate data points of 1 day.
forecast_out = int(math.ceil(0.01 * len(df)))
# Shifting the 'Adjusted Close' column by the number of days to be forecast, adding to columns 'label'.
# This is the actual y variable in consideration
df['label'] = df[forecast_col].shift(-forecast_out)
# Since data is shifted drop all NaN from below. Alternatively df.isfinite() method can be used as well but
# that seemed a bit more tedious
df.dropna(inplace=True)

# Initializing X matrix with 'label' (the y - variable) removed
X = np.array(df.drop(['label'], 1))
# Initializing y vector of 'label'
y = np.array(df['label'])

# X matrix is standardized i.e., centered to mean and component wise scale to unit variance
X = preprocessing.scale(X)
# Data is split into train and test set. 20% of data is kept as test set.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Defining classifier method as Ordinary Least Square Linear Regression. n_jobs = -1 => implying all cpu's will be used
# Good for training model when data set is large.
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Gives the r-square value of the model
confidence = clf.score(X_test, y_test)
print(confidence)

# for k in ['linear','poly','rbf','sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k,confidence)



