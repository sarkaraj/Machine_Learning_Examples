import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

from yahoo_fin_api import get_stock_data_current

# all "internal gelsd" warnings are ignored. Filtering is optional.
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

print "Data Acquisition started"

# Obtaining dataset for GOOGLE TICK GOOGL for 1 year
df = get_stock_data_current('GOOGL', days=365 * 5)


# Adding High-to-Low percentage change as column in dataframe
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
# Adding Close-to-Open percentage change as column in dataframe
df['CO_PCT'] = (df['Adj_Close'] - df['Open']) / df['Open'] * 100.0


# Filtering redundant columns and retaining columns required for regressive methods
df = df[['Adj_Close', 'HL_PCT', 'CO_PCT', 'Volume']]

print "Data cleansing completed"
print "Preparing Data for performing Linear Regression"


# y-column in consideration
forecast_col = 'Adj_Close'
# Replacing in-place all NaN/None/Null elements with -99999
df.fillna(value=-99999, inplace=True)

# Defining the size of the estimation --> Estimate until 1% of the len(dataframe). For example if dataframe is for
# 100 days then it will estimate data points of 1 day.
forecast_out = int(math.ceil(0.1 * len(df)))

# Shifting the 'Adjusted Close' column by the number of days to be forecast, adding to columns 'label'.
# This is the actual y variable in consideration
df['label'] = df[forecast_col].shift(-forecast_out)


# Initializing X matrix with 'label' (the y - variable) removed
X = np.array(df.drop(['label'], 1))

# X matrix is standardized i.e., centered to mean and component wise scale to unit variance
X = preprocessing.scale(X)

X_predict = X[-forecast_out:]
X = X[:-forecast_out]

# Since data is shifted drop all NaN from below. Alternatively df.isfinite() method can be used as well but
# that seemed a bit more tedious
df.dropna(inplace=True)

# Initializing y vector of 'label'
y = np.array(df['label'])



print "Data processing for regression completed"

# Data is split into train and test set. 20% of data is kept as test set.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Defining classifier method as Ordinary Least Square Linear Regression. n_jobs = -1 => implying all cpu's will be used
# Good for training models when data set is large.
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Gives the r-square value of the models
confidence = clf.score(X_test, y_test)
print(confidence)

# with open('models/regression/linearregression_5Y_10PCT_C.pickle','wb') as f:
#     pickle.dump(clf, f)

# Predicting values
forecast_set = clf.predict(X_predict)

print "Forecast Set"
print forecast_set
print "Forecast Out"
print forecast_out

# # Testing r-square values for different kernel methods
# for k in ['linear','poly','rbf','sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k,confidence)

style.use('ggplot')
df['Forecast'] = np.nan

# print df.iloc[1].name

last_date = df.iloc[-1].name

print last_date
# print type(last_date)
# next_day = last_date + pd.Timedelta(days=1)
# print next_day

one_day = pd.Timedelta(days=1)
next_date = last_date + one_day

for i in forecast_set:
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    next_date += one_day

df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
