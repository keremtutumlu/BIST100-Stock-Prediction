import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#İmporting libraries

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import sklearn.model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import layers


#This function takes date as string and returns datetime object
def convert_datetime(d):
    split = d.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year = year, month = month, day = day)


#This function sets our historical dataset
#It concatenates n-1 day values and nth day closing value
def set_df(dataframe, historicalDataIndex, strFirstDate, strLastDate, n):
    firstDate = convert_datetime(strFirstDate)
    lastDate = convert_datetime(strLastDate)

    targetDate = firstDate

    dates = []
    X1, X2, X3, X4, X5, X6, iX1, iX2, iX3, iX4, iX5, iX6, Y = [], [], [], [], [], [], [], [], [], [], [], [], []

    while True:
        df_subset = dataframe.loc[:targetDate].tail(n + 1)

        #We can't take Saturday and Sunday as target date because BIST100 not works this days.
        while len(df_subset) != n + 1 or targetDate.weekday() == 5 or targetDate.weekday() == 6:
            if targetDate.weekday() == 4:
                targetDate = targetDate + datetime.timedelta(days=3)
                df_subset = dataframe.loc[:targetDate].tail(n + 1)

            else:
                targetDate = targetDate + datetime.timedelta(days=1)
                df_subset = dataframe.loc[:targetDate].tail(n + 1)

            if (targetDate - lastDate).days >= 0:
                break

        dfIndex = historicalDataIndex.loc[:targetDate].tail(n + 1)

        val1 = df_subset['Close'].to_numpy()
        val2 = df_subset['Adj Close'].to_numpy()
        val3 = df_subset['Open'].to_numpy()
        val4 = df_subset['High'].to_numpy()
        val5 = df_subset['Low'].to_numpy()
        val6 = df_subset['Volume'].to_numpy()

        ival1 = dfIndex['Close'].to_numpy()
        ival2 = dfIndex['Adj Close'].to_numpy()
        ival3 = dfIndex['Open'].to_numpy()
        ival4 = dfIndex['High'].to_numpy()
        ival5 = dfIndex['Low'].to_numpy()
        ival6 = dfIndex['Volume'].to_numpy()

        y = val1[-1]
        x1, x2, x3, x4, x5, x6 = val1[:-1], val2[:-1], val3[:-1], val4[:-1], val5[:-1], val6[:-1]
        ix1, ix2, ix3, ix4, ix5, ix6 = ival1[:-1], ival2[:-1], ival3[:-1], ival4[:-1], ival5[:-1], ival6[:-1]

        dates.append(targetDate)

        X1.append(x1)
        X2.append(x2)
        X3.append(x3)
        X4.append(x4)
        X5.append(x5)
        X6.append(x6)

        iX1.append(ix1)
        iX2.append(ix2)
        iX3.append(ix3)
        iX4.append(ix4)
        iX5.append(ix5)
        iX6.append(ix6)

        Y.append(y)

        if targetDate.weekday() == 4:
            targetDate = targetDate + datetime.timedelta(days=3)
        else:
            targetDate = targetDate + datetime.timedelta(days=1)

        if (targetDate - lastDate).days >= 0:
            break

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    X5 = np.array(X5)
    X6 = np.array(X6)
    iX1 = np.array(iX1)
    iX2 = np.array(iX2)
    iX3 = np.array(iX3)
    iX4 = np.array(iX4)
    iX5 = np.array(iX5)
    iX6 = np.array(iX6)

    for i in range(0, n):
        ret_df[f'Close-{n - i}'] = X1[:, i]
        ret_df[f'Adj Close-{n - i}'] = X2[:, i]
        ret_df[f'Open-{n - i}'] = X3[:, i]
        ret_df[f'High-{n - i}'] = X4[:, i]
        ret_df[f'Low-{n - i}'] = X5[:, i]
        ret_df[f'Volume-{n - i}'] = X6[:, i]

        ret_df[f'I-Close-{n - i}'] = iX1[:, i]
        ret_df[f'I-Adj Close-{n - i}'] = iX2[:, i]
        ret_df[f'I-Open-{n - i}'] = iX3[:, i]
        ret_df[f'I-High-{n - i}'] = iX4[:, i]
        ret_df[f'I-Low-{n - i}'] = iX5[:, i]
        ret_df[f'I-Volume-{n - i}'] = iX6[:, i]

    ret_df['Daily Target'] = Y

    return ret_df

#This 3 functions returns numpy array from pandas setted dataframe.
def windowed_df_to_date(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()
  dates = df_as_np[:, 0]
  return dates

def windowed_df_to_X(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()
  dates = df_as_np[:, 0]
  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
  return X.astype(np.float32)

def windowed_df_to_Y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()
  Y = df_as_np[:, -1]
  return Y.astype(np.float32)

#Reading datas from .csv file.
stocksDf = pd.read_csv('C:/Users/kerem/Desktop/Masaustu/Dersler/Pdfler/5. Yarıyıl/Yapay Sinir Ağları/Proje/stockPredict/stocksList.csv')
stocksDf.index = stocksDf.pop('Sıra No')
stocksCode = stocksDf['Pay Kodu'].to_numpy()

#Extracting data from yahoo finance witf yfinance library.
historicalDataStocks = []
for i in range(0, len(stocksCode)):
    dfTemp = yf.download(stocksCode[i] + '.IS')
    historicalDataStocks.append(dfTemp)

length = len(historicalDataStocks)
i = 0
while i < length:
    if len(historicalDataStocks[i]) <= 5:
        del historicalDataStocks[i]
        stocksCode = np.delete(stocksCode, i)
        i = i - 1
        length = length - 1
    historicalDataStocks[i] = historicalDataStocks[i][['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close']]
    i = i + 1

#Extracting BIST100 index data.
historicalDataIndex = yf.download('XU100.IS')

settedDfs = []

#We use set_df function for setting our dataframe for training.
for i in range(0, len(historicalDataStocks)):
    settedDfs.append(set_df(
        historicalDataStocks[i],
        historicalDataIndex,
        (historicalDataStocks[i].index[0] + datetime.timedelta(days=4)).strftime("%Y-%m-%d"),
        (historicalDataStocks[i].index[-1]).strftime("%Y-%m-%d"),
        n=4
    ))

dates, X, Y = [], [], []

#We convert our setted dataframes to numpy arrays.
for i in range(0, len(settedDfs)):
    dates.append(windowed_df_to_date(settedDfs[i]))
    X.append(windowed_df_to_X(settedDfs[i]))
    Y.append(windowed_df_to_Y(settedDfs[i]))

X_train = np.concatenate(X)
Y_train = np.concatenate(Y)
dates_train = np.concatenate(dates)

#We use k-fold cross validation for prevent overfitting.
kFold = sklearn.model_selection.KFold(n_splits=5)

#We creating our model.
model = Sequential([
    layers.Input((48,1)),
    layers.LSTM(128),
    layers.Dropout(0.2),
    layers.Dense(256, activation='LeakyReLU'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='LeakyReLU'),
    layers.Dense(1)
])

model.compile(loss='mse',
              optimizer=AdamW(learning_rate=0.001),
              metrics=['mean_absolute_error'])

#Model training.
for train, test in kFold.split(X_train, Y_train):
    model.fit(X_train[train], Y_train[train], epochs=75)

#Saving our model.
model.save('model7.h5')