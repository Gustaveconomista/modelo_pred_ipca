# %% [markdown]
# # Modelo Preditivo de Inflação - AM

# %% [markdown]
# ## Instalação e Importe de Pacotes

# %%
#!pip install python-bcb

# %%
#!pip install pmdarima

# %%
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from bcb import sgs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import sys
sys.path.insert(0, '..')

# %% [markdown]
# ## Importação da Base de Dados

# %%
df = sgs.get({'IPCA': 433}, start='1994-08-01')

# %%
df

# %% [markdown]
# ## Visualização dos Dados

# %%
df.plot(figsize=(16, 5))

# %%


def plot_df(df, x, y, title="", xlabel='Date', ylabel='IPCA'):
    plt.figure(figsize=(16, 5))
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


plot_df(df, x=df.index, y=df.IPCA, title='Monthly IPCA from 1994 to 2022')

# %%
rolling_mean = df.rolling(3).mean()
rolling_std = df.rolling(3).std()

# %%
plt.figure(figsize=(16, 5))
plt.plot(df, color="blue", label="Monthly IPCA from 1994 to 2022")
plt.plot(rolling_mean, color="red", label="Rolling Mean in IPCA")
plt.plot(rolling_std, color="black",
         label="Rolling Standard Deviation in IPCA")
plt.title("IPCA Time Series, Rolling Mean, Standard Deviation")
plt.legend(loc="best")

# %% [markdown]
# ## Estacionariedade

# %%
adft = adfuller(df, autolag="AIC")

# %%
output_df = pd.DataFrame({"Values": [adft[0], adft[1], adft[2], adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']], "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",
                                                                                                                                   "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
print(output_df)

# %% [markdown]
# ## Autocorrelação

# %%
autocorrelation_lag1 = df['IPCA'].autocorr(lag=1)
print("One Month Lag: ", autocorrelation_lag1)
autocorrelation_lag3 = df['IPCA'].autocorr(lag=3)
print("Three Months Lag: ", autocorrelation_lag3)
autocorrelation_lag6 = df['IPCA'].autocorr(lag=6)
print("Six Months Lag: ", autocorrelation_lag6)
autocorrelation_lag9 = df['IPCA'].autocorr(lag=9)
print("Nine Months Lag: ", autocorrelation_lag9)

# %% [markdown]
# ## Decomposição

# %%
decompose = seasonal_decompose(df['IPCA'], model='additive', period=6)
decompose.plot()
plt.show()

# %% [markdown]
# ## Previsão

# %% [markdown]
# Criando as bases de treinamento e de teste:

# %%
df[df['Date'] <= '2021-09-01']

# %%
train = df[df['Date'] <= '2021-09-01']
train['train'] = train['IPCA']
del train['Date']
del train['IPCA']
test = df[df['Date'] >= '2021-09-01']
del test['Date']
test['test'] = test['IPCA']
del test['IPCA']
plt.figure(figsize=(16, 5))
plt.plot(train, color="black")
plt.plot(test, color="red")
plt.title("Train/Test split for IPCA Data")
plt.ylabel("IPCA")
plt.xlabel('Year-Month')
sns.set()
plt.show()

# %%
model = auto_arima(train, trace=True, error_action='ignore',
                   suppress_warnings=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

# %%
plt.figure(figsize=(16, 5))
plt.plot(train, color="blue", label="Train")
plt.plot(test, color="red", label="Test")
plt.plot(forecast, color="green", label="Prediction")
plt.title("IPCA Prediction")
plt.legend(loc="best")

# %%
rms = sqrt(mean_squared_error(test, forecast))
print("RMSE: ", rms)

# %%
