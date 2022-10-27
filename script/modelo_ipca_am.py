# %% [markdown]
# # Modelo Preditivo de Inflação - AM

# %% [markdown]
# ## Instalação e Importe de Pacotes

# %% [markdown]
# `!pip install python-bcb`
# 
# `!pip install pmdarima`
# 
# `!pip install darts`
# 
# `!pip install -U optuna`

# %%
import pandas as pd
import darts
import sys
sys.path.insert(0, '..')
from datetime import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
pd.options.display.float_format = '{:.2f}'.format
import time
t_start1 = time.perf_counter()
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)
import torch
import random
import numpy as np
from numpy import cov
import numbers
import math
from functools import reduce
from scipy.stats import normaltest
from bcb import sgs
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm.notebook import tqdm
from pytorch_lightning.callbacks import Callback, EarlyStopping
import pmdarima as pmd
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    Theta,
    BATS,
    RandomForest,
    CatBoostModel,
    BATS,
    LightGBMModel
)
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.dataprocessing.transformers.boxcox import BoxCox
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
%matplotlib inline

# %% [markdown]
# ## Carregamento dos Dados

# %% [markdown]
# O dataset importado aqui foi retirado do seguinte [repositório](https://github.com/gabrielrvsc/HDeconometrics), relacionado ao principal [artigo](https://www.sciencedirect.com/science/article/abs/pii/S0169207017300262) utilizado como referência.

# %%
df1 = pd.read_csv('BRinf.csv')
df1.rename(columns={"Unnamed: 0": "data", "Brazil CPI IPCA MoM": "IPCA", "Brazil Selic Target Rate": "Selic", "USD-BRL X-RATE": "Cambio"}, inplace=True)
df1['data']=pd.to_datetime(df1['data'])
df1 = df1[df1.data >= '2004-01-31']
df1.reset_index(drop=True, inplace=True)
df1['ano']= df1['data'].dt.year
df1['mes']= df1['data'].dt.month
df1['trim']=df1['data'].dt.quarter
df1

# %% [markdown]
# Com o comando 'shape' podemos ver a dimensão de nosso dataset, onde 'n' representa o nº de observações e 'p' o nº de variáveis.

# %%
df1.shape

# %% [markdown]
# Ou seja, nossa base tem 144 observações e 96 variáveis.

# %% [markdown]
# ## Tratamento dos Dados

# %% [markdown]
# Vamos 1º verificar as estatísticas descritivas de nossas séries.

# %%
df1.describe()

# %% [markdown]
# Para realizar as estimações e previsões com os vários modelos que construiremos aqui, precisamos primeiro transformar nossa base para o tipo "timeseries", ou seja, precisamos converte-la para o formato de série temporal.

# %%
target = TimeSeries.from_dataframe(df1, "data", "IPCA",fill_missing_dates=True, freq="M")
pcovariates = TimeSeries.from_dataframe(df1, "data", df1.columns.tolist()[2:],fill_missing_dates=True, freq="M")
fcovariates = TimeSeries.from_dataframe(df1, "data", df1.columns.tolist()[59:],fill_missing_dates=True, freq="M")

# %% [markdown]
# Ainda, vamos dividir a base em 3 subconjuntos, sendo o "target" a série de nossa variável de interesse (IPCA), o "pcovariates", nossas covariadas das quais utilizaremos seus valores passados, e o "fcovariates", nosso conjunto de séries das quais exploraremos seus valores futuros para dar suporte a estimação.

# %%
pcovariates.pd_dataframe()

# %%
fcovariates.pd_dataframe()

# %% [markdown]
# Definindo os parâmetros para o processo de estimação.

# %%
DROP = 0.1
LEARN = 0.001
EPOCH = 300  
MSEAS = 11                   # seasonality default
ALPHA = 0.05                  # significance level default
TRAIN_VAL_SPLIT = dt(2011,12,31).date() 
FC_N = 1

# %% [markdown]
# Verificando a presença de sazonalidade e sua ordem.

# %%
for m in range(2, 37):
    is_seasonal, mseas = check_seasonality(target, m=m, alpha=ALPHA)
    if is_seasonal:
        break

print("Seasonal? " + str(is_seasonal))
if is_seasonal:
    print('There is seasonality of order {}.'.format(mseas))

# %% [markdown]
# Dividindo as bases em conjuntos de treino e de validação.

# %%
if isinstance(TRAIN_VAL_SPLIT, numbers.Number):
    split_at = TRAIN_VAL_SPLIT
else:
    split_at = pd.Timestamp(TRAIN_VAL_SPLIT)
train, val = target._split_at(split_at)
fcov_tr, fcov_val = fcovariates._split_at(split_at)
pcov_tr, pcov_val = pcovariates.split_before(split_at)

# %% [markdown]
# Gerando séries da média móvel e do desvio-padrão móvel de nossa variável de interesse.

# %%
rolling_mean = df1['IPCA'].rolling(3).mean()
rolling_std = df1['IPCA'].rolling(3).std()

# %% [markdown]
# ## Visualização dos Dados

# %% [markdown]
# Plotando as séries de treinamento e validação de nossa variável de interesse.

# %%
plt.figure(101, figsize=(12, 5))
train.plot(label='training')
val.plot(label='validation')
plt.legend();

# %% [markdown]
# Plotando a série do IPCA junto com sua média móvel e seu desvio-padrão móvel.}

# %%
plt.figure(figsize=(9,5))
plt.plot(df1.data, df1['IPCA'], color="blue", label="Monthly IPCA from 2004 to 2015")
plt.plot(df1.data, rolling_mean, color="red", label="Rolling Mean in IPCA")
plt.plot(df1.data, rolling_std, color="black", label = "Rolling Standard Deviation in IPCA")
plt.title("IPCA Time Series, Rolling Mean, Rolling Standard Deviation")
plt.legend(loc="best")

# %% [markdown]
# Outro ponto importante sobre uma série temporal diz respeito a sua decomposição.
# Uma série de tempo pode ser dividida em 4 componentes:
# 
#     * Componente estacionário;
#     * Tendência;
#     * Sazonalidade;
#     * Ruído.
#     
# Vamos plotar 4 gráficos representando a decomposição de nossa série do IPCA.

# %%
decompose = seasonal_decompose(df1['IPCA'], model='additive', period=6)
decompose.plot()
plt.show()

# %% [markdown]
# Note o componente sazonal da inflação brasileira em passagens de um ano para o outro.

# %% [markdown]
# Por fim, abaixo segue o gráfico com valores do teste acf, para identificar a autocorrelação presente nos valores da série do IPCA.

# %%
plot_acf(target, max_lag=100)

# %% [markdown]
# Verificando então os valores do teste acf, considerando um número de defasagens de 100.

# %%
acf(df1.IPCA, nlags=100)

# %% [markdown]
# Dos resultados acima e do gráfico do teste acf, observamos que o IPCA apresenta maior autocorrelação serial estatísticamente significativa para os graus de defasagem de 1, 2 e 11.

# %% [markdown]
# ## Modelagem

# %% [markdown]
# Para construir os modelos de previsão e realizar as estimações, vamos utilizar uma rica biblioteca de séries temporais do python, chamada [darts](https://unit8co.github.io/darts/README.html). Nos debruçaremos sobre 8 modelos: 'NaiveDrift', 'ExponentialSmoothing', 'Prophet', 'AutoARIMA', 'RandomForest', 'LightGBMModel', 'CatBoostModel' e 'BATS'. Cada um deles tem sua especificidade, e é possível obter uma descrição mais detalhada de cada através do seguinte [link](https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878).

# %%
m_naive = NaiveDrift()
if is_seasonal:
    m_expon = ExponentialSmoothing(seasonal_periods=mseas)
else:
    m_expon = ExponentialSmoothing()

m_prophet = Prophet(country_holidays='BR', add_seasonalities= dict({
'name': 'fim_do_ano' , # (name of the seasonality component),
'seasonal_periods': 11 , # (nr of steps composing a season),
'fourier_order': 5,  # (number of Fourier components to use),
  # (a prior scale for this component),
'mode': 'multiplicative'  # ('additive' or 'multiplicative')
}))

# %%
y = np.asarray(target.pd_series())
# get order of first differencing: the higher of KPSS and ADF test results
n_kpss = pmd.arima.ndiffs(y, alpha=ALPHA, test='kpss', max_d=20)
n_adf = pmd.arima.ndiffs(y, alpha=ALPHA, test='adf', max_d=20)
n_diff = max(n_adf, n_kpss)
print(n_kpss, n_adf, n_diff)

# get order of seasonal differencing: the higher of OCSB and CH test results
n_ocsb = pmd.arima.OCSBTest(m=max(4, mseas)).estimate_seasonal_differencing_term(y)
n_ch = pmd.arima.CHTest(m=max(4, mseas)).estimate_seasonal_differencing_term(y)
ns_diff = max(n_ocsb, n_ch, is_seasonal * 1)

# set up the ARIMA forecaster
m_arima = AutoARIMA(
    start_p=1, d=n_diff, start_q=1,
    max_p=4, max_d=n_diff, max_q=4,
    start_P=0, D=ns_diff, start_Q=0, m=max(4, mseas), seasonal=is_seasonal,
    max_P=3, max_D=1, max_Q=3,
    max_order=5,                       # p+q+p+Q <= max_order
    stationary=False, 
    information_criterion="bic", alpha=ALPHA, 
    test="kpss", seasonal_test="ocsb",
    stepwise=True, 
    suppress_warnings=True, error_action="trace", trace=True, with_intercept="auto" , forecast_horizon=1)

# %%
m_rf= RandomForest(lags=[-1,-2, -11], 
                    lags_past_covariates=[-1], 
                    lags_future_covariates=[0], 
                    output_chunk_length=1, 
                    add_encoders=None, 
                    n_estimators=1000, 
                    max_depth=6)

# %%
m_lgbm = LightGBMModel(lags=[-1,-2, -11], lags_past_covariates=[-1], 
                    lags_future_covariates=[0], learning_rate=0.05, max_depth=6)

# %%
m_catboost = CatBoostModel(lags=[-1,-2, -11], lags_past_covariates=[-1], 
                    lags_future_covariates=[0], learning_rate=0.05, max_depth=6)

# %%
m_bats = BATS(seasonal_periods=[11])

# %% [markdown]
# Agregando todos os modelos num único objeto.

# %%
models = [m_arima, m_prophet, m_rf, m_catboost, m_lgbm, m_bats, m_naive, m_expon]

# %% [markdown]
# ### Avaliação dos Modelos

# %% [markdown]
# Definindo uma função de avaliação dos modelos, utilizando as seguintes métricas de desempenho: Erro Percentual Absoluto Médio (MAPE - Mean Absolut Percentage Error), Erro Absoluto Médio (MAE - Mean Absolut Error), R quadrado (R Squared), Raiz Quadrática Média do Erro Logarítmico (RMSLE - Root Mean Squared Log Error) e Tempo da Estimação (Time).

# %%
def eval_model(model):
    t_start =  time.perf_counter()
    print("beginning: " + str(model))

    if model in [m_rf, m_lgbm]:
    # fit the model and compute predictions
        res = model.fit(train, past_covariates=pcov_tr, future_covariates=fcov_tr)
        forecast = model.predict(len(val), past_covariates=pcov_val, future_covariates=fcov_val)
    elif model in [m_catboost]:
        res = model.fit(train, past_covariates=pcov_tr, future_covariates=fcov_tr)
        forecast = model.predict(len(val), past_covariates=pcov_val, future_covariates=fcov_val)
    else:
        res = model.fit(train)
        forecast = model.predict(len(val))
    # for naive forecast, concatenate seasonal fc with drift fc
    if model == m_naive:
        if is_seasonal:
            fc_drift = forecast
            modelS = NaiveSeasonal(K=mseas)
            modelS.fit(train)
            fc_seas = modelS.predict(len(val))
            forecast = fc_drift + fc_seas - train.last_value()


    # compute accuracy metrics and processing time
    res_mape = mape(val, forecast)
    res_mae = mae(val, forecast)
    res_r2 = r2_score(val, forecast)
    res_rmsle = rmsle(val, forecast)
    res_time = time.perf_counter() - t_start
    res_accuracy = {"MAPE":res_mape, "MAE":res_mae, "R squared":-res_r2, "RMSLE":res_rmsle, "time":res_time}

    results = [forecast, res_accuracy]
    print("completed: " + str(model) + ":" + str(res_time) + "sec")
    print(res_accuracy)
    return results

# %% [markdown]
# Utilizando então um loop para passar todos os modelos na função de avaliação.

# %%
model_predictions = [eval_model(model) for model in models]

# %% [markdown]
# Reproduzindo seus resultados numa tabela, criando uma função para isso.

# %%
df_acc = pd.DataFrame.from_dict(model_predictions[0][1], orient="index")
df_acc.columns = [str(models[0])]

for i, m in enumerate(models):
    if i > 0: 
        df_a = pd.DataFrame.from_dict(model_predictions[i][1], orient="index")
        df_a.columns = [str(m)]
        df_acc = pd.concat([df_acc, df_a], axis=1)
    i +=1

pd.set_option("display.precision",3)
df_acc.style.highlight_min(color="lightgreen", axis=1).highlight_max(color="grey", axis=1)

# %% [markdown]
# E criando uma representação gráfica do desempenho de cada um, como feito abaixo, podemos enxergar de maneira palpável quais modelos tiveram um melhor desempenho.

# %%
pairs = math.ceil(len(models)/2)                    # how many rows of charts
fig, ax = plt.subplots(pairs, 2, figsize=(20, 5 * pairs))
ax = ax.ravel()

for i,m in enumerate(models):
        target.plot(label="actual", ax=ax[i])
        model_predictions[i][0].plot(label="prediction: "+str(m), ax=ax[i])
        
        mape_model =  model_predictions[i][1]["MAPE"]
        time_model =  model_predictions[i][1]["time"]
        ax[i].set_title("\n\n" + str(m) + ": MAPE {:.1f}%".format(mape_model) + " - time {:.2f}sec".format(time_model))

        ax[i].set_xlabel("")
        ax[i].legend()

# %% [markdown]
# Como podemos ver dos outputs gerados acima, os 4 melhores modelos foram, em crescente de desempenho:
# 
# 1. RandomForest;
# 2. LGBM;
# 3. CatBoost;
# 4. Prophet.

# %% [markdown]
# Realizando então um 'backtest' para estes 4 melhores modelos, obtemos:

# %%
m_rf_historical = m_rf.historical_forecasts(
    target, past_covariates=pcovariates, future_covariates=fcovariates, start=0.6, forecast_horizon=1, verbose=True
)

target.plot(label="data")
m_rf_historical.plot(label="backtest 4-years ahead forecast (rf)")
print("MAPE = {:.2f}%".format(mape(m_rf_historical, target)))

# %%
m_lgbm_historical = m_lgbm.historical_forecasts(
    target, past_covariates=pcovariates, future_covariates=fcovariates, start=0.60, forecast_horizon=1, verbose=True
)

target.plot(label="data")
m_lgbm_historical.plot(label="backtest 4-years ahead forecast (lgbm)")
print("MAPE = {:.2f}%".format(mape(m_lgbm_historical, target)))

# %%
m_catboost_historical = m_catboost.historical_forecasts(
    target, past_covariates=pcovariates, future_covariates=fcovariates, start=0.6, forecast_horizon=1, verbose=True
)

target.plot(label="data")
m_catboost_historical.plot(label="backtest 4-years ahead forecast (catboost)")
print("MAPE = {:.2f}%".format(mape(m_catboost_historical, target)))

# %%
m_prophet_historical = m_prophet.historical_forecasts(
    target, past_covariates=pcovariates, future_covariates=fcovariates, start=0.6, forecast_horizon=1, verbose=True
)

target.plot(label="data")
m_prophet_historical.plot(label="backtest 4-years ahead forecast (prophet)")
print("MAPE = {:.2f}%".format(mape(m_prophet_historical, target)))

# %% [markdown]
# Apesar de ter se saido melhor na avaliação de desempenho, o modelo Random Forest apresentou o 3º menor MAPE, entre os 4 selecionados para o 'backtest'. O melhor modelo neste caso foi o CatBoost, com um MAPE de 21,55%.

# %% [markdown]
# Por fim, gerando as estatísticas descritivas dos residuos destes melhores modelos.

# %%
residual_rf =  pd.Series(np.abs(target.pd_series()['2011-01-31':] - m_rf_historical.pd_series()))
pd.DataFrame(residual_rf.describe()).rename(columns={0:' Estatisticas dos Erros - RF'})

# %%
residual_lgbm =  pd.Series(np.abs(target.pd_series()['2011-01-31':] - m_lgbm_historical.pd_series()))
pd.DataFrame(residual_lgbm.describe()).rename(columns={0:' Estatisticas dos Erros - LGBM'})

# %%
residual_catboost =  pd.Series(np.abs(target.pd_series()['2011-01-31':] - m_catboost_historical.pd_series()))
pd.DataFrame(residual_catboost.describe()).rename(columns={0:' Estatisticas dos Erros - CATBOOST'})

# %%
residual_prophet =  pd.Series(np.abs(target.pd_series()['2011-01-31':] - m_prophet_historical.pd_series()))
pd.DataFrame(residual_prophet.describe()).rename(columns={0:' Estatisticas dos Erros - PROPHET'})

# %% [markdown]
# ## Referências

# %% [markdown]
# * GARCIA, Márcio GP; MEDEIROS, Marcelo C.; VASCONCELOS, Gabriel FR. Real-time inflation forecasting with high-dimensional models: The case of Brazil. International Journal of Forecasting, v. 33, n. 3, p. 679-693, 2017.


