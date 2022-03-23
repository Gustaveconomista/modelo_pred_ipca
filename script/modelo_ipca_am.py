# %% [markdown]
# # Modelo Preditivo de Inflação - AM

# %% [markdown]
# ## Instalação e Importe de Pacotes

# %%
#!pip install python-bcb

# %%
#!pip install pmdarima

# %%
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from pandas.io.formats import style
import plotly.express as px
from statsmodels.tsa.vector_ar.vecm import *
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
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
%matplotlib notebook

# %% [markdown]
# ## Coleta dos Dados

# %% [markdown]
# Vamos importar nosso dataset inteiro com o pacote `bcb`, que utiliza a API do BCB para coletar dados de diversas séries temporais.

# %%
df = sgs.get({'IPCA': 433, 'IGP': 189, 'IPC': 191, 'INCC': 192, 'IPA': 225, 'INPC': 188, 'sinapi': 7495, 'Selic': 4390, 'PIB': 4380, 'EnergiaElet': 1406,
             'ProdAço': 28546, 'BM': 27840, 'Cambio': 20360, 'CambioReal': 11752, 'IPA_OGInd': 7459, 'BalComerc': 22704, 'DivSetPub': 2053}, start='1995-01-01', end='2021-12-01')
df

# %% [markdown]
# Com o comando 'shape' podemos ver a dimensão de nosso dataset, onde 'n' representa o nº de observações e 'p' o nº de variáveis.

# %%
df.shape

# %% [markdown]
# Ou seja, nossa base tem 324 observações e 17 variáveis.

# %% [markdown]
# ## Tratamento dos Dados

# %% [markdown]
# Vamos trabalhar com 3 tipos de modelos preditivos, tendo cada um deles suas singularidades, e demandando consequentemente um tratamento especifíco da base de dados. Assim, vamos dividi-lá em 3 sub datasets, mas antes de tudo, vamos verificar as estatísticas descritivas de nossas séries.

# %%
df.describe()

# %% [markdown]
# ### 1º Sub dataset

# %%
df1 = df['IPCA']
df1 = pd.DataFrame(df1)
df1['Date'] = df1.index
df1

# %% [markdown]
# Com esta sub base vamos utilizar os modelos univariados auto-regressivos e de médias móveis (AR, MA, ARMA e ARIMA), por isso apenas utilizaremos a nossa série do IPCA. Ainda, para realizar a previsão com o modelo ARIMA precisamos dividi-lá em bases de treinamento (train) e de teste (test).

# %%
train = df1[df1['Date'] <= '2020-09-01']
train = train.rename(columns={'IPCA': "train"})
del train['Date']
test = df1[df1['Date'] >= '2020-09-01']
test = test.rename(columns={'IPCA': "test"})
del test['Date']

# %% [markdown]
# ### 2º Sub dataset

# %% [markdown]
# Para os modelos multivariados, vamos trabalhar com a série do IPCA e as séries da taxa Selic e da taxa de Câmbio, que serão nossos regressores:

# %%
df2 = df.loc[:, ['IPCA', 'Selic', 'Cambio']]
df2

# %% [markdown]
# Ainda, vamos separar um dataset auxiliar com a 1ª diferença destas séries, caso elas não sejam estacionárias.

# %%
df_1ªdifference = df2.diff().dropna()

# %% [markdown]
# ### 3º Sub dataset

# %% [markdown]
# Por último, montaremos um modelo que faz uso de técnicas avançadas de deep learning, então precisaremos converter nosso DataFrame principal em array:

# %%
df3 = df.values

# %% [markdown]
# E também vamos precisar separar nossa base em grupos de tratamento e de teste:

# %%
n = df.shape[0]
p = df.shape[1]

# %%
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n

# %%
df_train = df3[np.arange(train_start, train_end), :]
df_test = df3[np.arange(test_start, test_end), :]

# %% [markdown]
# ## Visualização dos Dados

# %% [markdown]
# Vamos plotar nossa série do IPCA num gráfico para ter uma identificação visual de como ela se comporta.

# %%
plt.figure(figsize=(9, 5))
plt.plot(df.index, df['IPCA'], color='tab:blue')
plt.gca().set(title='Monthly IPCA from 1995 to 2021', xlabel='Date', ylabel='IPCA')
plt.show()

# %% [markdown]
# Estatísticas ricas em informações sobre nossa série são a média móvel e o desvio padrão móvel, que captam a dinâmica da média e da variação da série ao longo do tempo.

# %%
rolling_mean = df['IPCA'].rolling(3).mean()
rolling_std = df['IPCA'].rolling(3).std()

# %% [markdown]
# Podemos plota-lás em um gráfico, junto a nossa série do IPCA:

# %%
plt.figure(figsize=(9, 5))
plt.plot(df['IPCA'], color="blue", label="Monthly IPCA from 1995 to 2021")
plt.plot(rolling_mean, color="red", label="Rolling Mean in IPCA")
plt.plot(rolling_std, color="black",
         label="Rolling Standard Deviation in IPCA")
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
decompose = seasonal_decompose(df['IPCA'], model='additive', period=6)
decompose.plot()
plt.show()

# %% [markdown]
# Note o componente sazonal da inflação brasileira em passagens de um ano para o outro.

# %% [markdown]
# ## Modelagem

# %% [markdown]
# ### Estacionariedade

# %% [markdown]
# Como 1º passo para estimação dos modelos preditivos, vamos verificar se nossas séries são estacionárias.
# Para isso, vamos construir uma função para realizar o teste de Dickey-Fuller aumentado para estacionariedade.

# %%


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Função que executa o Teste Dickey-Fuller Aumentado para verificar a estacionaridade de determinada série e
    retorna um relatório"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(
        r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']
    def adjust(val, length=6): return str(val).ljust(length)

    # Print Summary
    print(f'    Teste Dickey-Fuller aumentado em "{name}"', "\n   ", '-'*47)
    print(f' Hipótese Nula: A série têm raiz unitária. Não é Estacionária.')
    print(f' Nível de significância    = {signif}')
    print(f' Estatística de teste        = {output["test_statistic"]}')
    print(f' Nº de lags escolhidos       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f' Valor crítico {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Valor = {p_value}. Rejeitando a Hipótese Nula.")
        print(f" => A série é estacionária.")
    else:
        print(
            f" => P-Valor = {p_value}. Evidência fraca para rejeitar a hipótese nula.")
        print(f" => A série é não estacionária.")

# %% [markdown]
# Verificando então a estacionariedade das nossas séries do IPCA, da Selic e do Câmbio, que são das quais precisaremos para os modelos univariados e multivariados.


# %%
for name, column in df2.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# %% [markdown]
# Como podemos ver acima, apenas a série do IPCA apresentou estacionáriedade em nível. Para corrigir este problema, podemos recorrer a 1ª diferença das respectivas séries. Vamos então utilizar o dataset auxiliar 'df_1ªdifference' para isso:

# %%
for name, column in df_1ªdifference.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# %% [markdown]
# Tirando a 1ª diferença das séries da Selic e do Câmbio as tornamos estacionárias, possibilindo então utilizar o modelo multivariado VAR.

# %% [markdown]
# ### Autocorrelação

# %% [markdown]
# Outra informação importante sobre nossa série de interesse diz respeito a seu grau de autocorrelação, dado que a maior parte dos modelos de séries temporais utilizam os valores passados das séries para explicar seus valores futuros.
# Para verificar isto, vamos utlizar a função de autocorrelação:

# %%
autocorrelation_lag1 = df1['IPCA'].autocorr(lag=1)
print("Lag de 1 mês: ", autocorrelation_lag1)
autocorrelation_lag3 = df1['IPCA'].autocorr(lag=3)
print("Lag de 3 meses: ", autocorrelation_lag3)
autocorrelation_lag6 = df1['IPCA'].autocorr(lag=6)
print("Lag de 6 meses: ", autocorrelation_lag6)
autocorrelation_lag9 = df1['IPCA'].autocorr(lag=9)
print("Lag de 9 meses: ", autocorrelation_lag9)

# %% [markdown]
# Dos resultados acima podemos ver que a maior autocorrelação entre as observações de nossa série do IPCA se dá com lag de 1 mês.

# %% [markdown]
# ## Previsão/Estimação utilizando modelos univariados

# %% [markdown]
# Começaremos nossa estimação com os modelos univariados, que consideram apenas os movimentos passados da série de interesse para estimar seus valores futuros.

# %% [markdown]
# ### MA

# %% [markdown]
# Em modelos de médias móveis, temos um fator fixo, que é a média da série, e adicionalmente temos um componente residual variante com o tempo.

# %%
model = ARIMA(df1['IPCA'], order=(0, 0, 1), freq='MS')
model_fit = model.fit()
yhat1 = model_fit.predict(len(df1['IPCA']), len(df1['IPCA']))
print(yhat1)

# %% [markdown]
# Pela estimação de médias móveis, nosso y estimado pode ser visto acima, assumindo um valor de 0,589024.

# %% [markdown]
# ### AR

# %% [markdown]
# Os processos Autorregressivos consideram única e exclusivamente os valores passados da série como regressores dos valores presente e futuro.

# %%
model = AutoReg(df1['IPCA'], lags=1)
model_fit = model.fit()
yhat2 = model_fit.predict(len(df1['IPCA']), len(df1['IPCA']))
print(yhat2)

# %% [markdown]
# Acima temos o valor médio estimado do IPCA com base nos valores passados em um modelo AR de ordem 1.

# %% [markdown]
# ### ARMA

# %% [markdown]
# Já o modelo ARMA, intuitivamente, considera tanto os valores passados da série quanto suas médias móveis, sendo uma junção dos dois modelos vistos acima.

# %%
model = ARIMA(df1['IPCA'], order=(2, 0, 1), freq='MS')
model_fit = model.fit()
yhat3 = model_fit.predict(len(df1['IPCA']), len(df1['IPCA']))
print(yhat3)

# %% [markdown]
# ### ARIMA

# %% [markdown]
# Para finalizar o grupo de modelos univariados, temos o modelo ARIMA, que considera aquelas séries que são integradas, ou seja, com tendência estocástica, e que possuem erros estacionários.

# %%
model = ARIMA(df1['IPCA'], order=(1, 1, 1))
model_fit = model.fit()
yhat4 = model_fit.predict(len(df1['IPCA']), len(df1['IPCA']), typ='levels')
print(yhat4)

# %% [markdown]
# Novamente, acima temos o valor estimado de nossa variável de interesse.

# %% [markdown]
# Vamos agora tentar prever o IPCA utilizando o modelo ARIMA.

# %%
model = auto_arima(train, seasonal=True, trace=True,
                   error_action='ignore', suppress_warnings=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

# %% [markdown]
# No gráfico abaixo temos os grupos de treinamento (azul), de teste (vermelho) e a previsão (verde).

# %%
plt.figure(figsize=(9, 5))
plt.plot(train, color="blue", label="Train")
plt.plot(test, color="red", label="Test")
plt.plot(forecast, color="green", label="Prediction")
plt.title("IPCA Prediction")
plt.legend(loc="best")

# %% [markdown]
# Como se pode ver, modelos univariados tem pouca acurácia para prever a inflação no Brasil medida pelo IPCA. Outra prova disso é o alto valor do erro quadrado médio neste modelo:

# %%
rms = sqrt(mean_squared_error(test, forecast))
print("RMSE: ", rms)

# %% [markdown]
# ## Previsão/Estimação utilizando modelos multivariados

# %% [markdown]
# Começaremos com o modelo VAR (Vector AutoRegressive), que abre espaço para a possibilidade de outras séries influenciarem nossa variável de interesse, mas que não podem ser não-estacionárias. Trabalharemos com 3 séries aqui, o IPCA, a Taxa Selic e a Taxa de Câmbio. Abaixo segue uma pré-visualização delas:

# %%
fig, axes = plt.subplots(nrows=3, ncols=1, dpi=120, figsize=(8, 5))
for i, ax in enumerate(axes.flatten()):
    data = df2[df2.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    ax.set_title(df2.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# %% [markdown]
# ### VAR

# %% [markdown]
# Como já visto acima, as séries da Selic e do Câmbio não são estacionárias em nível, impossibilitando a previsão com o modelo VAR. Nossa alternativa então foi tirar a 1ª diferença de tais séries para alcançar a condição de estacionariedade. Vamos agora separar nossos datasets auxiliares das primeiras diferenças em grupos de treinamento e de teste:

# %%
test_obs = 12
train1 = df_1ªdifference[:-test_obs]
test1 = df_1ªdifference[-test_obs:]

# %% [markdown]
# Verificando a Causalidade em nossas variáveis usando o Teste de Causalidade de Granger:

# %%
maxlag = 12


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Função que verifica a causalidade de Granger de todas as combinações possíveis da série temporal.
    As linhas são a variável de resposta, as colunas são preditores. Os valores da tabela
    são os p-valores. P-valores menores que o nível de significância (0,05), implicam na rejeição da
    Hipótese Nula de que os coeficientes dos valores passados correspondentes são
    zero, ou seja, que X não causa Y.

    data      : Dataframe contendo as séries de interesse.
    variables : Lista contendo os nomes das variáveis.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))),
                      columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(
                data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4)
                        for i in range(maxlag)]
            if verbose:
                print(f'Y = {r}, X = {c}, P-Valores = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# %%
grangers_causation_matrix(df_1ªdifference, variables=df2.columns)

# %% [markdown]
# Pela primeira linha da tabela é possível ver que nossos regressores Granger causam o IPCA, ou seja, temos forte evidência de que a Selic e o Câmbio apresentam uma relação de causalidade com nossa variável de interesse.

# %% [markdown]
# Vamos agora para a especificação do modelo, aumentando iterativamente sua ordem de defasagem e verificando os valores dos critérios de informação.

# %%
model = VAR(df_1ªdifference)
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

# %% [markdown]
# A tabela acima indica que o melhor nº de defasagem para especificação do modelo é 2.
# Estimando então o modelo:

# %%
model = VAR(train1)
model_fitted = model.fit(2)
model_fitted.summary()

# %% [markdown]
# Preparando o modelo para previsão:

# %%
forecast_input = df_1ªdifference.values[-model_fitted.k_ar:]
forecast_input

# %%
fc = model_fitted.forecast(y=forecast_input, steps=test_obs)
df_forecast = pd.DataFrame(
    fc, index=df2.index[-test_obs:], columns=df2.columns + '_2d')
df_forecast

# %%


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Função que reverte a diferenciação para obter a previsão na escala original."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1] -
                                     df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_prev'] = df_train[col].iloc[-1] + \
            df_fc[str(col)+'_1d'].cumsum()
    return df_fc


# %%
df_results = invert_transformation(train1, df_forecast, second_diff=True)
df_results.loc[:, ['IPCA_prev', 'Selic_prev', 'Cambio_prev']]

# %% [markdown]
# Plotando os resultados da previsão junto aos valores reais:

# %%
fig, axes = plt.subplots(nrows=3, ncols=1, dpi=150, figsize=(5, 5))
for i, (col, ax) in enumerate(zip(df2.columns, axes.flatten())):
    df_results[col+'_prev'].plot(legend=True,
                                 ax=ax).autoscale(axis='x', tight=True)
    test1[col][-test_obs:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Previsão vs Real")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# %% [markdown]
# Como visto acima, o modelo VAR também apresenta suas limitações relativas a acurácia da previsão.

# %% [markdown]
# Tratemos agora de um modelo mais completo, chamado modelo Vetor de Correção de Erros, ou VECM.

# %% [markdown]
# ### VECM

# %% [markdown]
# O trunfo do VECM está no fato de ele corrigir um dos principais problemas do VAR. Resumidamente, se usamos um VAR com variáveis não estacionárias, mas com suas diferenças, podemos estar omitindo variáveis relevantes ao modelo.

# %% [markdown]
# Nosso 1º passo será realizar o Teste de Cointegração de Johansen, para verificar se nossas séries são cointegradas.
# Vamos criar uma função para isto:

# %%


def cointegration_test(df, alpha=0.05):
    """Função que realize o teste de cointegração de Johanson e devolve um resumo dele"""
    out = coint_johansen(df, 0, 2)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length=6): return str(val).ljust(length)

    # Resumo
    print('Nome   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9),
              ">", adjust(cvt, 8), ' =>  ', trace > cvt)


# %%
cointegration_test(df_1ªdifference)

# %% [markdown]
# A um nível de significancia de 5%, temos fortes evidências de que as 3 séries são cointegradas, indicando uma correlação de longo prazo entre elas.

# %% [markdown]
# O próximo passo diz respeito a identificação do modelo, onde precisamos realizar o teste de verificação do rank de nossas séries.

# %%
rank_test = select_coint_rank(df_1ªdifference, 0, 2, method="trace",
                              signif=0.05)
rank_test.rank

# %%
rank_test.summary()

# %%
print(rank_test)

# %% [markdown]
# O valor do rank então é 3.
# Assim, estamos aptos a estimar o modelo:

# %%
model_vecm = VECM(df_1ªdifference, deterministic="ci", seasons=4,
                  k_ar_diff=model_fitted.k_ar,  # =3
                  coint_rank=rank_test.rank)

# %%
vecm_result = model_vecm.fit()

# %%
vecm_result.summary()

# %%
vecm_result.alpha

# %%
vecm_result.stderr_alpha

# %%
vecm_result.predict(steps=5)

# %%
vecm_result.predict(steps=5, alpha=0.05)
for text, vaĺues in zip(("forecast", "lower", "upper"), vecm_result.predict(steps=5, alpha=0.05)):
    print(text+":", vaĺues, sep="\n")

# %% [markdown]
# Por fim, podemos plotar os resultados da estimação, como se segue abaixo:

# %%
vecm_plot = vecm_result.plot_forecast(steps=5, n_last_obs=12)

# %% [markdown]
# ## Previsão/Estimação usando Deep Learning

# %% [markdown]
# Conforme dito no início, para este modelo usando Deep Learning, precisamos que nosso dataset seja convertido para array. Além disso, aqui estaremos trabalhando com nosso dataset principal completo, ou seja, com a série do IPCA mais 16 outras variáveis que podem ter um poder de influência sobre ela.

# %%
df3

# %%
df_train

# %%
df_test

# %% [markdown]
# ### Escalonamento dos dados

# %% [markdown]
# Nosso modelo de Dl fará uso de técnicas de redes neurais. Desta maneira, como quase todas as redes neurais se beneficiam do reescalonamento dos inputs (e algumas vezes dos outputs), e como nossos dados não estão na mesma escala, vamos utilizar a função 'MinMaxScaler' do pacote de mesmo nome para reestrutura-los numa escala de -1 a 1.

# %%
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df_train)

# %%
scaler

# %%
df_train = scaler.transform(df_train)
df_test = scaler.transform(df_test)

# %%
df_train

# %% [markdown]
# Depois disso, precisaremos separar os inputs dos outputs.

# %%
x_train = df_train[:, 1:]
y_train = df_train[:, 0]
x_test = df_test[:, 1:]
y_test = df_test[:, 0]

# %%
x_train

# %%
y_train

# %%
x_test

# %%
y_test

# %% [markdown]
# ### Construção do modelo

# %% [markdown]
# Vamos separar em um objeto o nº de variáveis em nossa base de treinamento.

# %%
n_vars = x_train.shape[1]
print(n_vars)

# %% [markdown]
# Um ponto importante a se notar é que iremos construir a rede neural sem utilizarmos o Keras. Inicialmente precisamos definir uma sessão:

# %%
net = tf.compat.v1.InteractiveSession()

# %%
net

# %% [markdown]
# Precisamos então construir as camadas com os neurônios e definir como os dados entrarão nelas.
# Para isso, precisamos primeiro criar nossos placeholders, que serão usados para armazenar os dados dos inputs e do output. Serão dois placeholders, X e Y, com X contendo os inputs da nossa rede (as variáveis que podem ter um poder de explicação sobre o IPCA, em T = t) e Y contendo o output (o IPCA em T = t+1).
# O shape dos placeholders corresponde a [None, n_vars], onde "None" significa que os inputs vem em uma matriz bi-dimensional e o output de um vetor uni-dimensional.

# %%
tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_vars])
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

# %%
X

# %%
Y

# %% [markdown]
# #### Neurônios:
# O modelo irá possuir quatro camadas (hidden layers), além da camada de entrada e da camada de saída. A primeira camada vai conter 32, e as camadas seguintes possuirão como tamanho a metade das camadas anteriores, sendo 16, 8 e 4, respectivamente. A redução no número de neurônios vai comprimindo a informação identificada em cada camada anterior.

# %%
n_neurons_1 = 32
n_neurons_2 = 16
n_neurons_3 = 8
n_neurons_4 = 4

# %% [markdown]
# #### Inicializadores:
# Inicializadores são utilizados para inicializar as variáveis da rede antes do treinamento. Como as redes neurais são treinadas utilizando técnicas de otimização numérica, a condição inicial do problema de otimização é crucial para se obter boas soluções para o problema. O TensorFlow possui diferentes inicializadores, mas aqui usaremos 2, sendo eles:
#
# **variance_scaling_initializer:** Constrói um inicializador que gera tensores sem reescalonar a variância. É sempre bom se pudermos manter a escala da variância dos inputs constante ao inicializarmos uma rede de forma que não exploda nem diminua ao chegarmos na camada final de nossa rede.
# Quando utilizamos uma distribuição normal as amostras serão obtidas da distribuição com média zero e variância,
#
# \begin{equation}
#     stddev = sqrt(scale / n)
# \end{equation}
#
# e com os seguintes parâmetros:
#
#         * n:
#             - Será o número de conexões no tensor de input, se mode = "fan_in";
#             - Será o número de conexões no tensor de output, se mode = "fan_out";
#             - Será a média de conexões no tensor de input e output, se mode = "fan_avg";
#
#         * distribution: Distribuição aleatória a ser utilizada ("normal", "uniforme", etc);
#
#         * scale: Fator de escalonamento (float positiva).
#
# Será usado como inicializador dos pesos.
#
# **zeros_initializer:** Inicializador que gera tensores inicializados em 0. Será usado como inicializador do viés.

# %%
sigma = 1
weight_initializer = tf.compat.v1.variance_scaling_initializer(
    mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.compat.v1.zeros_initializer()

# %%
weight_initializer

# %%
bias_initializer

# %% [markdown]
# #### Variáveis:

# %%
# Camada 1: Variáveis para pesos e viés
W_hidden_1 = tf.compat.v1.Variable(weight_initializer([n_vars, n_neurons_1]))
bias_hidden_1 = tf.compat.v1.Variable(bias_initializer([n_neurons_1]))

# Camada 2: Variáveis para pesos e viés
W_hidden_2 = tf.compat.v1.Variable(
    weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.compat.v1.Variable(bias_initializer([n_neurons_2]))

# Camada 3: Variáveis para pesos e viés
W_hidden_3 = tf.compat.v1.Variable(
    weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.compat.v1.Variable(bias_initializer([n_neurons_3]))

# Camada 4: Variáveis para pesos e viés
W_hidden_4 = tf.compat.v1.Variable(
    weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.compat.v1.Variable(bias_initializer([n_neurons_4]))

# Output weights
# Camada de Output: Variáveis para pesos e viés do output
W_out = tf.compat.v1.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.compat.v1.Variable(bias_initializer([1]))

# %% [markdown]
# ### Desenhando a Arquitetura da Rede Neural

# %% [markdown]
# Os placeholders (dados) e as variáveis (pesos e viés) precisam ser combinados em um sistema de multiplicação matricial sequencial.
# Também, as camadas de nossa rede serão transformadas por funções de ativação. Funções de ativação são elementos importantes de redes neurais uma vez que introduzem não linearidade ao sistema. Existem dezenas de funções de ativação, uma das mais comuns é a rectified linear unit (ReLU), que usaremos aqui nesta rede.
# Do tensorflow, usaremos ainda as seguintes funcionalidades:
#
#     * tf.nn: Encapsulador para operações com redes neurais primitivas (NN);
#     * tf.matmul: Multiplica a matriz A pela matriz B, produzindo A * B.

# %%
hidden_1 = tf.compat.v1.nn.relu(tf.compat.v1.add(
    tf.compat.v1.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.compat.v1.nn.relu(tf.compat.v1.add(
    tf.compat.v1.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.compat.v1.nn.relu(tf.compat.v1.add(
    tf.compat.v1.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.compat.v1.nn.relu(tf.compat.v1.add(
    tf.compat.v1.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# %% [markdown]
# #### Camada de output (transposta):

# %%
out = tf.compat.v1.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# %%
out

# %% [markdown]
# #### Função Custo (Perda):
# A função custo ou função perda é utilizada para gerar uma medida de desvio entre a previsão da rede e o valor real observado no treinamento. Para problemas de regressão o erro quadrado médio (MSE) é bastante utilizado.

# %%
mse = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(out, Y))

# %%
mse

# %% [markdown]
# #### Otimizador:
# O otimizador se encarrega dos cálculos necessários para adaptar os pesos e viés da rede durante o treinamento. Esses cálculos utilizam gradientes que indicam a direção em que os pesos e viés devem ser modificados de forma a minimizar a função custo. O desenvolvimento de otimizadores estáveis e rápidos é uma area em que há grande pesquisa atualmente.
# Utilizaremos aqui o otimizador ADAM (Adaptive Moment Estimation).

# %%
opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)

# %%
opt

# %% [markdown]
# #### Treinamento:
# Após definirmos placeholders, variáveis, inicializadores, custo e otimizadores, o modelo precisa ser treinado.

# %%
net.run(tf.compat.v1.global_variables_initializer())

# %% [markdown]
# Em paralelo a isso, criaremos um gráfico interativo com nossas linhas de teste (azul) e de treinamento (laranja), deslocando esta última 1/2 de modo a visualizar seu movimento em direção ao nível do valor real.

# %% [markdown]
# #### Rodar o modelo:
# Durante o treinamento do modelo amostras aleatórias de tamanho  𝑛=  batch_size serão retiradas da base de treinamento. Este procedimento continua até todos os batches serem apresentados a rede. Uma apresentação completa de todos os batches a rede é chamada de uma época.

# %%
epochs = 50
batch_size = 1
mse_train = []
mse_test = []

# %% [markdown]
# Como nosso nº de observações é relativamente baixo, utilizaremos uma grande quantidade de épocas (50) e o valor mínimo para o batch_size (1).

# %%
np.mean(y_test)

# %%
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
fig = plt.gcf()
plt.show()

# %%
for e in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
        # Show progress
        if np.mod(i, 10) == 0:  # Return element-wise remainder of division
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: x_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: x_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: x_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            fig.canvas.draw()
            plt.pause(0.05)

# %% [markdown]
# Podemos ver no gráfico acima que o modelo de Dl é o que melhor se ajusta aos valores reais, tendo a maior acurácia entre todos mostrados até aqui.

# %% [markdown]
# ## Referências

# %% [markdown]
# * BUENO, Rodrigo De Losso da Silveira. Econometria de séries temporais. [S.l: s.n.], 2012.

# %% [markdown]
# * Pacheco, C. A. R., & Pereira, N. S. (2018). Deep Learning Conceitos e Utilização nas Diversas Áreas do Conhecimento. Revista Ada Lovelace, 2, 34–49. Recuperado de http://anais.unievangelica.edu.br/index.php/adalovelace/article/view/4132
