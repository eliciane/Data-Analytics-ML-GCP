
import pandas as pd


query = "SELECT * FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` WHERE date BETWEEN '20170101' and '20170131'"

df = pd.read_gbq(query=query, project_id='essential-truth-396022', dialect='standard')

df.hits[0] # que temos dentro desta lista # toda vez que tiver um hit de um produto vamos abrir uma tabela...

df.hits[365] #lista saber quais produtos o usuário teve interação

df[['hits']]

df['hits'][365][0]['product']


# +
produtos_sessao =[] # criar uma lista vazia
precos_sessao = []

for linha in df.hits: #iterar dentro de cada linha do df.hits
    produtos_hit = []  # verficar quais produtos que foram iterados dentro desses hits. Criar uma lista separada para armezenar todos os produtos.
    # A iteração faz com que pegue todos os produtos desta lista seja colacada numa lista maior, que abrimos inicialmente
    precos_hit = []
    for hit in linha: #iterar dentro de cada hit que está dentro desta sessão
        for produto in hit['product']: # estou acessando cada produto dentro de hit product
        # verficar quais produtos que foram iterados dentro desses hits. Criar uma lista separada para armezenar todos os produtos
            produtos_hit.append(produto['productSKU']) # fazer o apende de cada um desses produtos dntro da lista com todos os produtos
            precos_hit.append(produto['productPrice'])
    produtos_sessao.append(produtos_hit) #DENTRO DE PRODUTO SESSÃO VAMamos ter uma lista de cada produto que tiveram interações com hit
    precos_sessao.append(sum(precos_hit)) # uma soma de todos esses preços do hit
# -

len(produtos_sessao)

produtos_sessao[365] # objetivo pegar todos os produtos que o cara teve iteração na sessão

precos_sessao[365] # a soma de todos os produtos que ele teve iteracao

# qual o total de preços
# quais os produtos
# transformar a listas de produtos em informações 0 ou 1
s= pd.Series(produtos_sessao)
print(s)

# varias colunas com o valor de todos os produtos (sendo 0 ou 1)
from sklearn.preprocessing import MultiLabelBinarizer

mbl = MultiLabelBinarizer()

mbl.fit_transform(s) # todos que aparecem 1 é porque tiveram iteração com o produto

df_produtos = pd.DataFrame(mbl.fit_transform(s), columns= mbl.classes_, index=df.index)  # criar um dataframe com os nomes das colunas

df = df.join(df_produtos)
print(df)

# +
df['preco'] = precos_sessao #criar a coluna preço

df.head()
# -

visitas = df.drop(['hits', 'customDimensions'], axis=1) # excluir colunas que não serão usadas

visitas.head()

# abrir todas as variáveis do tipo dicionários em alguma coluna em novas variáveis
dicionarios = ['device', 'trafficSource', 'geoNetwork', 'totals']

import json

# +
# Transformando as chaves dos dicionários em novas colunas ; verificar as variáveis e excluir colunas
for coluna in dicionarios:
  visitas = visitas.join(
    pd.DataFrame([json.loads(json.dumps(linha))
      for linha in visitas[coluna]]) , rsuffix=('_' + coluna) )
visitas.drop( dicionarios, axis=1, inplace=True)

# Corrigindo o formato das colunas com valores quantitativos
totals = df.totals[0].keys()
totals = list(totals)
for coluna in totals:
  visitas[coluna] = pd.to_numeric(visitas[coluna])

# Limpando os dados
visitas.drop('adwordsClickInfo', axis=1, inplace=True)
# Remove as colunas cujo domínio só tem um elemento
coluna_na = []
for coluna in visitas.columns:
  print(str(coluna) + ': ' + str(len(visitas[coluna].unique())))
  if len( visitas[coluna].unique()) == 1:
    coluna_na.append(coluna)
visitas.drop( coluna_na, axis=1, inplace=True)
# -

visitas.head()

visitas = visitas.drop('transactions',axis=1)

visitas.totalTransactionRevenue = visitas.totalTransactionRevenue / 1000000

visitas.head()

# +
# Criando lista com variaveis quantitativas
quant = list(set(totals) - set(coluna_na) - set(['transactionRevenue','transactions']))

#Criando dataframe com os resultadas quantitativos das sessoes
visitas_totals = visitas.groupby('fullVisitorId',as_index=False)[quant].sum()

# Última visita
visitas_ultima = visitas.groupby('fullVisitorId',as_index=False)
visitas_ultima = visitas_ultima['visitNumber'].max()

# Combinação entre visitantes e visitas únicos
usuarios_visitas_unicos = visitas.drop_duplicates(subset=['fullVisitorId','visitNumber'])

# Dataframe usuários com todos os usuários unicos e sua última visita
usuarios = pd.merge(visitas_ultima,usuarios_visitas_unicos,left_on=['fullVisitorId','visitNumber'],
                    right_on=['fullVisitorId','visitNumber'],how='left')

# Primeira visita
visitas_primeira = visitas.groupby('fullVisitorId',as_index=False)['visitNumber'].min()
visitas_primeira.set_index('fullVisitorId',inplace=True)

# Dataframe usuários com todos os usuários unicos + sua última visita + primeira visita
usuarios = usuarios.join(visitas_primeira,how='left',on='fullVisitorId',rsuffix='_primeira')
usuarios = pd.merge(usuarios,usuarios_visitas_unicos,left_on=['fullVisitorId','visitNumber_primeira'],
                    right_on=['fullVisitorId','visitNumber'],how='left',suffixes=('_ultima','_primeira'))

# Dataframe usuários com todos os usuários unicos + sua última visita + primeira visita + somatório das colunas quant
usuarios = pd.merge(usuarios,visitas_totals,left_on=['fullVisitorId'],
                    right_on=['fullVisitorId'],how='left')

# Removendo totais
for i in quant:
    usuarios.drop(i+'_primeira',axis=1,inplace=True)
    usuarios.drop(i+'_ultima',axis=1,inplace=True)

# Calculando o tempo entre primeira e última visista
usuarios['tempo_visitas'] = usuarios.visitStartTime_ultima - usuarios.visitStartTime_primeira
# -

print(usuarios.head())

# +
#Removendo as colunas Ids
ids = ['fullVisitorId', 'visitId_ultima','visitId_primeira']
usuarios.drop(ids,axis=1,inplace=True)

#Criando uma variavel y com a coluna resposta
y = usuarios.totalTransactionRevenue.copy()
y.fillna(0,inplace=True)
y[y<0] = 0

#Criando uma variavel X com todas as variaveis menos a resposta
X = usuarios.drop('totalTransactionRevenue',axis=1)


#Transformando as variaveis qualitativas em numeros
from sklearn.preprocessing import LabelEncoder
quali = usuarios.dtypes[usuarios.dtypes == 'object'].keys()
for col in quali:
    lbl = LabelEncoder()
    lbl.fit(list(X[col].values.astype('str')))
    X[col] = lbl.transform(list(X[col].values.astype('str')))
# -

X.shape

X.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

X_train.shape

X_train.head()

X_train.isnull().sum()

y_train = y_train

y_train

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)

reg_predict = reg.predict(X_test)

reg_predict

from sklearn.metrics import mean_squared_error

import numpy as np

np.sqrt(mean_squared_error(y_test, reg_predict))

from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()

gb.fit(X_train,y_train)

gb_predict = gb.predict(X_test)

gb_predict[gb_predict < 0] = 0 #seleciona todos os casos qu tiveram o valor menor que zero e converte para zero. Não podemos ter clientes com compras negativas

np.sqrt(mean_squared_error(y_test, gb_predict))

resultados = pd.DataFrame() # criar o dataframe
resultados['revenue'] = y_test #quanto que o usuário gastou (que é o que estamos querendo prever)
resultados['predict'] = gb_predict
resultados['erro'] = gb_predict - y_test

resultados.head()

resultados[resultados.revenue > 0].head()

from sklearn.dummy import DummyRegressor

dr = DummyRegressor()  # aqui tento fazer um chute....ver se o modelo está melhor que a média....vamos usar dummies

dr.fit(X_train, y_train)

dr_predict = dr.predict(X_test)

np.sqrt(mean_squared_error(y_test, dr_predict)) # verificar se este erro está próximo do modelo anterior

# tentar mudar a pergunta e ver se podemos usar outro modelo
# ao invés tentar prever quando que o usuário gastou (revenue), tentar prever se ele gastou ou não gastou. Ou seja, transformar a variável em binária
y_train.values[y_train <= 0] = 0
y_train.values[y_train > 0] = 1
y_test.values[y_test <= 0] = 0
y_test.values[y_test > 0] = 1

from sklearn.ensemble import GradientBoostingClassifier

clf_gb = GradientBoostingClassifier()

clf_gb.fit(X_train, y_train)

clf_gb_predict = clf_gb.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,clf_gb_predict)

from sklearn.metrics import classification_report

print(classification_report(y_test,clf_gb_predict))

X_train['revenue'] = y_train

X_train.head()

X_train.head()

usuarios_com_gastos = X_train[X_train.revenue > 0]

usuarios_com_gastos.shape

usuarios_sem_gastos = X_train[X_train.revenue <= 0]

usuarios_sem_gastos.shape

from sklearn.utils import resample

usuarios_sem_gastos_ds = resample(usuarios_sem_gastos,
                                  replace=False, # aqui siginifica que não será resposta a linha excluida atomaticamente
                                  n_samples=30000, # estamos reduzindo para 30.000 usuários
                                  random_state=42)

usuarios_sem_gastos_ds.shape

X_train_ds = pd.concat([usuarios_sem_gastos_ds, usuarios_com_gastos])

X_train_ds.shape

y_train_ds = X_train_ds.revenue.copy()
X_train_ds.drop('revenue',axis=1,inplace=True)

clf_gb_ds = GradientBoostingClassifier(random_state=42)

clf_gb_ds.fit(X_train_ds, y_train_ds)

clf_gb_predict_ds = clf_gb_ds.predict(X_test)

confusion_matrix(y_test,clf_gb_predict_ds)

print(classification_report(y_test,clf_gb_predict_ds))

# !pip install --upgrade xgboost

from xgboost import XGBClassifier

xgb = XGBClassifier(seed=42,n_estimators=300)

X_train_ds = X_train_ds.loc[:,~X_train_ds.columns.duplicated()]

X_train_ds.shape

X_test = X_test.loc[:,~X_test.columns.duplicated()]

clf_xgb_predict_ds = xgb.predict(X_test)


