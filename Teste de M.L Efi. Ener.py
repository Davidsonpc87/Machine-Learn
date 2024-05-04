# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregamento dos dados
data = pd.read_csv('dados_climatizacao.csv')  # Substitua 'dados_climatizacao.csv' pelo nome do seu arquivo CSV

# Visualização dos primeiros registros dos dados
print(data.head())

# Divisão dos dados em variáveis de entrada (features) e variável de saída (target)
X = data.drop('temperatura', axis=1)  # Considerando 'temperatura' como variável alvo
y = data['temperatura']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões no conjunto de teste
predictions = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Visualização dos resultados
plt.scatter(y_test, predictions)
plt.xlabel('Temperatura Real')
plt.ylabel('Temperatura Prevista')
plt.title('Comparação entre Temperatura Real e Prevista')
plt.show()
