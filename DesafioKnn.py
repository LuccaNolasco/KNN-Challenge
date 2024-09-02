import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from Funcoes import *

""" 
Deixei algumas funções no arquivo "Funcoes". Outras, como as do KNN deixei 
aqui até mesmo para estudo próprio. Para seguir, é só ir fechando as imagens
e se atentar aos prints no terminal
"""

# 1 explorando os dados
dataFrame_treino = pd.read_csv("train.csv")
dataFrame_teste = pd.read_csv("test.csv")


# 2 exibindo os dados iniciais
exibir(dataFrame_teste, 'Teste')
exibir(dataFrame_treino, 'Treino')

# 3 preparando os dados

normalizador = MinMaxScaler() #Optei por uma normalização min max

treinoNormalizado = pd.DataFrame(normalizador.fit_transform(dataFrame_treino[['x','y']]), columns = ['x', 'y'])
treinoNormalizado['petroleo'] = dataFrame_treino['petroleo'].values

#Uso apenas transform para garantir que seja normalizado com os mesmos parametros
testeNormalizado = pd.DataFrame(normalizador.transform(dataFrame_teste[['x','y']]), columns = ['x', 'y'])
testeNormalizado['petroleo'] = dataFrame_teste['petroleo'].values



# 4 preparando o algoritmo e previsoes iniciais

# Aqui, dividi as características e as etiquetas
X = treinoNormalizado[['x', 'y']] # As 'features' : coordenadas
Y = treinoNormalizado['petroleo'] # As 'labels' : Tem ou não petroleo

#Escolhi K = 3 e treinei o modelo
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X,Y)

#As features do teste normalizado
X_teste = testeNormalizado[['x','y']]
y_teste = testeNormalizado['petroleo']

previsoesIniciais = knn.predict(X_teste)

acuracia_inicial = accuracy_score(y_teste, previsoesIniciais)
print(f"Acurácia das previsoes K = 3: {acuracia_inicial:.5f}")

# 5 testes para diferentes valores de k

valores_k, acuracia_k =testarK(testeNormalizado,3,30,X,Y) # Tive de passar X e Y como parametro para evitar imports circulares

#Criando um data frame para melhorar a exibição
tabela_acuracia = pd.DataFrame({'k': valores_k, 'Acurácia': acuracia_k})

#Aqui está a tabela
print(tabela_acuracia)

#Exibindo o gráfico
plt.figure(figsize=(10, 6))
plt.plot(valores_k, acuracia_k, marker='o')
plt.title('Acurácia do KNN para Diferentes Valores de K')
plt.xlabel('Número de Vizinhos (K)')
plt.ylabel('Acurácia')
plt.xticks(valores_k)  # Define os ticks do eixo x para serem os valores de K
plt.grid()
plt.show()

print(f"\nConclusões: Apesar de certos valores posteriores possuírem maior acurácia que seus antecessores,\n"
      f"no geral a acurácia tende a cair à medida que o valor de K aumenta. Os valores para K= 3 e K = 5\n"
      f"mostraram-se extremamente semelhantes, então é válido usar K=3 para reduzir o custo computacional.")


"""
Conclusões: Apesar de certos valores posteriores possuirem maior acurácia que seus antecessores, no geral a 
acurácia tende a cair à medida que o valor de K aumenta. Os valores para K= 3 e K = 5 mostraram-se extremamente
semelhantes, então é válido usar K=3 para reduzir o custo computacional
"""
