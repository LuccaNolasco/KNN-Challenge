import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
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

# Reencontrando o melhor K para a matriz de confusão
melhor_k = valores_k[acuracia_k.index(max(acuracia_k))]

knn = KNeighborsClassifier(n_neighbors = melhor_k)
knn.fit(X,Y)
previsoesFinais = knn.predict(X_teste)

#Criando e Exibindo a matriz de confusão
matrizConfusao = confusion_matrix(y_teste, previsoesFinais)
matrizConfusaoDisplay = ConfusionMatrixDisplay(matrizConfusao,display_labels=[0,1])

#Criando os subplots pois quero ambos num lugar só
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plotando o gráfico de acurácia
ax[0].plot(valores_k, acuracia_k, marker='o')
ax[0].set_title('Acurácia do KNN para Diferentes Valores de K')
ax[0].set_xlabel('Número de Vizinhos (K)')
ax[0].set_ylabel('Acurácia')
ax[0].set_xticks(valores_k)
ax[0].grid(True)

# Plotando a matriz de confusão
matrizConfusaoDisplay.plot(ax=ax[1], cmap='plasma', values_format='d')
ax[1].set_title(f'Matriz de Confusão (K = {melhor_k})')

plt.tight_layout()


precisao = precision_score(y_teste, previsoesFinais)
recall = recall_score(y_teste, previsoesFinais)
f1 = f1_score(y_teste, previsoesFinais)

print(f"\nPrecisão para K = {melhor_k}: {precisao:.4f}\n"
      f"Recall para K = {melhor_k}: {recall:.4f}\n"
      f"F1 Score para K = {melhor_k}: {f1:.4f}")

print(f"\nConclusões: Apesar de certos valores posteriores possuírem maior acurácia que seus antecessores,\n"
      f"no geral a acurácia tende a cair à medida que o valor de K aumenta. Os valores para K= 3 e K = 5\n"
      f"mostraram-se extremamente semelhantes, então é válido usar K=3 para reduzir o custo computacional.\n"
      f"A precisão indica cerca de 94% de verdadeiros positivos previstos, creio que possa haver uma melhora.\n"
      f"O recall de cerca de 98% é bom. O modelo indentificou corretamente quase todas as instâncias de petróleo\n"
      f"Um bom valor da média harmõnica de 96.5% mostra que é equilibrado,")

plt.show()