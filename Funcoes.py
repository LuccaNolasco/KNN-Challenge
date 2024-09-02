import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#Aqui deixei as funções que usei, para uma melhor visualização


#Funcao generica para exibir
def exibir(dataFrame, nome):
  plt.figure(figsize=(10, 6))
  
  #Definindo coordenadas
  plt.scatter(dataFrame['x'], dataFrame['y'], c=dataFrame['petroleo'].astype(int), cmap= 'bwr', alpha= 0.6)

  # Adicionando rótulos e título
  plt.xlabel('Coordenada X')
  plt.ylabel('Coordenada Y')
  titulo = 'Distribuição de Petróleo nas Coordenadas: ' + nome
  plt.title(titulo)
  plt.grid()

  # Mostrando o gráfico
  plt.show()

def testarK(testeNormalizado, limiteInferior, limiteSuperior, X, Y):
  valores_K = []
  acuracia_K = []
  for k in range (limiteInferior, limiteSuperior, 2):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X,Y)
    X_teste = testeNormalizado[['x','y']]
    y_teste = testeNormalizado['petroleo']
    previsoesIniciais = knn.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoesIniciais)
    valores_K.append(k)
    acuracia_K.append(acuracia)
    print(f"Acurácia das previsoes K = {k}: {acuracia:.5f}")
  
  return (valores_K, acuracia_K)