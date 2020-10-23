import numpy as np
import random
from scipy.spatial import distance

class KMeans():
    def __init__(self):
        pass
    
    def fit(self, X, k=2, dimensoes=2, epochs=30):
        self.X = X

        #Cria centroids iniciais
        centroids = np.array(np.zeros((1, dimensoes)))
        for i in range(0,k):
            centroids = np.vstack((centroids, np.random.permutation(X)[0]))
        centroids = centroids[1:,:]

        #Atribuição de centroids e recalculo de centroids
        epoch = 0
        old_centroids = np.array([])
        while not np.array_equal(centroids, old_centroids) and epoch < epochs: 
            #Atribuição de centroid a cada ponto
            y = np.array(np.zeros((1,dimensoes)))
            for x in X:
                min_distance = np.Inf
                centroid_mais_prox = centroids[0]
                for c in centroids:
                    this_distance = distance.euclidean(x, c)
                    if (this_distance < min_distance):
                        min_distance = this_distance
                        centroid_mais_prox = c
                y = np.vstack((y, np.array(centroid_mais_prox)))
            y = y[1:,:]

            #Atualiza centroides
            old_centroids = centroids.copy()
            for i in range(0, 1):
                neighbors = np.array(np.zeros((1,dimensoes)))
                for j in range(0,X.shape[0]):
                    if np.array_equal(y[j], c):
                        neighbors = np.vstack((neighbors, np.array(X[j])))
                neighbors = neighbors[1:,:]
                centroids[i] = np.mean(neighbors.copy(), axis=0)
            
            epoch = epoch + 1
       
        self.centroids = centroids
        self.y = y

        means = np.array([])
        for i in range(0,X.shape[0]):
            means = np.append(means, distance.euclidean(X[i], y[i]))
        self.mean_distance = np.mean(means)

        return self


class PCA():
    def __init__(self, k=2):
        self.k = k
        pass

    def fit(self, X, normalize=True):
        #Escala segundo o padrão, entre -1 e 1
        X = (X - X.mean()) / X.std()

        #Cria matriz de covariância
        cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for fl in range(0, X.shape[1]):
            for fc in range(0, X.shape[1]):
                cov_matrix[fl, fc] = np.sum(X[:,fl] * X[:,fc])/X.shape[0]

        #Extrai autovalores e autovetores
        auto_valores, auto_vetores = np.linalg.eig(cov_matrix)

        #Seleciona os k autovetores dos respectivos k maiores autovalores
        maxIndices = auto_valores.argsort()[-(self.k):][::-1]
        w = []
        variancias = []
        for i in maxIndices:
            w.append(auto_vetores[:,i])
            variancias.append(auto_valores[i]/np.sum(auto_valores))
        self.w = np.array(w).T
        self.variancias = np.array(variancias)
        return self.w

    def transform(self, X):
        X = X @ self.w
        return X


