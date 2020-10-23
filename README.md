# K-means, PCA, Árvores de Decisão

## Implementação e Relatório

1. k-means
   * Implemente o k-means usando a distância euclidiana.
   * Execute o k-means para k ={2,3,4,5}
     * a. Plote a distância média de cada ponto para o seu centroide em um gráfico linha em função de k (média sobre 20 rodadas)
     * b. Discuta qual seria o k ideal a ser usado
2. PCA
    * Implemente o PCA
        * a. Você deve implementar a função de calcular a matriz de covariância
        * b. A função de achar os autovetores e os autovalores pode ser usado pronto do [numpy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)
    * Reduza o conjunto de dados original em um conjunto com apenas duas variáveis (2 componentes principais de maior autovalor)
        * a. Reporte quanto de variância foi preservado
        * b. Plote cada ponto do conjunto transformado em um gráfico de dispersão 2d  atribuindo uma cor para cada uma das classes (3 classes no total).

## Conjunto de dados

Carregar trab4.data
