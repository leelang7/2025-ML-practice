import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 데이터를 불러오고, 데이터 프레임 형태로 만든 후 반환하는 함수입니다.
def load_data():
    iris = load_iris() # Bunch 객체
    #print(iris.items()) 
    irisDF = pd.DataFrame(data = iris.data, columns = iris.feature_names) # X
    irisDF['target'] = iris.target # y
    print(irisDF['target'][0])

    return irisDF

"""
1. K-means 클러스터링을 
   수행하는 함수를 구현합니다.   
"""

def k_means_clus(irisDF):
    kmeans = KMeans(init='random', n_clusters=3, random_state=200)
    kmeans.fit(irisDF.drop('target', axis=1))

    irisDF['cluster'] = kmeans.labels_ # 군집 넘버

    # 군집화 결과를 보기 위해 groupby 함수를 사용해보겠습니다.
    # 정답지랑 군집 넘버랑 비교하기 위해 
    iris_result = irisDF.groupby(['target','cluster'])['sepal length (cm)'].count()
    print(iris_result)
    
    return iris_result, irisDF

# 군집화 결과 시각화하기
def Visualize(irisDF):
    #print(irisDF.head())
    pca = PCA(n_components=2) # 4 => 2개의 주성분(축)만 추출
    pca_transformed = pca.fit_transform(irisDF)
    print(pca_transformed[0])
    irisDF['pca_x'] = pca_transformed[:,0]
    irisDF['pca_y'] = pca_transformed[:,1]
    
    # 군집된 값이 0, 1, 2 인 경우, 인덱스 추출
    idx_0 = irisDF[irisDF['cluster'] == 0].index
    idx_1 = irisDF[irisDF['cluster'] == 1].index
    idx_2 = irisDF[irisDF['cluster'] == 2].index

    # 각 군집 인덱스의 pca_x, pca_y 값 추출 및 시각화
    fig, ax = plt.subplots()
    ax.scatter(x=irisDF.loc[idx_0, 'pca_x'], y= irisDF.loc[idx_0, 'pca_y'], marker = 'o')
    ax.scatter(x=irisDF.loc[idx_1, 'pca_x'], y= irisDF.loc[idx_1, 'pca_y'], marker = 's')
    ax.scatter(x=irisDF.loc[idx_2, 'pca_x'], y= irisDF.loc[idx_2, 'pca_y'], marker = '^')
    ax.set_title('K-menas')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    fig.savefig("k-means_plot.png")

def main():
    irisDF = load_data()
    iris_result, irisDF = k_means_clus(irisDF)
    Visualize(irisDF)

if __name__ == "__main__":
    main()