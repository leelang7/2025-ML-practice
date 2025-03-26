import preprocess as pre
import model as md
import numpy as np
from sklearn.metrics import silhouette_score

def silhouette_calculator(X, cluster):
    # 군집화 결과에 대한 silhouette score를 계산합니다.
    labels = list(cluster.values())  # 클러스터에서 레이블 추출
    silhouette = silhouette_score(X, labels)  # silhouette score 계산
    return silhouette

# 군집의 개수로 3을 선택합니다. 필요에 따라 원하는 군집의 개수로 변경하세요.
def your_choice(X):
    return md.KMeans(K=2).fit(X)  # K값을 3으로 설정하고 학습 수행

def main():

    X = pre.load_data()
    
    Kmeans = your_choice(X)  # KMeans 모델 생성 및 학습
    
    # 모델 학습의 결과로 silhouette 값을 계산합니다.
    silhouette = silhouette_calculator(X, Kmeans.cluster)  # silhouette 값 계산하여 cluster 사용
    
    # 군집의 개수와 silhouette 값을 출력합니다. 
    print('K = {}, silhouette 값: {}'.format(2, silhouette))
                

if __name__ == "__main__":
    main()