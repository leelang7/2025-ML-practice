import preprocess as pre
import model as md
import numpy as np

def variance_calculator(model):
    # PCA 결과에 대한 분산의 유지율 계산하여 반환합니다.
    
    # 원본의 분산을 구합니다.
    original_var = model.lambda_.sum()
    
    # 차원축소된 벡터공간에서의 분산 유지율을 계산합니다.
    order = (-model.lambda_).argsort()
    print(model.lambda_)
    print(-model.lambda_)
    shrinked_var = model.lambda_[order[:model.n_components]].sum()  # 상위 n_components 고유값의 합계
    
    # 차원축소된 벡터공간에서 유지된 분산을 원본의 분산으로 나눠 분산 유지율을 계산해 반환합니다.
    return shrinked_var / original_var

def your_choice(X, n_components=2):  # n_components를 함수의 인자로 추가합니다.
    myPCA = md.PCA(n_components=n_components)  # n_components로 설정
    myPCA.fit_transform(X)
    return myPCA

def main():
    X = pre.load_data()
    
    # PCA의 n_components에 성분의 개수를 입력해 여러가지 PCA 모델을 시도해보세요.
    n_components = 2  # 원하는 주성분의 수
    myPCA = your_choice(X, n_components)  # PCA 모델 생성 및 데이터 변환
    
    # 모델 학습의 결과로 원본 대비 축소후의 분산의 비율을 계산합니다.
    percentage = variance_calculator(myPCA)  # 분산 유지율 계산
    
    # 축소후 차원의 수와 원본 대비 축소후의 분산의 비율을 출력합니다.
    print('원본 데이터의 차원 = {}, n_components = {}, 원본 대비 축소후의 분산의 비율: {}'.format(X.shape[1], n_components, percentage))

if __name__ == "__main__":
    main()