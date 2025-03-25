import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#  데이터의 랜덤 생성을 위한 seed를 설정합니다. 
np.random.seed(0)

# 1. 데이터 x와 y를 생성하는 함수와 선형 회귀를 반환하는 함수를 살펴봅니다. 
def load_data():
    X = 5*np.random.rand(100,1)
    y = 3*X + 5*np.random.rand(100,1)
    
    # 학습용 데이터와 테스트용 데이터를 분리합니다. 
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3, random_state=0)
    
    return train_X, train_y, test_X, test_y

def Linear_Regression(train_X, train_y):
    # LinearRegression 클래스를 불러와서 회귀 모델을 만들어 봅니다.
    lr = LinearRegression()

    # 회귀 모델을 데이터 학습용 데이터에 맞추어 학습시킵니다.
    lr.fit(train_X,train_y)
    
    return lr
    
## 그래프로 나타내어 봅니다.
def plotting_graph(test_X, test_y, predicted):
    plt.scatter(test_X,test_y)
    plt.plot(test_X, predicted, color='r')
    
    plt.savefig("mae_mse.png")


# 2. 정의한 함수들을 이용하여 main() 함수를 완성하세요.
def main():
    # 생성한 데이터를 학습용 데이터와 테스트 데이터를 분리하여 반환하는 함수를 호출합니다.
    train_X, train_y, test_X, test_y = load_data()
    
    # 학습용 데이터를 바탕으로 학습한 선형 회귀를 저장합니다. 
    lr = Linear_Regression(train_X, train_y)
    
    # 학습된 모델을 바탕으로, 테스트 데이터 x의 회귀 결과를 출력합니다.
    predicted = lr.predict(test_X)
    
    # 모델의 예측 결과 즉, 회귀 알고리즘을 평가하기 위한 MSE, MAE 값을 저장합니다.
    MAE = mean_absolute_error(test_y, predicted)
    MSE = mean_squared_error(test_y,predicted)
    # RMSE : MSE의 제곱근
    
    print("> MSE :",MSE)
    print("> MAE :",MAE)
    
    plotting_graph(test_X, test_y, predicted)
    
    return MSE, MAE

if __name__=="__main__":
    main()