import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as lg



#Classe com o metódo de solução de otimização análitico
class LinearRegression():
    def __init__(self) -> None:
        self.__theta = 0
        
    def fit(self,X,y) -> None:
        X_b = pd.DataFrame(X)
        X_b.insert(0,"Bias",np.ones((len(X),)))
        
        self.__theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        pass
    
    def predict(self,data) -> np.float32:
        predict = data@self.__theta
        return predict
        
    def getTheta(self) -> np.array:
        return self.__theta
    
    def plot(self,X) -> np.array:
        X_b = pd.DataFrame(X)
        X_b.insert(0,"Bias",np.ones((len(X),)))

        return X_b.dot(self.__theta)


X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.rand(100,1)
obj = LinearRegression()
obj.fit(X,y)
fx = obj.plot(X)

plt.scatter(X,y)
plt.plot(X,fx,label="fit",color="red")
plt.legend()
plt.show()