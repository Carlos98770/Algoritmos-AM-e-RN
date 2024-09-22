import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self,X,y) -> None:
        self.__theta = np.random.rand(len(X[0]) + 1,1)
        self.__X = pd.DataFrame(X)
        self.__X.insert(0,"Bias",np.ones((len(X),)))
        self.__y = y

    def fit(self,learning_reating,n_interations) -> None:
        m = len(self.__X)
        for i in range(n_interations):
            grad = 2/m * self.__X.T @ (self.__X@self.__theta - self.__y)
            self.__theta = self.__theta - learning_reating*grad
 
    def predict(self,data) -> np.float32:
        predict = data@self.__theta
        return predict

    def getTheta(self) -> np.array:
        return self.__theta

    def plot(self) -> np.array:
        return self.__X@self.__theta
    

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.rand(100,1)
learning_reating = 0.1
n_interations = 1000
obj = LinearRegression(X,y)
obj.fit(learning_reating,n_interations)

fx = obj.plot()

plt.scatter(X,y)
plt.plot(X,fx,label="fit",color="red")
plt.legend()
plt.show()