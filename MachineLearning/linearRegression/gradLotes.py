import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


class LinearRegression():

    #X e y devem ser um matriz 2d
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
    

#Como o dataset é pequeno, não iremos dividir em conjunto de treino e teste, para simplificar



df = pd.read_csv("Salary_dataset.csv")

X = df["YearsExperience"].to_numpy().reshape(-1,1)
y = df["Salary"].to_numpy().reshape(-1,1)

learning_reating = 0.01
n_interations = 1000
start_time = time.time()


obj = LinearRegression(X,y)
obj.fit(learning_reating,n_interations)
end_time = time.time()

print(f"Tempo de execução do Gradiente em lotes: {end_time - start_time} s")
#print(obj.getTheta())
fx = obj.plot()

plt.scatter(X,y)
plt.plot(X,fx,label="fit",color="red")
plt.xlabel("Anos de experiência")
plt.ylabel("Salário")
plt.legend()
plt.show()
