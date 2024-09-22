import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt




class LinearRegression():
    def __init__(self,X,y) -> None:
        self.__theta = np.random.rand(len(X[0]) + 1,1)
        self.__X = pd.DataFrame(X)
        self.__X.insert(0,"Bias",np.ones((len(X),)))
        self.__y = y
        self.__t0 = 0
        self.__t1 = 0

    def __redLearningReating(self,t):
        return self.__t0 / (t+self.__t1)

    def fit(self,n_interations,t0,t1) -> None:
        m = len(self.__X)
        self.__t0 = t0
        self.__t1 = t1
        for epoch in range(n_interations):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = self.__X[random_index:random_index+1]
                yi = self.__y[random_index:random_index+1]
                grad = 2 * xi.T @(xi@self.__theta - yi)
                eta = self.__redLearningReating(epoch *m + i)
                self.__theta = self.__theta - eta * grad
                
 
    def predict(self,data) -> np.float32:
        predict = data@self.__theta
        return predict

    def getTheta(self) -> np.array:
        return self.__theta

    def plot(self) -> np.array:
        return self.__X@self.__theta




df = pd.read_csv("Salary_dataset.csv")

X = df["YearsExperience"].to_numpy().reshape(-1,1)
y = df["Salary"].to_numpy().reshape(-1,1)
n_epochs = 50
t0,t1 = 5,700

obj = LinearRegression(X,y)

start_time = time.time()
obj.fit(n_epochs,t0,t1)
end_time = time.time()


print(f"Tempo de execução do Gradiente Estocástico: {end_time - start_time} s")

fx = obj.plot()


plt.scatter(X,y)
plt.plot(X,fx,label="fit",color="red")
plt.xlabel("Anos de experiência")
plt.ylabel("Salário")
plt.legend()
plt.show()
