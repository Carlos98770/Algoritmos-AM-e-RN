import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as lg
import time



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

#Como o dataset é pequeno, não iremos dividir em conjunto de treino e teste, para simplificar

df = pd.read_csv("Salary_dataset.csv")

X = df["YearsExperience"]
y = df["Salary"]
start_time = time.time()
obj = LinearRegression()
obj.fit(X,y)
end_time = time.time()

print(f"Tempo de execução do Mininos Quadrados: {end_time - start_time} s")

fx = obj.plot(X)
print(obj.getTheta())

plt.scatter(X,y)
plt.plot(X,fx,label="fit",color="red")
plt.xlabel("Anos de experiência")
plt.ylabel("Salário")
plt.legend()
plt.show()