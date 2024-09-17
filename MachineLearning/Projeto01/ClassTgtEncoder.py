import numpy as np
import pandas as pd

class TgtEncoder():

    def __init__(self,dataframe,categorys,target) -> None:
        self.__df = dataframe
        self.__X = self.__df[categorys]
        self.__y = self.__df[target]


    def __particao(self) -> int:
        particoes = next(i for i in range(5,11) if len(self.__df) % i == 0)
        return particoes

    def __folders(self) -> list:
        folders_X = np.array_split(self.__X,self.__particao())
        folders_y = np.array_split(self.__y,self.__particao())

        folders = []


        for i in range(self.__particao()):
            df_aux = pd.DataFrame()
            df_aux["Feature"] = folders_X[i]
            df_aux["Target"] = folders_y[i]
            
            folders.append(df_aux)

        return folders


    def __meanFunction(self,dataframe):
        #Coluna 0 é as categorias e 1 e os target
        categorias = dataframe.iloc[:,0].unique()
        target = dataframe.iloc[:,1].to_numpy()

        mean = []
        categoria = []
        for i in range(len(categorias)):
            n = 0
            mediaClasse = 0
            nTargets = 0
            dic = {}
            for j in range(len(dataframe)):
                if dataframe.iloc[j,0] == categorias[i]:
                    n += 1
                    nTargets += target[j]
            categoria.append(categorias[i])

        
            mean.append(nTargets/n)


        for i, cat in enumerate(categoria):
            dic[cat] = mean[i]
        
        return dic
    

    def __means(self) -> list:
        folders = self.__folders()
        meansFolders = []
        for i in range(self.__particao()):
            dataframe = pd.DataFrame()
            dfs = []
            for j in range(self.__particao()):
                if j != i:
                    dfs.append(folders[j])
                    

            df_cat = pd.concat(dfs,axis=0)
            mean = self.__meanFunction(df_cat)
            meansFolders.append(mean)
        
        return meansFolders
    

    def encoder() -> pd.DataFrame:
        pass
        
