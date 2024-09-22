import numpy as np
import pandas as pd

class TgtEncoder():

    def __init__(self,dataframe,categorys,target) -> None:
        self.__df = dataframe
        self.__X = self.__df[categorys]
        self.__y = self.__df[target]
        self.__categorization = categorys
        self.__target = target


    def __particao(self) -> int:
        try:
            particoes = next(i for i in range(2,11) if len(self.__df) % i == 0)
        except StopIteration:
            particoes = 5

        return particoes

    def __folders(self) -> list:
        folders_X = np.array_split(self.__X,self.__particao())
        folders_y = np.array_split(self.__y,self.__particao())

        folders = []


        for i in range(self.__particao()):
            df_aux = pd.DataFrame()
            df_aux[self.__categorization] = folders_X[i]
            df_aux[self.__target] = folders_y[i]
            
            folders.append(df_aux)

        
        return folders


    def __meanFunction(self,dataframe) -> dict:
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
    

    def encoder(self) -> pd.DataFrame:
        meansFolders = self.__means()
        dfEncoders = []
        indRows1 = 0
        indRows2 = 0
        ctd = 0
        for i in range(1,len(meansFolders)):
	
            indRows2 = int(i*len(self.__df)/self.__particao())
            folder = self.__df.iloc[indRows1:indRows2,:].copy()
            for category in (folder.loc[indRows1:indRows2,self.__categorization].unique()):
                print(category)
                try:
                    folder.loc[folder[self.__categorization] == category,self.__categorization] = meansFolders[i - 1][category]

                except Exception:
                    
                    continue

            
            ctd +=1
            indRows1 = indRows2
            dfEncoders.append(folder)
 

        #print(meansFolders)
        df_enco = pd.concat(dfEncoders,axis=0)
        df_enco.rename(columns={self.__categorization: self.__categorization + '_econding'}, inplace=True)
        df_final = pd.concat([df_enco,self.__df.drop(self.__target,axis=1)],axis=1)

        return df_final
    

















        
