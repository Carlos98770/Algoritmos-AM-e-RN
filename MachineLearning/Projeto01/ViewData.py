import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ClassTgtEncoder import TgtEncoder



def treatPrices(dataframe):
    pricesNotFormat = pd.Series(dataframe["Price"])
    
    for i in range(len(pricesNotFormat)):
        
        if "Preço abaixo do mercado" in pricesNotFormat[i]:
            string = pricesNotFormat[i].replace("R$"," ").replace("Preço abaixo do mercado",'').replace('.','')
            number = float(string)
            pricesNotFormat[i] = number
        else:
            string = pricesNotFormat[i].replace("R$"," ").replace('.','')
            number = float(string)
            pricesNotFormat[i] = number
            

    dataframe['Price'] = pricesNotFormat.astype(np.float32)
    


def valuesMissing():
    pass

def categorization():
    #Utilizar o target encoding e K_fold target encoding, e comparar os resultados
    pass

def main():
    df = pd.read_csv("teste.csv")
    treatPrices(df)
    #print(df.head())
    #print(df.columns.tolist())
    
    #print(len(df))
    categorization = TgtEncoder(df,'Address','Price')
    df_final = categorization.encoder()
    print(df_final.head())
    #print(df.head())
    X = df.iloc[:,0:len(df.columns)-1]
    y = df.loc[:,"Price"]

    

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.32,random_state=32)

        

if __name__ == "__main__":
    main()