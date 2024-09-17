import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



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
    df = pd.read_csv("house_prices_aracaju_v2.csv")
    treatPrices(df)

    X = df.iloc[:,0:len(df.columns)-1]
    y = df.loc[:,"Price"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.32,random_state=32)

    print(len(X_train))

    particoes = next(i for i in range(4, 11) if len(X_train) % i == 0)

    folders = np.array(())
    
    print(folders)
    #for i in range(particoes):
        

    #Address = df["Address"]
    #AddressOneHot = pd.get_dummies(Address)
    #print(AddressOneHot.head())
    

if __name__ == "__main__":
    main()