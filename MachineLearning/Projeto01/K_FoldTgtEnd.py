
import pandas as pd
import numpy as np
from ClassTgtEncoder import TgtEncoder

def main():
	df = pd.DataFrame()
	df["Address"] = ["Rua Pina Ramos, 6 - Luzia, Aracaju - SE","Rua F, 176 - Aruana, Aracaju - SE","Rua B, 13 - Jabotiana, Aracaju - SE"]

	df["Area"] = [160,325,80]
	df["Rooms"] = [3,3,3]
	df["Bathrooms"] = [3,3,1]
	df["Garage Cars"] = [1,3,1]
	df["Price"] = [280000,360000,279000]

	#print(df)

	categorization = TgtEncoder(df,"Address","Price")
	df_final = categorization.encoder()
	print(df_final)

if __name__ == "__main__":
	main()

###Testar a classe TgtEncoder para dados continuos e não classificatório
#######################

# Para os dados de teste, ou dados que serão inseridos após o treinamento, a categorização será feita pela média das categorias nas amostras de treinamento



 















