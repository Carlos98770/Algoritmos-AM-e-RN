
import pandas as pd
import numpy as np
from ClassTgtEncoder import TgtEncoder

def main():
	df = pd.DataFrame()
	df["Feature"] = ['A','B','B','B','B','A','B','A','A','B','A','A','B','A','A','B','B','B','A','A']

	df["Target"] = [1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,1,1]

	X = df["Feature"]
	y = df["Target"]

	categorization = TgtEncoder(df,"Feature","Target")
	df_final = categorization.encoder()
	print(df_final)

if __name__ == "__main__":
	main()


#######################

# Para os dados de teste, ou dados que serão inseridos após o treinamento, a categorização será feita pela média das categorias nas amostras de treinamento


















