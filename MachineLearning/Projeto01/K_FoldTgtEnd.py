
import pandas as pd
import numpy as np

df = pd.DataFrame()
df["Feature"] = ['A','B','B','B','B','A','B','A','A','B','A','A','B','A','A','B','B','B','A','A']

df["Target"] = [1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,1,1]

X = df["Feature"]
y = df["Target"]


particoes = next(i for i in range(5,11) if len(X) % i == 0)

folders_X = np.array_split(X,particoes)
folders_y = np.array_split(y,particoes)

folders = []


for i in range(particoes):
	df_aux = pd.DataFrame()
	df_aux["Feature"] = folders_X[i]
	df_aux["Target"] = folders_y[i]
	
	folders.append(df_aux)


def meanFunction(dataframe):
	#Coluna 0 é as categorias e 1 e os target
	categorias = dataframe.iloc[:,0].unique()
	#print(type(categorias))
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

		#print(n,nTargets)
		mean.append(nTargets/n)


	for i, cat in enumerate(categoria):
		dic[cat] = mean[i]
	
	
	return dic


meansFolders = []
for i in range(particoes):
	dataframe = pd.DataFrame()
	dfs = []
	for j in range(particoes):
		if j != i:
			dfs.append(folders[j])
			

	df_cat = pd.concat(dfs,axis=0)
	mean = meanFunction(df_cat)
	meansFolders.append(mean)


#print(meansFolders)
#print(df)
dfEncoders = []
indRows1 = 0
indRows2 = 0
ctd = 0
for i in range(1,6):
	
	indRows2 = int(i*len(df)/particoes)
	folder = df.iloc[indRows1:indRows2,:].copy()
	#print(folder.loc[indRows1:indRows2,"Feature"])
	for category in (folder.loc[indRows1:indRows2,"Feature"].unique()):

		#print(j)
		#print("I-esima categoria")
		#print(category)
		#print("------------")
		#print(meansFolders[ctd][category])
		folder.loc[folder["Feature"] == category,"Feature"] = meansFolders[i - 1][category]
		#print(folder.loc[folder["Feature"] == category,"Feature"])

	
	ctd +=1
	indRows1 = indRows2
	
	dfEncoders.append(folder)



df_enco = pd.concat(dfEncoders,axis=0)
df_enco.rename(columns={'Feature': 'Feature_econding'}, inplace=True)
df_final = pd.concat([df_enco,df.drop("Target",axis=1)],axis=1)
#print(df_final)

#######################

# Para os dados de teste, ou dados que serão inseridos após o treinamento, a categorização será feita pela média das categorias nas amostras de treinamento


















