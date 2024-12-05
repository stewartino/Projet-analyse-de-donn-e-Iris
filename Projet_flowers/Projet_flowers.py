import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import sklearn
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('~/Desktop/code/Data_science/Projet_flowers/iris.csv', delimiter=',')

# x = x.values.reshape(-1, 1)  # Si x est une seule colonne, transforme-le en 2D

# print(df.head())

ab = df.isnull()
# print(ab)

# Examiner les types de données et rechercher les valeurs manquantes.
df.dropna()
df.info()
# print(df.info())

unique_strings = df.iloc[:, 4].unique()
print(unique_strings)

max_sepal_length = df.iloc[:, 0].max()
print(max_sepal_length)
max_sepal_length = df.iloc[:, 0].min()
print(max_sepal_length)
# Calculer les statistiques descriptives des colonnes numériques (moyenne, écart-type, etc.).
# Statistiques sepal_length


mean_sepal_length = np.mean(df.iloc[:, 0])
print(mean_sepal_length)
median_sepal_length = np.median(df.iloc[:, 0])
print(median_sepal_length)
std_sepal_length = np.std(df.iloc[:, 0])
print(std_sepal_length)