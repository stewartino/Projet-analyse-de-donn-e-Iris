import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('~/Desktop/code/Data_science/Projet_flowers/iris.csv', delimiter=',')


print(df.head(5))

x = df.sepal_length
y = df.sepal_width
x = x.values.reshape(-1, 1)  # Si x est une seule colonne, transforme-le en 2D

# initialisation du modèle
regression_model = LinearRegression()
# Adapter les données (entraînement du modèle)
regression_model.fit(x, y)
# Prédiction
y_predicted = regression_model.predict(x)
# Évaluation du modèle
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
# Affichage des valeurs
print("Pente : " ,regression_model.coef_)
print("Ordonnée à l'origine : ", regression_model.intercept_)
print("Racine carrée de l'erreur quadratique moyenne : ", rmse)
print('Sccore R2 : ', r2)
# Tracée des valeurs
# Points de données
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
# Valeurs prédites
plt.plot(x, y_predicted, color='r')
plt.show()