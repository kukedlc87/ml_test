import joblib
import numpy as np

# Carga el modelo previamente entrenado
clf = joblib.load('modelo.pkl')

# Define el conjunto de características para una sola muestra
nueva_muestra = np.array([30, 38632, 216226, 649, 99, 2, 11.76, 24, 0.19, 3, 3, 2, 0, 1, 1, 1]).reshape(1, -1)

# Realiza la predicción de probabilidad para la nueva muestra
y_pred = clf.predict_proba(nueva_muestra)

# Muestra las predicciones
print(y_pred[:1,0])
variable = (y_pred[:1,0] * 100).astype(int)
print(variable)