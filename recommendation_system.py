#recommendation_system.py
#Importar las librerias necesarias
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Cargar datos de ejemplo
X = np.array([[10, 2], [15, 3], [100, 8], [120, 9], [150, 10]])
y = np.array(['Bajo', 'Bajo', 'Alto', 'Alto', 'Alto'])

# Preprocesamiento y división de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# Realizar predicciones y evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Función de recomendación
def recommend(product_features):
    prediction = model.predict([product_features])
    return prediction[0]

# Ejemplo de uso
example_product = [11.0, 2.0]
recommended_product = recommend(example_product)
print(f"Recommended Product: {recommended_product}")