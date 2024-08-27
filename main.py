from itertools import accumulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para llenar los valores faltantes con media ± desviación estándar
def llenar_faltantes(row):
    for columna in df.columns:
        if pd.isnull(row[columna]):
            mean_val = estadisticas.loc['mean', columna]
            std_val = estadisticas.loc['std', columna]
            row[columna] = np.random.randint(mean_val - std_val, mean_val + std_val + 1)
    return row

# Función para calcular la función sigmoide
def sigmoid_function(X):
    return 1 / (1 + np.exp(-X))

def calcular_costo(h, Y, theta, lambda_=0.01, epsilon=1e-5):
    h = np.clip(h, epsilon, 1 - epsilon)
    reg_term = (lambda_ / (2 * len(Y))) * np.sum(np.square(theta[1:]))
    return -np.mean(Y * np.log(h) + (1 - Y) * np.log(1 - h)) + reg_term

# Función para calcular la matriz de confusión
def matriz_confusion(Y_true, Y_pred):
    TP = np.sum((Y_true == 1) & (Y_pred == 1))
    TN = np.sum((Y_true == 0) & (Y_pred == 0))
    FP = np.sum((Y_true == 0) & (Y_pred == 1))
    FN = np.sum((Y_true == 1) & (Y_pred == 0))
    return np.array([[TP, FP], [FN, TN]])

# Función para calcular precisión y recall
def precision_recall(confusion_matrix):
    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, accuracy

# Evaluación para diferentes umbrales
def evaluar_umbral(Y_test_prob, Y_pred_prob, umbrales):
    resultados = []
    for umbral in umbrales:
        Y_pred = (Y_pred_prob >= umbral).astype(int)
        conf_matrix = matriz_confusion(Y_test_prob, Y_pred)
        precision, recall, accuracy = precision_recall(conf_matrix)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        resultados.append((umbral, conf_matrix, precision, recall, f1_score, accuracy))
    return resultados

"""
-- Limpieza de datos --
"""

# Importar el Dataset
df = pd.read_csv('framingham.csv')

# Calcular la cantidad y proporción de datos faltantes por columna
faltantes_por_columna = df.isnull().sum()
proporcion_faltantes_por_columna = df.isnull().mean()

# Filtrar las columnas que tienen datos faltantes
faltantesPorColumna = faltantes_por_columna[faltantes_por_columna > 0]

# Graficar solo las columnas con datos faltantes
if not faltantesPorColumna.empty:
    fig = plt.figure(figsize=(11,4))
    plt.bar(faltantesPorColumna.index, faltantesPorColumna)
    plt.title('Cantidad de Datos Faltantes en las Features')
    plt.xlabel("Features")
    plt.ylabel("Instancias")

    for i, value in enumerate(faltantesPorColumna):
        plt.text(i, value + 5, str(value), ha='center')

    plt.show()

# Mostrar resultados
for columna in df.columns:
    faltantes = faltantes_por_columna[columna]
    proporcion = proporcion_faltantes_por_columna[columna]
    if faltantes > 0:
        print(f"Columna: {columna}")
        print(f"  - Cantidad de datos faltantes: {faltantes}")
        print(f"  - Proporción de datos faltantes: {proporcion:.2%}")
        print()

# Calcular la media y desviación estándar para las columnas con datos faltantes
estadisticas = df.agg(['mean', 'std'])

# Aplicar la función a cada fila del DataFrame
df = df.apply(llenar_faltantes, axis=1)

# Separar en X (features) e Y (label)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Escalar manualmente las features en X
X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Inicializar theta
theta = np.random.randn(X_scaled.shape[1] + 1, 1)

# Agregar la columna de unos a X
X_vect = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# Configurar la proporción de división
proporcion_train = 0.8
proporcion_test = 0.2

# Calcular el número de filas para el conjunto de entrenamiento
num_train = int(len(X_scaled) * proporcion_train)

# Generar una lista de índices aleatorios
indices = np.random.permutation(X_scaled.shape[0])

# Dividir los índices en entrenamiento y prueba
indices_train = indices[:num_train]
indices_test = indices[num_train:]

# Crear los conjuntos de entrenamiento y prueba para X y Y
X_train = X_vect[indices_train]
X_test = X_vect[indices_test]
Y_train = Y[indices_train]
Y_test = Y[indices_test]

Y_train = Y_train.astype(int)  # Convierte Y_train a enteros
Y_test = Y_test.astype(int)  # Convierte Y_test a enteros

# Mostrar la cantidad de filas en cada conjunto
print(f"Número de filas en train: {len(X_train)}")
print(f"Número de filas en test: {len(X_test)}\n")


"""
-- Regresión Logística --
"""

def log_regresion(X, Y, theta, alpha, max_epochs, tol):
    m = len(Y)
    prev_cost = float('inf')  # Inicializamos con un valor muy alto para la primera comparación
    for epoch in range(max_epochs):
        # Calcula las predicciones
        z = np.dot(X, theta)
        h = sigmoid_function(z)

        # Calcula el error
        error = h - Y.reshape(-1, 1)

        # Actualiza los parámetros
        gradient = np.dot(X.T, error) / m
        theta -= alpha * gradient

        # Calcula el costo para monitorear el entrenamiento
        cost = calcular_costo(h, Y, theta)

        # Verifica la diferencia con el costo anterior
        """
        if abs(prev_cost - cost) < tol:
            print(f"Entrenamiento detenido en el epoch {epoch} debido a que la diferencia de costo es menor a {tol}")
            print(f"Costro Previo: {prev_cost}")
            print(f"Costo Actual: {cost}")
            break
        """

        prev_cost = cost  # Actualiza el costo previo para la siguiente iteración

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Costo: {cost}")

    return theta


"""
-- Entrenamiento y Predicciones --
"""

# Inicializa los parámetros del modelo
alphas = [0.1, 0.5, 1.0]
epochs = 501
umbrales = np.arange(0.4, 0.66, 0.01)
dif_avg_loss = 0.0001
mejores_resultados = []


for alpha in alphas:
    print(f"\nProbando con alpha = {alpha}")

    # Reentrenar el modelo con los valores actuales de alpha y epochs
    theta = np.random.randn(X_scaled.shape[1] + 1, 1)  # Re-inicializar theta
    # Realiza el gradient descent
    theta_opt = log_regresion(X_train, Y_train, theta, alpha, epochs, dif_avg_loss)

    # Realiza predicciones en el conjunto de prueba
    z_test = np.dot(X_test, theta_opt)
    h_test = sigmoid_function(z_test)
    h_test = h_test.flatten()  # Aplana la matriz h_test
    print()

    # Evaluar umbrales y obtener resultados
    resultados = evaluar_umbral(Y_test, h_test, umbrales)

    # Inicializar variables para almacenar los mejores resultados
    mejor_precision = 0
    mejor_recall = 0
    mejor_f1_score = 0
    mejor_accuracy = 0

    umbral_mejor_precision = 0
    umbral_mejor_recall = 0
    umbral_mejor_f1_score = 0
    umbral_mejor_accuracy = 0

    # Buscar el mejor umbral para cada métrica
    for umbral, conf_matrix, precision, recall, f1_score, accuracy in resultados:
        if precision > mejor_precision:
            mejor_precision = precision
            umbral_mejor_precision = umbral

        if recall > mejor_recall:
            mejor_recall = recall
            umbral_mejor_recall = umbral

        if f1_score > mejor_f1_score:
            mejor_f1_score = f1_score
            umbral_mejor_f1_score = umbral

        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            umbral_mejor_accuracy = umbral

        print(f"Umbral: {umbral:.2f}, \nMatriz de Confusión: {conf_matrix[0]},{conf_matrix[1]}, Precisión: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}, Accuracy: {accuracy:.2f}")

    # Mostrar los umbrales con mejores métricas
    print(f"\nMejor Precision: {mejor_precision:.2f} en umbral {umbral_mejor_precision:.2f}")
    print(f"Mejor Recall: {mejor_recall:.2f} en umbral {umbral_mejor_recall:.2f}")
    print(f"Mejor F1 Score: {mejor_f1_score:.2f} en umbral {umbral_mejor_f1_score:.2f}")
    print(f"Mejor Accuracy: {mejor_accuracy:.2f} en umbral {umbral_mejor_accuracy:.2f}")
    mejores_resultados.append((alpha, epochs, umbral_mejor_precision, mejor_precision, umbral_mejor_recall, mejor_recall, umbral_mejor_f1_score, mejor_f1_score, umbral_mejor_accuracy, mejor_accuracy))

# Mostrar los mejores resultados
mejores_resultados.sort(key=lambda x: x[3], reverse=True)
print("\nMejores combinaciones de alpha, epochs y umbral:")
for resultado in mejores_resultados[:5]: # Mostrar los 5 mejores
    print(f"Alpha: {resultado[0]}, Epochs: {resultado[1]}, Umbral: {resultado[2]:.2f}, Mejor Precision: {resultado[3]:.2f}, Umbral: {resultado[4]:.2f}, Mejor Recall: {resultado[5]:.2f}, Umbral: {resultado[6]:.2f}, Mejor F1 Score: {resultado[7]:.2f}, Umbral: {resultado[8]:.2f}, Mejor Accuracy: {resultado[9]:.2f}")