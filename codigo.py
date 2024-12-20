import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función para calcular el caudal simulado con el modelo SISO


def siso_model(b0, a1, delta, rain_depth, observed_flow, dt):
    simulated_flow = np.zeros_like(observed_flow)
    for k in range(1, len(rain_depth)):
        if k - delta >= 0:
            P_k_delta = rain_depth[k - delta]  # Precipitación en el desfase
        else:
            P_k_delta = 0  # Si el desfase es mayor que el índice, usar cero
        simulated_flow[k] = b0 * P_k_delta - a1 * simulated_flow[k - 1]
    return simulated_flow

# Función para calcular el NSE (Índice de Eficiencia de Nash-Sutcliffe)


def calculate_nse(observed, simulated):
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else np.nan

# Generar valores aleatorios para los parámetros y calcular el NSE


def random_nse_iterations(rain_depth, observed_flow, dt, num_iterations, param_ranges):
    results = []
    for _ in range(num_iterations):
        b0 = np.random.uniform(*param_ranges["b0"])
        a1 = np.random.uniform(*param_ranges["a1"])
        delta = int(np.random.uniform(*param_ranges["delta"]))

        simulated_flow = siso_model(
            b0, a1, delta, rain_depth, observed_flow, dt)
        nse = calculate_nse(observed_flow, simulated_flow)
        results.append((b0, a1, delta, nse))

    return results


# Configuración de datos
qobs_path = r"C:\Users\LENOVO\Downloads\Evaluacion 1\Caudales observados.csv"
pobs_path = r"C:\Users\LENOVO\Downloads\Evaluacion 1\Precipitaciones observadas.csv"

qobs_data = pd.read_csv(qobs_path, sep=',', header=None,
                        names=['Datetime', 'Q_obs'])
pobs_data = pd.read_csv(pobs_path, sep=',', header=None,
                        names=['Datetime', 'P_obs'])

qobs_data['Datetime'] = pd.to_datetime(qobs_data['Datetime'], errors='coerce')
pobs_data['Datetime'] = pd.to_datetime(pobs_data['Datetime'], errors='coerce')

qobs_data.set_index('Datetime', inplace=True)
pobs_data.set_index('Datetime', inplace=True)

# Eliminar duplicados en el índice
qobs_data = qobs_data[~qobs_data.index.duplicated(keep='first')]
pobs_data = pobs_data[~pobs_data.index.duplicated(keep='first')]

qobs_data = qobs_data.asfreq('T')
pobs_data = pobs_data.asfreq('T')

common_index = qobs_data.index.union(pobs_data.index)
qobs_data = qobs_data.reindex(common_index)
pobs_data = pobs_data.reindex(common_index)

pobs_data['P_obs'] = pd.to_numeric(pobs_data['P_obs'], errors='coerce')
pobs_data['P_obs'].fillna(0, inplace=True)
qobs_data['Q_obs'].interpolate(method='time', inplace=True)

rain_depth = pobs_data['P_obs'].values
observed_flow = qobs_data['Q_obs'].values / 1000  # Convertir a m3/s

# Configurar el paso de tiempo (1 minuto)
dt = 60  # Segundos

# Pedir al usuario la cantidad de iteraciones
num_iterations = int(
    input("Ingrese la cantidad de iteraciones para el análisis de sensibilidad: "))

# Definir los rangos de parámetros
param_ranges = {
    "b0": (-1.1, 1.1),
    "a1": (-1, 1),
    "delta": (14, 25)  # En minutos
}

# Generar iteraciones aleatorias
results = random_nse_iterations(
    rain_depth, observed_flow, dt, num_iterations, param_ranges)

# Convertir resultados a un DataFrame
results_df = pd.DataFrame(results, columns=['b0', 'a1', 'delta', 'NSE'])

# Obtener el mejor resultado
best_result = results_df.loc[results_df['NSE'].idxmax()]

# Simular el caudal con los mejores parámetros
simulated_flow_best = siso_model(best_result['b0'], best_result['a1'], int(
    best_result['delta']), rain_depth, observed_flow, dt)

# Graficar el análisis de sensibilidad
plt.figure(figsize=(18, 6))

# 1 - NSE vs b0
plt.subplot(1, 3, 1)
plt.scatter(results_df['b0'], 1 - results_df['NSE'],
            color='magenta', s=30, marker=".")
plt.title("1 - NSE vs b0")
plt.xlabel("b0")
plt.ylabel("1 - NSE")
plt.grid(True)

# 1 - NSE vs a1
plt.subplot(1, 3, 2)
plt.scatter(results_df['a1'], 1 - results_df['NSE'],
            color='purple', s=30, marker="o")
plt.title("1 - NSE vs a1")
plt.xlabel("a1")
plt.ylabel("1 - NSE")
plt.grid(True)

# 1 - NSE vs delta
plt.subplot(1, 3, 3)
plt.scatter(results_df['delta'], 1 - results_df['NSE'],
            color='lime', s=30, marker="h")
plt.title("1 - NSE vs delta")
plt.xlabel("delta")
plt.ylabel("1 - NSE")
plt.grid(True)

plt.tight_layout()
plt.show()

# Guardar los resultados
output_path = r"C:\Users\LENOVO\Downloads\Evaluacion 1\sensibilidad_parametros.csv"
results_df.to_csv(output_path, index=False)
print(f"Resultados del análisis de sensibilidad guardados en: {output_path}")
