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

        simulated_flow = siso_model(b0, a1, delta, rain_depth, observed_flow, dt)
        nse = calculate_nse(observed_flow, simulated_flow)
        results.append((b0, a1, delta, nse))

    return results

# Configuración de datos
qobs_path = r"C:\\Users\\Hernan Moreno\\Desktop\\Evaluacion1\\Qobs31.csv"
pobs_path = r"C:\\Users\\Hernan Moreno\\Desktop\\Evaluacion1\\Pobs31.csv"

qobs_data = pd.read_csv(qobs_path, sep=',', header=None, names=['Datetime', 'Q_obs'])
pobs_data = pd.read_csv(pobs_path, sep=',', header=None, names=['Datetime', 'P_obs'])

qobs_data['Datetime'] = pd.to_datetime(qobs_data['Datetime'], errors='coerce')
pobs_data['Datetime'] = pd.to_datetime(pobs_data['Datetime'], errors='coerce')

qobs_data.set_index('Datetime', inplace=True)
pobs_data.set_index('Datetime', inplace=True)

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
num_iterations = int(input("Ingrese la cantidad de iteraciones para el análisis de sensibilidad: "))

# Definir los rangos de parámetros
param_ranges = {
    "b0": (-1.1, 0.6),
    "a1": (-1, 1),
    "delta": (14, 25)  # En minutos
}

# Generar iteraciones aleatorias
results = random_nse_iterations(rain_depth, observed_flow, dt, num_iterations, param_ranges)

# Convertir resultados a un DataFrame
results_df = pd.DataFrame(results, columns=['b0', 'a1', 'delta', 'NSE'])

# Obtener el mejor resultado
best_result = results_df.loc[results_df['NSE'].idxmax()]

# Simular el caudal con los mejores parámetros
simulated_flow_best = siso_model(best_result['b0'], best_result['a1'], int(best_result['delta']), rain_depth, observed_flow, dt)

# Graficar el hidrograma
plt.figure(figsize=(12, 6))
time_index = qobs_data.index[:len(observed_flow)]
plt.plot(time_index, observed_flow * 1000, label="Q Observado (L/s)", color="blue")
plt.plot(time_index, simulated_flow_best * 1000, label="Q Simulado (L/s)", color="orange", linestyle="--")
plt.title(f"Hidrograma Simulado vs Observado\nMejor NSE: {best_result['NSE']:.4f}")
plt.xlabel("Tiempo")
plt.ylabel("Caudal (L/s)")
plt.legend()
plt.grid()
plt.show()

# Graficar el análisis de sensibilidad
plt.figure(figsize=(18, 6))

# NSE vs b0
plt.subplot(1, 3, 1)
plt.scatter(results_df['b0'], 1 - results_df['NSE'], color='blue', s=10)
plt.title("NSE vs b0")
plt.xlabel("b0")
plt.ylabel("NSE")
plt.grid(True)

# NSE vs a1
plt.subplot(1, 3, 2)
plt.scatter(results_df['a1'], 1 - results_df['NSE'], color='green', s=10)
plt.title("NSE vs a1")
plt.xlabel("a1")
plt.ylabel("NSE")
plt.grid(True)

# NSE vs delta
plt.subplot(1, 3, 3)
plt.scatter(results_df['delta'], 1 - results_df['NSE'], color='red', s=10)
plt.title("NSE vs delta")
plt.xlabel("delta")
plt.ylabel("NSE")
plt.grid(True)

plt.tight_layout()
plt.show()

def calculate_nse(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE).
    """
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

# Filtrar las simulaciones con un NSE mayor o igual a 0.75
good_simulations = results_df[results_df['NSE'] >= 0.75]

# Verificar si hay simulaciones comportamentales
if good_simulations.empty:
    print("No se encontraron simulaciones con NSE mayor o igual a 0.75.")
else:
    # Almacenar los hidrogramas correspondientes a las simulaciones buenas
    good_hydrographs = [
        siso_model(row['b0'], row['a1'], int(row['delta']), rain_depth, observed_flow, dt)
        for _, row in good_simulations.iterrows()
    ]

    # Convertir las listas de hidrogramas a un array para calcular percentiles
    good_hydrographs = np.array(good_hydrographs)

    # Calcular los percentiles 5% y 95% para los hidrogramas simulados
    percentile_5 = np.percentile(good_hydrographs, 5, axis=0)
    percentile_95 = np.percentile(good_hydrographs, 95, axis=0)

    # Seleccionar el mejor hidrograma simulado (con el mayor NSE)
    best_simulation = good_simulations.loc[good_simulations['NSE'].idxmax()]
    best_simulated = siso_model(
        best_simulation['b0'], best_simulation['a1'], int(best_simulation['delta']),
        rain_depth, observed_flow, dt
    )

    # Generar gráfico con banda de incertidumbre y los simulados
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, observed_flow * 1000, label="Q Observado (L/s)", color="blue")
    plt.plot(time_index, best_simulated * 1000, label="Q Simulado Mejor (L/s)", color="orange", linestyle="--")
    plt.plot(time_index, percentile_5 * 1000, label="Percentil 5% (Simulado)", color="green", linestyle="--")
    plt.plot(time_index, percentile_95 * 1000, label="Percentil 95% (Simulado)", color="green", linestyle="--")
    plt.fill_between(time_index, percentile_5 * 1000, percentile_95 * 1000, color="green", alpha=0.3)
    plt.title(f"Análisis de Incertidumbre - Mejor Simulación\nNSE: {best_simulation['NSE']:.4f}")
    plt.xlabel("Tiempo")
    plt.ylabel("Caudal (L/s)")
    plt.legend()
    plt.grid()
    plt.show()

    # Guardar los resultados en un archivo CSV
    uncertainty_hydrograph = pd.DataFrame({
        'Datetime': time_index,
        'Q_Observado_Ls': observed_flow * 1000,
        'Percentil_5_Ls': percentile_5 * 1000,
        'Percentil_95_Ls': percentile_95 * 1000,
    })
    uncertainty_hydrograph.to_csv("uncertainty_hydrograph.csv", index=False)

    print("Análisis de incertidumbre completado y resultados guardados en 'uncertainty_hydrograph.csv'.")