"""
predict_heatmap.py
-------------------
Corre la inferencia del ENSEMBLE de XGBoost (todos los xgboost_sample*.joblib
generados por entrenar_ensemble_paralelo.py) sobre una cuadricula (grid) de
Centroamerica, en paralelo usando multiples nucleos. Disenado para Kabre.

Se usa solo el ensemble de xgboost porque: (1) fue el de mejor PR-AUC en las
metricas del equipo, y (2) no necesita StandardScaler como la logistica, lo
que simplifica la inferencia sobre datos nuevos.

Uso:
    python predict_heatmap.py --input grid_marzo.csv \
                               --model-dir modelos_ensemble \
                               --output predicciones_marzo.csv \
                               --n-jobs 8

--n-jobs debe coincidir con $SLURM_CPUS_PER_TASK en el script de SLURM.
"""

import argparse
import glob
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

# EDITAR solo si tu companera cambio el orden/nombres en modelo_tes.py.
# Este orden es el que usa modelo_tes.py / entrenar_ensemble_paralelo.py
FEATURES = ["elevation", "pendiente", "temperatura", "precipitacion",
            "viento_u", "viento_v", "ndvi", "cobertura", "mes"]

NODATA = -9999
COBERTURAS_EXCLUIDAS = [80, 70]  # agua, nieve - igual que en el entrenamiento


def predecir_chunk(rutas_modelos, chunk_df):
    """Predice un pedazo del grid promediando TODOS los modelos del ensemble."""
    X = chunk_df[FEATURES].values.astype(np.float32)
    probas = []
    for ruta in rutas_modelos:
        modelo = joblib.load(ruta)
        probas.append(modelo.predict_proba(X)[:, 1])
    return np.mean(probas, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV de la cuadricula con features")
    parser.add_argument("--model-dir", required=True, help="Carpeta con xgboost_sample*.joblib")
    parser.add_argument("--output", required=True, help="CSV de salida con probabilidades")
    parser.add_argument("--n-jobs", type=int, default=1, help="Numero de nucleos a usar")
    parser.add_argument("--mes", type=int, default=3, help="Mes a fijar en la cuadricula (3=marzo)")
    args = parser.parse_args()

    rutas_modelos = sorted(glob.glob(str(Path(args.model_dir) / "xgboost_sample*.joblib")))
    if not rutas_modelos:
        raise FileNotFoundError(
            f"No se encontraron xgboost_sample*.joblib en {args.model_dir}. "
            "Corre primero entrenar_ensemble_paralelo.py"
        )
    print(f"[INFO] Ensemble con {len(rutas_modelos)} modelos xgboost")

    print(f"[INFO] Cargando grid desde {args.input}")
    grid = pd.read_csv(args.input)
    print(f"[INFO] Total de celdas: {len(grid)}")

    # Fijar el escenario temporal (marzo = epoca seca)
    grid["mes"] = args.mes

    # Misma limpieza que en el entrenamiento: excluir agua/nieve y nodata
    if "cobertura" in grid.columns:
        grid = grid[~grid["cobertura"].isin(COBERTURAS_EXCLUIDAS)]
    for col in FEATURES:
        if col in grid.columns:
            grid = grid[grid[col] != NODATA]
    grid = grid.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"[INFO] Celdas validas tras limpieza: {len(grid)}")

    t0 = time.time()
    chunks = np.array_split(grid, args.n_jobs)
    resultados = Parallel(n_jobs=args.n_jobs)(
        delayed(predecir_chunk)(rutas_modelos, chunk) for chunk in chunks
    )
    grid["probabilidad"] = np.concatenate(resultados)
    elapsed = time.time() - t0
    print(f"[RESULTADO] n_jobs={args.n_jobs}  tiempo={elapsed:.3f}s  celdas={len(grid)}")

    grid[["lat", "lon", "probabilidad"]].to_csv(args.output, index=False)
    print(f"[INFO] Predicciones guardadas en {args.output}")

    with open("benchmark_inferencia.csv", "a") as f:
        f.write(f"{args.n_jobs},{elapsed:.4f},{len(grid)}\n")


if __name__ == "__main__":
    main()
