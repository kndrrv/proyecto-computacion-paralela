"""
entrenar_ensemble_gpu.py
------------------------
Version GPU (RAPIDS cuML) de entrenar_ensemble_paralelo.py.

Diferencia clave con la version CPU: ahi el paralelismo lo daba joblib
repartiendo los N_SAMPLES entre nucleos. Aca cada sample entrena random
forest y xgboost DENTRO de la GPU (que ya paraleliza internamente), asi
que se recorre el loop de samples de forma secuencial -- no tiene sentido
lanzar varios procesos de joblib peleando por la misma GPU.

Mismo diseno, mismos features, mismos hiperparametros que la version CPU:
- RandomForestClassifier de sklearn -> cuml.ensemble.RandomForestClassifier
- XGBClassifier: se le agrega device="cuda"
- LogisticRegression se deja en sklearn/CPU (dataset chico por sample,
  no es el cuello de botella real)

Uso (en la particion GPU de Kabre):
    salloc --partition=nukwa --cpus-per-task=4 --time=02:00:00
    python entrenar_ensemble_gpu.py --n-samples 25

Guarda el tiempo total en benchmark_entrenamiento_gpu.csv (columnas:
n_samples,tiempo_segundos) -- insumo directo para la fila de GPU en la
tabla de speedup/eficiencia (junto con benchmark_entrenamiento.csv de la
version CPU).
"""

import argparse
import time
import joblib
import numpy as np
import cupy as cp
import polars as pl
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from cuml.ensemble import RandomForestClassifier as cuRF
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# config (identico a entrenar_ensemble_paralelo.py)
# ---------------------------------------------------------------------------
CSV_PATH = "dataset_gee_centroamerica_completo.csv"
OUT_DIR = Path("resultados_ensemble_gpu")
MODEL_DIR = Path("modelos_ensemble_gpu")

N_POR_CLASE = 60_000
CON_REEMPLAZO = False
NODATA = -9999
TRAIN_FRAC_ANIOS = 0.7
SEED = 42

TARGET = "fuego"
FEATURES = ["elevation", "pendiente", "temperatura", "precipitacion",
            "viento_u", "viento_v", "ndvi", "cobertura", "mes"]
COLS_NODATA = ["ndvi", "elevation", "pendiente", "precipitacion",
               "temperatura", "viento_u", "viento_v"]
COBERTURAS_EXCLUIDAS = [80, 70]
COL_ANIO = "anio"

OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def _a_numpy(arr):
    """cuML/XGBoost a veces devuelven cupy, a veces numpy. Normalizamos."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def entrenar_un_sample_gpu(s, X_pos, X_neg, X_test):
    """Entrena los 3 modelos para UN sample (RF y XGBoost en GPU)."""
    t0 = time.time()
    rng_local = np.random.default_rng(SEED + s)

    idx_pos = rng_local.choice(len(X_pos), size=N_POR_CLASE, replace=CON_REEMPLAZO)
    idx_neg = rng_local.choice(len(X_neg), size=N_POR_CLASE, replace=CON_REEMPLAZO)

    Xb = np.vstack([X_pos[idx_pos], X_neg[idx_neg]]).astype(np.float32)
    yb = np.concatenate([np.ones(N_POR_CLASE), np.zeros(N_POR_CLASE)]).astype(np.float32)

    probas = {}

    # --- random forest: cuML, corre en GPU ---
    Xb_gpu = cp.asarray(Xb)
    yb_gpu = cp.asarray(yb)
    X_test_gpu = cp.asarray(X_test)

    rf = cuRF(n_estimators=100, max_depth=20, random_state=SEED)
    rf.fit(Xb_gpu, yb_gpu)
    p_rf = rf.predict_proba(X_test_gpu)[:, 1]
    probas["random_forest"] = _a_numpy(p_rf)
    joblib.dump(rf, MODEL_DIR / f"random_forest_sample{s}.joblib")

    # --- xgboost: mismo estimador de siempre, solo cambia el device ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        tree_method="hist", device="cuda", n_jobs=1,
        random_state=SEED, eval_metric="logloss")
    xgb.fit(Xb, yb)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    probas["xgboost"] = _a_numpy(p_xgb)
    joblib.dump(xgb, MODEL_DIR / f"xgboost_sample{s}.joblib")

    # --- logistica: se deja en sklearn/CPU (no es el cuello de botella) ---
    sc = StandardScaler()
    log = LogisticRegression(max_iter=1000)
    log.fit(sc.fit_transform(Xb), yb)
    p_log = log.predict_proba(sc.transform(X_test))[:, 1]
    probas["logistica"] = p_log
    joblib.dump(log, MODEL_DIR / f"logistica_sample{s}.joblib")
    joblib.dump(sc, MODEL_DIR / f"scaler_sample{s}.joblib")

    return probas, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=25)
    args = parser.parse_args()

    print(">> cargando datos...")
    t0 = time.time()
    lf = (
        pl.scan_csv(CSV_PATH)
        .select(FEATURES + [TARGET, COL_ANIO])
        .filter(~pl.col("cobertura").is_in(COBERTURAS_EXCLUIDAS))
    )
    lf = lf.with_columns([
        pl.when(pl.col(c) == NODATA).then(None).otherwise(pl.col(c)).alias(c)
        for c in COLS_NODATA
    ])
    df = lf.drop_nulls().collect()
    print(f"filas tras limpieza: {df.height:,}  ({time.time()-t0:.1f}s)")

    anios = sorted(df[COL_ANIO].unique().to_list())
    corte = anios[int(len(anios) * TRAIN_FRAC_ANIOS) - 1]
    train = df.filter(pl.col(COL_ANIO) <= corte)
    test = df.filter(pl.col(COL_ANIO) > corte)
    del df

    X_test = test.select(FEATURES).to_numpy().astype(np.float32)
    y_test = test[TARGET].to_numpy()
    tasa_base = y_test.mean()
    print(f"test: {len(y_test):,} filas | tasa base de incendio: {tasa_base:.4f}")

    train_pos = train.filter(pl.col(TARGET) == 1)
    train_neg = train.filter(pl.col(TARGET) == 0)
    X_pos = train_pos.select(FEATURES).to_numpy().astype(np.float32)
    X_neg = train_neg.select(FEATURES).to_numpy().astype(np.float32)
    del train, train_pos, train_neg

    n_samples = min(args.n_samples, len(X_pos))
    print(f"n_samples: {n_samples} (GPU, secuencial)")

    t0 = time.time()
    resultados = [
        entrenar_un_sample_gpu(s, X_pos, X_neg, X_test)
        for s in range(n_samples)
    ]
    elapsed = time.time() - t0
    print(f"[BENCHMARK-GPU] n_samples={n_samples} tiempo_total={elapsed:.1f}s")

    # log para la fila de GPU en la tabla de benchmark (speedup, eficiencia)
    with open("benchmark_entrenamiento_gpu.csv", "a") as f:
        f.write(f"{n_samples},{elapsed:.2f}\n")

    # combinar resultados (promedio de probabilidades)
    probas_por_modelo = {"logistica": [], "random_forest": [], "xgboost": []}
    tiempos = []
    for probas, t in resultados:
        for nombre, p in probas.items():
            probas_por_modelo[nombre].append(p)
        tiempos.append(t)

    metricas = []
    for nombre, lista_p in probas_por_modelo.items():
        P = np.vstack(lista_p)
        p_promedio = P.mean(axis=0)
        voto_final = ((P >= 0.5).mean(axis=0) >= 0.5).astype(int)
        metricas.append({
            "modelo": nombre,
            "pr_auc": round(average_precision_score(y_test, p_promedio), 4),
            "roc_auc": round(roc_auc_score(y_test, p_promedio), 4),
            "f1_voto": round(f1_score(y_test, voto_final), 4),
        })
        np.save(OUT_DIR / f"probas_promedio_{nombre}.npy", p_promedio)

    tabla = pd.DataFrame(metricas)
    tabla.to_csv(OUT_DIR / "metricas_ensemble_gpu.csv", index=False)
    print(tabla.to_string(index=False))
    print(f"tiempo promedio por sample: {np.mean(tiempos):.1f}s")


if __name__ == "__main__":
    main()
