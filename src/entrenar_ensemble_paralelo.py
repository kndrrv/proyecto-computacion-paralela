"""
entrenar_ensemble_paralelo.py
------------------------------
Version paralelizada de modelo_tes.py: reparte los N_SAMPLES entre nucleos
usando joblib.Parallel, en vez de entrenarlos uno por uno.

Mismo diseno, mismos features, mismos hiperparametros que el original de
la companera - solo se cambia COMO se recorre el loop de samples, para
aprovechar Kabre.

Uso:
    python entrenar_ensemble_paralelo.py --n-jobs 8 --n-samples 25

--n-jobs debe coincidir con $SLURM_CPUS_PER_TASK en el script de SLURM.
"""

import argparse
import time
import joblib
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# config (identico a modelo_tes.py)
# ---------------------------------------------------------------------------
CSV_PATH = "dataset_gee_centroamerica_completo.csv"
OUT_DIR = Path("resultados_ensemble")
MODEL_DIR = Path("modelos_ensemble")

N_POR_CLASE = 90_000
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


def entrenar_un_sample(s, X_pos, X_neg, X_test):
    """Entrena los 3 modelos para UN sample. Corre en un proceso separado."""
    t0 = time.time()
    rng_local = np.random.default_rng(SEED + s)

    idx_pos = rng_local.choice(len(X_pos), size=N_POR_CLASE, replace=CON_REEMPLAZO)
    idx_neg = rng_local.choice(len(X_neg), size=N_POR_CLASE, replace=CON_REEMPLAZO)

    Xb = np.vstack([X_pos[idx_pos], X_neg[idx_neg]])
    yb = np.concatenate([np.ones(N_POR_CLASE), np.zeros(N_POR_CLASE)])

    modelos = {
        "logistica": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=1, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            tree_method="hist", n_jobs=1, random_state=SEED,
            eval_metric="logloss"),
    }
    # nota: n_jobs=1 dentro de cada modelo porque el paralelismo ya lo da
    # joblib.Parallel repartiendo los SAMPLES entre nucleos, no cada modelo.

    probas = {}
    for nombre, modelo in modelos.items():
        if nombre == "logistica":
            sc = StandardScaler()
            modelo.fit(sc.fit_transform(Xb), yb)
            p = modelo.predict_proba(sc.transform(X_test))[:, 1]
            # se guarda el scaler junto al modelo (el original no lo hacia)
            joblib.dump(sc, MODEL_DIR / f"scaler_sample{s}.joblib")
        else:
            modelo.fit(Xb, yb)
            p = modelo.predict_proba(X_test)[:, 1]
        probas[nombre] = p
        joblib.dump(modelo, MODEL_DIR / f"{nombre}_sample{s}.joblib")

    return probas, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
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
    print(f"n_samples: {n_samples} | n_jobs: {args.n_jobs}")

    t0 = time.time()
    resultados = Parallel(n_jobs=args.n_jobs)(
        delayed(entrenar_un_sample)(s, X_pos, X_neg, X_test)
        for s in range(n_samples)
    )
    elapsed = time.time() - t0
    print(f"[BENCHMARK] n_jobs={args.n_jobs} n_samples={n_samples} tiempo_total={elapsed:.1f}s")

    # log para la tabla de benchmark (speedup, eficiencia)
    with open("benchmark_entrenamiento.csv", "a") as f:
        f.write(f"{args.n_jobs},{n_samples},{elapsed:.2f}\n")

    # combinar resultados (promedio de probabilidades = insumo del mapa de calor)
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
    tabla.to_csv(OUT_DIR / "metricas_ensemble.csv", index=False)
    print(tabla.to_string(index=False))
    print(f"tiempo promedio por sample: {np.mean(tiempos):.1f}s")


if __name__ == "__main__":
    main()
