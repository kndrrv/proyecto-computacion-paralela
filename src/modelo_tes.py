"""
ensemble por bootstrapping - ocurrencia de incendios centroamerica
enfoque recomendado por el profesor:
  1. armar N "samples" balanceadas: N_POR_CLASE incendios + N_POR_CLASE no-incendios cada una
     (N_POR_CLASE se ajusta solo al total de incendios disponibles en train)
  2. cada sample muestrea no-incendios distintos (con o sin reemplazo, se comparan ambos)
  3. entrenar un modelo por sample (logistica, random forest y xgboost)
  4. prediccion final por votacion de mayoria: si la mayoria de los modelos dice
     "incendio", el punto se clasifica como incendio
  5. tambien se promedian las probabilidades (util para el mapa de calor)

por que funciona: cada modelo ve un dataset 50/50 (sin desbalance), y al combinar
varios modelos que vieron negativos distintos, el ensemble cubre la variedad de la
clase mayoritaria sin ahogarse en ella.

escalar a kabre: cada sample es independiente, asi que en el cluster se entrena
cada una en un proceso/nodo distinto. localmente se corre secuencial con pocos
samples para validar el codigo.

correr:  python ensemble_bootstrap.py
salidas: metricas en 'resultados_ensemble/', modelos en 'modelos_ensemble/'
"""

import time
import joblib
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
CSV_PATH = "dataset_gee_centroamerica_completo.csv"  # ajusta la ruta si hace falta
OUT_DIR = Path("resultados_ensemble")
MODEL_DIR = Path("modelos_ensemble")

N_SAMPLES = 5          # local: 5 para probar | kabre: subir a 25-30
N_POR_CLASE = 90_000   # tope deseado por clase; se ajusta solo si hay menos incendios
CON_REEMPLAZO = False  # True = bootstrap clasico | False = sin repetidos
                       # (ya se comparo: sin reemplazo dio mejor pr-auc y fue mas rapido)
NODATA = -9999
TRAIN_FRAC_ANIOS = 0.7   # fraccion de años (los mas viejos) para entrenar
SEED = 42

TARGET = "fuego"
FEATURES = ["elevation", "pendiente", "temperatura", "precipitacion",
            "viento_u", "viento_v", "ndvi", "cobertura", "mes"]
COLS_NODATA = ["ndvi", "elevation", "pendiente", "precipitacion",
               "temperatura", "viento_u", "viento_v"]
COBERTURAS_EXCLUIDAS = [80, 70]   # agua, nieve
COL_ANIO = "anio"

OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. carga y limpieza (igual que en el modelo base)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. split temporal (identico al modelo base: pasado entrena, futuro evalua)
# ---------------------------------------------------------------------------
años = sorted(df[COL_ANIO].unique().to_list())
corte = años[int(len(años) * TRAIN_FRAC_ANIOS) - 1]
print(f"train: {años[0]}-{corte} | test: {corte + 1}-{años[-1]}")

train = df.filter(pl.col(COL_ANIO) <= corte)
test = df.filter(pl.col(COL_ANIO) > corte)
del df

# el TEST se queda con su desbalance real, asi se evalua en condiciones reales
X_test = test.select(FEATURES).to_numpy().astype(np.float32)
y_test = test[TARGET].to_numpy()
tasa_base = y_test.mean()
print(f"test: {len(y_test):,} filas | tasa base de incendio: {tasa_base:.4f}")

# separar positivos y negativos del train (de aqui salen los samples)
train_pos = train.filter(pl.col(TARGET) == 1)
train_neg = train.filter(pl.col(TARGET) == 0)
print(f"train: {train_pos.height:,} incendios | {train_neg.height:,} no-incendios")
del train

# ajustar N_POR_CLASE a los incendios disponibles, usamos como maximo todos los
# que hay, no tiene sentido pedir mas (ni con ni sin reemplazo)
N_POR_CLASE = min(N_POR_CLASE, train_pos.height)
print(f"n por clase ajustado a: {N_POR_CLASE:,} (limite = incendios disponibles)")

# ---------------------------------------------------------------------------
# 3. entrenar N samples balanceadas
#    cada sample: N_POR_CLASE incendios + N_POR_CLASE no-incendios (50/50)
# ---------------------------------------------------------------------------
def hacer_modelos():
    """devuelve un set fresco de los tres modelos (uno nuevo por sample)"""
    return {
        "logistica": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            tree_method="hist", n_jobs=-1, random_state=SEED,
            eval_metric="logloss"),
    }
    # nota: ya no se usan pesos de clase, porque cada sample viene 50/50

# aqui se acumulan las probabilidades predichas de cada modelo de cada sample
probas_por_modelo = {m: [] for m in hacer_modelos()}
tiempos = []

for s in range(N_SAMPLES):
    t0 = time.time()
    semilla = SEED + s   # semilla distinta -> negativos distintos en cada sample

    # muestrear el sample (con o sin reemplazo segun config)
    pos = train_pos.sample(n=N_POR_CLASE, with_replacement=CON_REEMPLAZO, seed=semilla)
    neg = train_neg.sample(n=N_POR_CLASE, with_replacement=CON_REEMPLAZO, seed=semilla)
    sample = pl.concat([pos, neg])

    Xb = sample.select(FEATURES).to_numpy().astype(np.float32)
    yb = sample[TARGET].to_numpy()

    # entrenar los tres modelos sobre este sample y predecir el test completo
    for nombre, modelo in hacer_modelos().items():
        if nombre == "logistica":
            sc = StandardScaler()
            modelo.fit(sc.fit_transform(Xb), yb)
            p = modelo.predict_proba(sc.transform(X_test))[:, 1]
        else:
            modelo.fit(Xb, yb)
            p = modelo.predict_proba(X_test)[:, 1]
        probas_por_modelo[nombre].append(p)
        joblib.dump(modelo, MODEL_DIR / f"{nombre}_sample{s}.joblib")

    tiempos.append(time.time() - t0)
    print(f"sample {s + 1}/{N_SAMPLES} listo ({tiempos[-1]:.1f}s)")

# ---------------------------------------------------------------------------
# 4. combinar: votacion de mayoria + promedio de probabilidades
# ---------------------------------------------------------------------------
print("\n>> combinando el ensemble...")
resultados = []

for nombre, lista_p in probas_por_modelo.items():
    P = np.vstack(lista_p)              # forma: (n_samples, n_test)

    # votacion de mayoria: cada sample vota 1 si p >= 0.5; gana la mayoria
    votos = (P >= 0.5).astype(int)
    voto_final = (votos.mean(axis=0) >= 0.5).astype(int)

    # promedio de probabilidades: mas suave, es lo que usa el mapa de calor
    p_promedio = P.mean(axis=0)

    resultados.append({
        "modelo": nombre,
        "pr_auc": round(average_precision_score(y_test, p_promedio), 4),
        "roc_auc": round(roc_auc_score(y_test, p_promedio), 4),
        "precision_voto": round(precision_score(y_test, voto_final), 4),
        "recall_voto": round(recall_score(y_test, voto_final), 4),
        "f1_voto": round(f1_score(y_test, voto_final), 4),
    })

    # guardar las probabilidades promedio (insumo del mapa de calor)
    np.save(OUT_DIR / f"probas_promedio_{nombre}.npy", p_promedio)

tabla = pd.DataFrame(resultados)
tabla.to_csv(OUT_DIR / "metricas_ensemble.csv", index=False)

print(f"\nconfig: {N_SAMPLES} samples | {N_POR_CLASE:,} por clase | "
      f"reemplazo: {CON_REEMPLAZO}")
print(f"tasa base del test: {tasa_base:.4f}")
print("\n=== metricas del ensemble ===")
print(tabla.to_string(index=False))
print(f"\ntiempo promedio por sample: {np.mean(tiempos):.1f}s "
      f"(total: {sum(tiempos):.1f}s)")
print("nota kabre: los samples son independientes -> cada uno puede ir a un "
      "proceso o nodo distinto")

# ---------------------------------------------------------------------------
# 5. figura: acuerdo entre samples (que tan de acuerdo estan los votantes)
# ---------------------------------------------------------------------------
P = np.vstack(probas_por_modelo["xgboost"])
acuerdo = (P >= 0.5).mean(axis=0)   # fraccion de samples que votan "incendio"

plt.figure(figsize=(9, 6))
plt.hist(acuerdo, bins=N_SAMPLES + 1, color="#c1440e", edgecolor="white")
plt.xlabel("fraccion de samples que votan 'incendio'")
plt.ylabel("cantidad de puntos del test")
plt.title(f"acuerdo entre los {N_SAMPLES} samples (xgboost)")
plt.tight_layout()
plt.savefig(OUT_DIR / "acuerdo_samples.png", dpi=150)
plt.close()

print("\n>> listo. metricas y figuras en 'resultados_ensemble/'.")