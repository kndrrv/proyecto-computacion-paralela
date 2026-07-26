"""
eda + analisis estadistico - ocurrencia de incendios centroamerica
dataset: dataset_gee_centroamerica_completo.csv (~15.6M filas, 14 columnas, ~2.14 GB)
target: fuego (1 = incendio, 0 = no incendio)

estrategia de memoria (importante en 16 GB):
  - agregaciones baratas (balance, nodata, tasas) -> polars lazy sobre todo el csv
  - correlaciones / info mutua / graficas -> muestra estratificada de 500k filas
nunca se carga el csv completo a un dataframe de golpe.
"""

import polars as pl
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
CSV_PATH = "dataset_gee_centroamerica_completo.csv"  # ajusta la ruta si hace falta
OUT_DIR = Path("eda_salidas")
FIG_DIR = Path("figuras")
NODATA = -9999                # valor centinela de "sin dato" (visto en ndvi sobre agua)
N_SAMPLE = 500_000            # muestra estratificada para correlaciones/graficas
MI_SAMPLE = 120_000           # submuestra para info mutua (es lenta)
SEED = 42

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")

TARGET = "fuego"
# columnas numericas donde -9999 es fisicamente imposible => es nodata
NUM_SENTINEL = ["ndvi", "elevation", "pendiente", "precipitacion", "temperatura", "viento_u", "viento_v"]
# codigos de cobertura esa worldcover -> nombre legible
COBERTURA = {10: "Bosque", 20: "Matorral", 30: "Pastizal", 40: "Cultivo",
             50: "Urbano", 60: "Suelo desnudo", 70: "Nieve/hielo", 80: "Agua",
             90: "Humedal", 95: "Manglar", 100: "Musgo/liquen"}

# ---------------------------------------------------------------------------
# 1. agregaciones sobre TODO el dataset (polars lazy, sin cargar todo en ram)
# ---------------------------------------------------------------------------
print(">> escaneando dataset completo (lazy)...")
lf = pl.scan_csv(CSV_PATH)

n_total = lf.select(pl.len()).collect().item()
print(f"filas totales: {n_total:,}")

# balance de clases
balance = (lf.group_by(TARGET).agg(pl.len().alias("n"))
             .with_columns((pl.col("n") / n_total * 100).round(3).alias("pct"))
             .sort(TARGET).collect())
print("\nbalance de clases (fuego):")
print(balance)
balance.write_csv(OUT_DIR / "balance_clases.csv")

# conteo de nodata (-9999) por columna, sobre todo el dataset
print("\n>> contando nodata (-9999) por columna...")
nodata_counts = lf.select([(pl.col(c) == NODATA).sum().alias(c) for c in NUM_SENTINEL]).collect()
nd = nodata_counts.transpose(include_header=True, column_names=["n_nodata"])
nd = nd.with_columns((pl.col("n_nodata") / n_total * 100).round(3).alias("pct_nodata"))
print(nd)
nd.write_csv(OUT_DIR / "nodata_por_columna.csv")

# tasa de fuego y nodata-ndvi por clase de cobertura
#    revela si los negativos estan dominados por agua/roca (negativos triviales)
print("\n>> tasa de fuego por cobertura...")
por_cobertura = (lf.group_by("cobertura").agg([
        pl.len().alias("n"),
        pl.col(TARGET).mean().round(4).alias("tasa_fuego"),
        (pl.col("ndvi") == NODATA).mean().round(4).alias("pct_ndvi_nodata"),
    ]).sort("n", descending=True).collect())
por_cobertura = por_cobertura.with_columns(
    pl.col("cobertura").replace_strict(COBERTURA, default="otro").alias("clase"))
print(por_cobertura)
por_cobertura.write_csv(OUT_DIR / "tasa_fuego_por_cobertura.csv")

# tasa de fuego por mes (estacionalidad)
por_mes = (lf.group_by("mes").agg([
        pl.len().alias("n"),
        pl.col(TARGET).sum().alias("n_incendios"),
        pl.col(TARGET).mean().round(4).alias("tasa_fuego"),
    ]).sort("mes").collect())
por_mes.write_csv(OUT_DIR / "incendios_por_mes.csv")

# ---------------------------------------------------------------------------
# 2. muestra estratificada + limpieza de nodata
# ---------------------------------------------------------------------------
print("\n>> construyendo muestra estratificada...")
frac = min(1.0, N_SAMPLE / n_total)
partes = []
for clase in [0, 1]:
    parte = (lf.filter(pl.col(TARGET) == clase).collect().sample(fraction=frac, seed=SEED))
    partes.append(parte)
df = pl.concat(partes)
print(f"filas en la muestra: {df.height:,}")

# limpieza: -9999 -> null en las columnas centinela
df = df.with_columns([
    pl.when(pl.col(c) == NODATA).then(None).otherwise(pl.col(c)).alias(c) for c in NUM_SENTINEL
])

# feature engineering: magnitud del viento
df = df.with_columns((pl.col("viento_u") ** 2 + pl.col("viento_v") ** 2).sqrt().alias("wind_speed"))

# guardar la muestra limpia para reusar en el modelo
df.write_parquet(OUT_DIR / "muestra_estratificada.parquet")
print("muestra limpia guardada en eda_salidas/muestra_estratificada.parquet")

# columnas numericas para correlacion / mi (lat/lon incluidas como diagnostico de sesgo)
num_cols = ["latitude", "longitude", "elevation", "pendiente",
            "temperatura", "precipitacion", "wind_speed", "viento_u", "viento_v", "ndvi"]

# ---------------------------------------------------------------------------
# 3. estadistica descriptiva por clase + tamano de efecto
# ---------------------------------------------------------------------------
print("\n>> estadistica descriptiva por clase...")
medias = df.group_by(TARGET).agg([pl.col(c).mean().alias(c) for c in num_cols]).sort(TARGET)
medias.write_csv(OUT_DIR / "stats_media_por_clase.csv")
print(medias)

m0 = medias.filter(pl.col(TARGET) == 0).drop(TARGET).to_numpy().ravel()
m1 = medias.filter(pl.col(TARGET) == 1).drop(TARGET).to_numpy().ravel()
sd = df.select([pl.col(c).std().alias(c) for c in num_cols]).to_numpy().ravel()
cohen_d = (m1 - m0) / np.where(sd == 0, np.nan, sd)
efecto = pd.DataFrame({"variable": num_cols, "media_no_fuego": m0,
                       "media_fuego": m1, "cohen_d": np.round(cohen_d, 4)})
efecto = efecto.reindex(efecto["cohen_d"].abs().sort_values(ascending=False).index)
efecto.to_csv(OUT_DIR / "diferencia_medias_cohend.csv", index=False)
print("\nvariables por tamano de efecto (|cohen d|):")
print(efecto.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. correlaciones: pearson y spearman
# ---------------------------------------------------------------------------
print("\n>> correlaciones...")
pdf = df.select(num_cols + [TARGET]).drop_nulls().to_pandas()
print(f"filas usadas (sin nulos): {len(pdf):,}")

pearson = pdf.corr(method="pearson")
spearman = pdf.corr(method="spearman")
pearson.to_csv(OUT_DIR / "corr_pearson.csv")
spearman.to_csv(OUT_DIR / "corr_spearman.csv")

corr_target = pd.DataFrame({
    "pearson_con_fuego": pearson[TARGET].drop(TARGET),
    "spearman_con_fuego": spearman[TARGET].drop(TARGET),
})
corr_target["abs_spearman"] = corr_target["spearman_con_fuego"].abs()
corr_target = corr_target.sort_values("abs_spearman", ascending=False)
corr_target.to_csv(OUT_DIR / "correlacion_con_fuego.csv")
print("\ncorrelacion con fuego (ordenado por |spearman|):")
print(corr_target.round(4).to_string())

# heatmap de multicolinealidad
plt.figure(figsize=(11, 9))
mat = pearson.drop(TARGET).drop(TARGET, axis=1)
mask = np.triu(np.ones_like(mat, dtype=bool))
sns.heatmap(mat, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 9}, cbar_kws={"label": "pearson"})
plt.title("multicolinealidad entre features")
plt.tight_layout(); plt.savefig(FIG_DIR / "heatmap_multicolinealidad.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# 5. informacion mutua
# ---------------------------------------------------------------------------
print("\n>> informacion mutua...")
sub = pdf.sample(n=min(MI_SAMPLE, len(pdf)), random_state=SEED)
mi = mutual_info_classif(sub[num_cols].to_numpy(), sub[TARGET].to_numpy(),
                         discrete_features=False, random_state=SEED)
mi_df = pd.DataFrame({"variable": num_cols, "info_mutua": np.round(mi, 4)}).sort_values("info_mutua", ascending=False)
mi_df.to_csv(OUT_DIR / "informacion_mutua.csv", index=False)
print(mi_df.to_string(index=False))

plt.figure(figsize=(10, 7))
sns.barplot(data=mi_df, y="variable", x="info_mutua", color="#c1440e")
plt.title("informacion mutua con fuego"); plt.xlabel("info mutua (nats)"); plt.ylabel("")
plt.tight_layout(); plt.savefig(FIG_DIR / "info_mutua_fuego.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# 6. cramer's v: cobertura vs fuego
# ---------------------------------------------------------------------------
cont = df.select(["cobertura", TARGET]).to_pandas()
tabla = pd.crosstab(cont["cobertura"], cont[TARGET])
chi2, pval, _, _ = ss.chi2_contingency(tabla)
n = tabla.to_numpy().sum(); r, k = tabla.shape
cramer_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
print(f"\ncramer's v (cobertura vs fuego): {cramer_v:.4f} | p = {pval:.3e}")

tc = por_cobertura.to_pandas()
plt.figure(figsize=(11, 7))
sns.barplot(data=tc, y="clase", x="tasa_fuego", color="#2c7fb8")
plt.title(f"tasa de fuego por cobertura (cramer's v = {cramer_v:.3f})")
plt.xlabel("proporcion con incendio"); plt.ylabel("")
plt.tight_layout(); plt.savefig(FIG_DIR / "tasa_fuego_cobertura.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# 7. distribuciones por clase de las top features
# ---------------------------------------------------------------------------
print("\n>> distribuciones por clase...")
top = corr_target.index[:6].tolist()
plot_df = df.select(top + [TARGET]).drop_nulls()
plot_df = plot_df.sample(n=min(80_000, plot_df.height), seed=SEED).to_pandas()
plot_df[TARGET] = plot_df[TARGET].map({0: "no fuego", 1: "fuego"})

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, feat in zip(axes.ravel(), top):
    sns.violinplot(data=plot_df, x=TARGET, y=feat, hue=TARGET, legend=False, ax=ax, cut=0,
                   palette={"no fuego": "#4575b4", "fuego": "#d73027"})
    ax.set_title(feat); ax.set_xlabel("")
fig.suptitle("distribucion de top features por clase", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR / "distribuciones_por_clase.png", dpi=150, bbox_inches="tight"); plt.close()

print("\n>> listo. tablas en 'eda_salidas/', figuras en 'figuras/'.")
print(">> muestra limpia lista para el modelo: eda_salidas/muestra_estratificada.parquet")