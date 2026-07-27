# Estimación de Ocurrencia de Incendios Forestales en Centroamérica

Proyecto del curso de Computación Paralela y Distribuida — LEAD University.

Estimación de la probabilidad de ocurrencia de incendios forestales en Centroamérica
mediante aprendizaje automático supervisado, a partir de variables geoespaciales
multicanal (clima, vegetación, topografía y cobertura del suelo), con generación de
un mapa de calor de riesgo y evaluación de rendimiento computacional en cómputo
paralelo y distribuido.



## Fuentes de datos

| Fuente | Variable | ID en Google Earth Engine | Resolución |
|---|---|---|---|
| NASA FIRMS | Focos activos (objetivo) | `FIRMS` | ~1 km |
| MODIS Terra | NDVI | `MODIS/061/MOD13A2` | ~1 km / 16 días |
| ECMWF ERA5-Land | Clima | `ECMWF/ERA5_LAND/DAILY_AGGR` | ~9 km / diario |
| USGS SRTM | Elevación | `USGS/SRTMGL1_003` | 30 m |
| ESA WorldCover | Cobertura de suelo | `ESA/WorldCover/v200` | 10 m |

Dataset consolidado: ~15.6 millones de registros, ~2.14 GB. **No se sube a este
repositorio** por su tamaño; se distribuye vía Google Drive (enlace: *pendiente*).

## Estructura del repositorio

```
.
├── src/
│   ├── mirar.py                 # inspección rápida del csv (schema, filas)
│   ├── eda.py                   # análisis exploratorio + correlaciones
│   └── modelo_tes.py            # modelo actual: ensemble por bootstrapping balanceado
├── docs/
│   └── main.tex                 # informe IEEE (Entrega 1 + resultados)
├── resultados_modelo/           # métricas y figuras del modelo base (generado)
├── resultados_ensemble/         # métricas y figuras del ensemble (generado)
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

```powershell
pip install -r requirements.txt
```

## Cómo reproducir

Todos los scripts asumen que `dataset_gee_centroamerica_completo.csv` está en la
misma carpeta desde la que se ejecutan (o se ajusta la variable `CSV_PATH` al
inicio de cada script).

1. **Análisis exploratorio** — genera tablas y figuras en `eda_salidas/` y `figuras/`,
   y deja una muestra estratificada limpia lista para el modelo:
   ```powershell
   python src/eda_incendios.py
   ```

2. **Ensemble por bootstrapping** (modelo de referencia — ver
   `docs/main.tex`, sección de Resultados Preliminares):
   ```powershell
   python src/ensemble_bootstrap.py
   ```
   Configuración ajustable al inicio del script: `N_SAMPLES`, `N_POR_CLASE`,
   `CON_REEMPLAZO`. Local se corre con pocos samples (5); en Kabré se sube a 25-30,
   distribuyendo cada sample como un job independiente.

## Estado actual

- [x] Pipeline de adquisición e integración de las 5 fuentes (Entrega 1)
- [x] Análisis exploratorio y correlaciones
- [x] Modelo base con pesos de clase
- [x] Ensemble por bootstrapping balanceado (con/sin reemplazo comparados)
- [ ] Benchmark de rendimiento en Kabré (CPU multinúcleo, Dask distribuido, GPU)
- [ ] Mapa de calor sobre grilla (Google Earth Engine)
- [ ] Dashboard interactivo (Plotly/Folium)
- [ ] Informe final e integración

## Resultados actuales (referencia)

Ensemble por bootstrapping, 5 samples, sin reemplazo, split temporal
(train 2011–2020, test 2021–2025):

| Modelo | PR-AUC | ROC-AUC |
|---|---|---|
| Regresión logística | 0.0909 | 0.8976 |
| Random Forest | 0.1127 | 0.9295 |
| **XGBoost** | **0.1274** | **0.9326** |

Tasa base de incendio en el conjunto de prueba: 0.0090 (XGBoost ≈ 14× el azar).
## Mapa de calor de riesgo (completado)

Pipeline para generar el mapa de riesgo de incendio sobre toda Centroamérica
(no solo puntos del dataset), corrido en Kabré.

**Scripts** (en `src/`):
- `entrenar_ensemble_paralelo.py` — versión de `modelo_tes.py` que reparte
  los samples del ensemble entre núcleos con `joblib.Parallel`, en vez de
  entrenarlos secuencialmente. Uso:
Guarda log de tiempos en `benchmark_entrenamiento.csv` (columnas:
  n_jobs, n_samples, tiempo_segundos) — insumo directo para la tabla de
  speedup/eficiencia CPU.
- `generar_grid_marzo.js` — script de Google Earth Engine (se corre en
  code.earthengine.google.com, no en Kabré) que genera una cuadrícula de
  Centroamérica a 5 km con las condiciones típicas de marzo (época seca) de
  las mismas 5 fuentes del dataset original. Exporta CSV a Google Drive.
- `predict_heatmap.py` — carga el ensemble de XGBoost ya entrenado
  (`modelos_ensemble/xgboost_sample*.joblib`) y predice la probabilidad de
  incendio sobre cada celda de la cuadrícula, en paralelo. Uso:
- `generar_mapa.py` — pinta el CSV de probabilidades como mapa de calor
  interactivo (Folium), corre en cualquier compu (no necesita Kabré).

**Resultados** (en `resultados_mapa_calor/`):
- `grid_marzo_centroamerica.csv` — cuadrícula de 24,095 celdas con features
- `predicciones_marzo.csv` — probabilidad de incendio por celda (lat, lon, probabilidad)
- `mapa_riesgo_marzo.html` — mapa interactivo, listo para el dashboard

**Modelos entrenados**: `modelos_ensemble.tar.gz` (132MB, no está en git por
tamaño) está en la carpeta de Drive del proyecto, junto al dataset. Contiene
5 samples × 3 modelos (logística, random forest, xgboost) entrenados en
Kabré (partición `kura`, 4 núcleos).

## Pendiente: benchmark de rendimiento en Kabré

Falta la comparación CPU vs GPU que menciona el informe (Sección VI.D,
cuML de RAPIDS). Plan:

1. **Benchmark CPU**: correr `entrenar_ensemble_paralelo.py` en la
   partición `kura` con distintos `--n-jobs` (1, 2, 4, 8, 16...) y armar
   la tabla de speedup/eficiencia con `benchmark_entrenamiento.csv`.
2. **Benchmark GPU (falta escribir el script)**: misma lógica pero
   reemplazando `RandomForestClassifier`/`XGBClassifier` de sklearn por sus
   equivalentes de cuML (`cuml.ensemble.RandomForestClassifier`, y
   `XGBClassifier(device="cuda")`), corriendo en la partición `nukwa` (GPU).
   Comparar tiempos CPU vs GPU para el mismo N_SAMPLES.
