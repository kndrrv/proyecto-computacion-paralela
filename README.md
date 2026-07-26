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