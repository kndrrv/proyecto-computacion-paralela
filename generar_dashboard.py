"""
generar_dashboard.py
---------------------
Junta en una sola vista HTML todo lo que hasta ahora vivia en archivos
sueltos: el mapa de calor de riesgo (Folium), las metricas del ensemble
(PR-AUC, ROC-AUC, F1, importancia de variables) y las curvas de rendimiento
de Kabre (tiempo, speedup, eficiencia CPU vs GPU).


Uso:
    python generar_dashboard.py --output dashboard_incendios.html

Si algun archivo de entrada no existe, esa seccion del dashboard se marca
como "no disponible" en vez de romper el script -- para quese pueda correr
en distintas etapas del proyecto (por ejemplo antes de tener el benchmark
GPU) y regenerarlo despues.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ---------------------------------------------------------------------------
# config -- ajusta rutas si tu estructura de carpetas es distinta
# ---------------------------------------------------------------------------
PRED_CSV = "resultados_mapa_calor/predicciones_marzo.csv"                                   # lat, lon, probabilidad
GRID_CSV = "resultados_mapa_calor/grid_marzo_centroamerica.csv"        # lat, lon + features (para popups del mapa)
BENCH_CPU_CSV = "benchmark_entrenamiento.csv"                            # n_jobs, n_samples, tiempo_segundos
BENCH_GPU_CSV = "benchmark_entrenamiento_gpu.csv"                      # n_samples, tiempo_segundos
METRICS_CPU_CSV = "resultados_ensemble/metricas_ensemble.csv"          # modelo, pr_auc, roc_auc, f1_voto
METRICS_GPU_CSV = "resultados_ensemble_gpu/metricas_ensemble_gpu.csv"
MODEL_DIR_XGB = "modelos_ensemble"                                     # xgboost_sample*.joblib (CPU)

FEATURES = ["elevation", "pendiente", "temperatura", "precipitacion",
            "viento_u", "viento_v", "ndvi", "cobertura", "mes"]
FEATURE_LABELS = {
    "elevation": "Elevacion", "pendiente": "Pendiente", "temperatura": "Temperatura",
    "precipitacion": "Precipitacion", "viento_u": "Viento (u)", "viento_v": "Viento (v)",
    "ndvi": "NDVI", "cobertura": "Cobertura de suelo", "mes": "Mes",
}

TASA_BASE = 0.0090  # tasa base de incendio en el test, para contextualizar el mapa

# paleta -- misma escala que el gradiente del heatmap de folium, reusada
# como identidad visual del dashboard (ver build_css)
RISK_GRADIENT = ["#2e6ff2", "#3ec78f", "#e8d84a", "#e8892f", "#d1392b"]
BG = "#12161a"
PANEL = "#181d22"
INK = "#eef1f0"
MUTED = "#8b979a"
LINE = "#252c31"
CPU_COLOR = "#4fb0a5"
GPU_COLOR = "#e2621b"


def cargar_csv_seguro(ruta, **kwargs):
    p = Path(ruta)
    if not p.exists():
        print(f"[aviso] no se encontro {ruta}; esa seccion se marcara como no disponible")
        return None
    return pd.read_csv(p, **kwargs)


# ---------------------------------------------------------------------------
# 1. mapa de calor (folium) -- misma logica que generar_mapa.py, mas
#    popups con las features de la celda y capas conmutables
# ---------------------------------------------------------------------------
def build_map_html(df_pred, df_grid=None, top_n=20):
    if df_pred is None:
        return None, {}

    # si hay grid con features, la cruzamos por lat/lon para popups mas ricos
    # (redondeo porque ambos csv vienen del mismo sample() de GEE pero el
    # merge exacto de floats a veces falla por precision de punto flotante)
    df = df_pred.copy()
    tiene_features = False
    if df_grid is not None:
        df["_lat_r"] = df["lat"].round(4)
        df["_lon_r"] = df["lon"].round(4)
        g = df_grid.copy()
        g["_lat_r"] = g["lat"].round(4)
        g["_lon_r"] = g["lon"].round(4)
        cols_extra = [c for c in FEATURES if c in g.columns and c != "mes"]
        df = df.merge(g[["_lat_r", "_lon_r"] + cols_extra], on=["_lat_r", "_lon_r"], how="left")
        tiene_features = len(cols_extra) > 0

    mapa = folium.Map(location=[12.5, -85.0], zoom_start=6, tiles="CartoDB dark_matter")

    capa_calor = folium.FeatureGroup(name="Mapa de calor", show=True)
    puntos = df[["lat", "lon", "probabilidad"]].values.tolist()
    HeatMap(
        puntos, radius=8, blur=6, max_zoom=8,
        gradient={0.2: RISK_GRADIENT[0], 0.4: RISK_GRADIENT[1], 0.6: RISK_GRADIENT[2],
                  0.8: RISK_GRADIENT[3], 1.0: RISK_GRADIENT[4]},
    ).add_to(capa_calor)
    capa_calor.add_to(mapa)

    capa_top = folium.FeatureGroup(name=f"Top {top_n} celdas de riesgo", show=True)
    topN = df.sort_values("probabilidad", ascending=False).head(top_n)
    for _, fila in topN.iterrows():
        if tiene_features:
            filas_popup = "".join(
                f"<tr><td style='color:#8b979a;padding-right:8px'>{FEATURE_LABELS.get(c, c)}</td>"
                f"<td>{fila[c]:.2f}</td></tr>"
                for c in cols_extra if pd.notna(fila.get(c))
            )
            popup_html = (
                f"<div style='font-family:monospace;font-size:12px'>"
                f"<b>Probabilidad: {fila['probabilidad']:.4f}</b>"
                f"<table style='margin-top:6px'>{filas_popup}</table></div>"
            )
            popup = folium.Popup(popup_html, max_width=260)
        else:
            popup = None
        folium.CircleMarker(
            location=[fila["lat"], fila["lon"]], radius=6,
            color="#eef1f0", weight=1, fill=True, fill_color="#12161a", fill_opacity=0.85,
            tooltip=f"Probabilidad: {fila['probabilidad']:.4f}", popup=popup,
        ).add_to(capa_top)
    capa_top.add_to(mapa)

    folium.LayerControl(collapsed=False).add_to(mapa)

    kpis = {
        "celdas": len(df_pred),
        "prob_media": df_pred["probabilidad"].mean(),
        "prob_max": df_pred["probabilidad"].max(),
        "riesgo_relativo_max": df_pred["probabilidad"].max() / TASA_BASE,
    }
    # _repr_html_() esta pensado para notebooks (deja un placeholder de
    # "Trust Notebook" en vez del mapa real). get_root().render() da el
    # documento HTML completo y autonomo, que es lo que necesitamos aca.
    return mapa.get_root().render(), kpis


# ---------------------------------------------------------------------------
# 2. importancia de variables -- promedio sobre los xgboost del ensemble
# ---------------------------------------------------------------------------
def build_feature_importance_fig(model_dir):
    rutas = sorted(glob.glob(str(Path(model_dir) / "xgboost_sample*.joblib")))
    if not rutas:
        print(f"[aviso] no se encontraron modelos xgboost en {model_dir}")
        return None

    importancias = []
    for r in rutas:
        modelo = joblib.load(r)
        importancias.append(modelo.feature_importances_)
    prom = np.mean(importancias, axis=0)
    desv = np.std(importancias, axis=0)

    orden = np.argsort(prom)
    feats = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in orden]
    vals = prom[orden]
    errs = desv[orden]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        error_x=dict(type="data", array=errs, color=MUTED, thickness=1),
        marker_color=GPU_COLOR,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Importancia de variables (promedio, {len(rutas)} samples XGBoost)",
        xaxis_title="Importancia (gain)", margin=dict(l=10, r=10, t=50, b=10),
    )
    return apply_theme(fig)


# ---------------------------------------------------------------------------
# 3. metricas CPU vs GPU
# ---------------------------------------------------------------------------
def build_metrics_fig(cpu_csv, gpu_csv):
    cpu = cargar_csv_seguro(cpu_csv)
    gpu = cargar_csv_seguro(gpu_csv)
    if cpu is None and gpu is None:
        return None

    metricas = ["pr_auc", "roc_auc", "f1_voto"]
    nombres = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1_voto": "F1 (voto)"}

    fig = make_subplots(rows=1, cols=3, subplot_titles=[nombres[m] for m in metricas])
    for i, m in enumerate(metricas, start=1):
        if cpu is not None:
            fig.add_trace(go.Bar(x=cpu["modelo"], y=cpu[m], name="CPU",
                                  marker_color=CPU_COLOR, showlegend=(i == 1)),
                          row=1, col=i)
        if gpu is not None:
            fig.add_trace(go.Bar(x=gpu["modelo"], y=gpu[m], name="GPU",
                                  marker_color=GPU_COLOR, showlegend=(i == 1)),
                          row=1, col=i)
    fig.update_layout(title="Calidad del modelo: CPU (scikit-learn) vs GPU (cuML)",
                       barmode="group", margin=dict(l=10, r=10, t=70, b=10))
    return apply_theme(fig)


# ---------------------------------------------------------------------------
# 4. speedup / eficiencia CPU vs GPU
# ---------------------------------------------------------------------------
def build_speedup_fig(bench_cpu_csv, bench_gpu_csv):
    # estos csv se escriben con f.write() (modo "a"), sin encabezado -- ver
    # entrenar_ensemble_paralelo.py / entrenar_ensemble_gpu.py
    cpu = cargar_csv_seguro(bench_cpu_csv, header=None,
                             names=["n_jobs", "n_samples", "tiempo_segundos"])
    gpu = cargar_csv_seguro(bench_gpu_csv, header=None,
                             names=["n_samples", "tiempo_segundos"])
    if cpu is None and gpu is None:
        return None

    fig = make_subplots(rows=1, cols=3,
                         subplot_titles=["Tiempo total de entrenamiento", "Speedup vs 1 nucleo", "Eficiencia"])

    labels, tiempos, colores = [], [], []
    base = None
    if cpu is not None:
        # si el csv acumulo varias corridas para el mismo n_jobs (se escribe en
        # modo "a"), nos quedamos con el promedio de cada n_jobs
        cpu = cpu.groupby("n_jobs", as_index=False)["tiempo_segundos"].mean()
        cpu = cpu.sort_values("n_jobs")
        base = cpu.iloc[0]["tiempo_segundos"]
        for _, r in cpu.iterrows():
            labels.append(f"CPU {int(r['n_jobs'])} nucleo(s)")
            tiempos.append(r["tiempo_segundos"])
            colores.append(CPU_COLOR)
    if gpu is not None:
        t_gpu = gpu["tiempo_segundos"].iloc[-1]
        labels.append("GPU (cuML)")
        tiempos.append(t_gpu)
        colores.append(GPU_COLOR)

    fig.add_trace(go.Bar(x=labels, y=tiempos, marker_color=colores,
                          text=[f"{t:.1f}s" for t in tiempos], textposition="outside",
                          showlegend=False), row=1, col=1)

    if cpu is not None and base:
        speedup_cpu = base / cpu["tiempo_segundos"]
        eficiencia_cpu = speedup_cpu / cpu["n_jobs"]

        fig.add_trace(go.Scatter(x=cpu["n_jobs"], y=cpu["n_jobs"], mode="lines",
                                  line=dict(color=MUTED, dash="dash"), name="Ideal (lineal)"),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=cpu["n_jobs"], y=speedup_cpu, mode="lines+markers",
                                  line=dict(color=CPU_COLOR), name="CPU (medido)",
                                  hovertemplate="%{x} nucleos: %{y:.2f}x<extra></extra>"),
                      row=1, col=2)

        fig.add_trace(go.Scatter(x=cpu["n_jobs"], y=[1] * len(cpu), mode="lines",
                                  line=dict(color=MUTED, dash="dash"), showlegend=False),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=cpu["n_jobs"], y=eficiencia_cpu, mode="lines+markers",
                                  line=dict(color=CPU_COLOR), name="Eficiencia CPU", showlegend=False,
                                  hovertemplate="%{x} nucleos: %{y:.2f}<extra></extra>"),
                      row=1, col=3)

        if gpu is not None:
            speedup_gpu = base / gpu["tiempo_segundos"].iloc[-1]
            fig.add_trace(go.Scatter(x=[cpu["n_jobs"].max()], y=[speedup_gpu], mode="markers",
                                      marker=dict(color=GPU_COLOR, size=14), name="GPU (vs 1 nucleo)"),
                          row=1, col=2)

    fig.update_layout(title="Rendimiento en Kabre: CPU multinucleo vs GPU",
                       margin=dict(l=10, r=10, t=70, b=10))
    fig.update_xaxes(title_text="Nucleos (CPU)", row=1, col=2)
    fig.update_yaxes(title_text="Speedup", row=1, col=2)
    fig.update_xaxes(title_text="Nucleos (CPU)", row=1, col=3)
    fig.update_yaxes(title_text="Eficiencia (speedup / nucleos)", row=1, col=3)
    return apply_theme(fig)


def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=INK, family="IBM Plex Mono, monospace", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


# ---------------------------------------------------------------------------
# 5. ensamblar el HTML final
# ---------------------------------------------------------------------------
def fig_to_div(fig, first=False):
    if fig is None:
        return '<div class="no-data">no disponible todavia -- corre el script correspondiente y regenera el dashboard</div>'
    return pio.to_html(fig, full_html=False, include_plotlyjs=("cdn" if first else False),
                        config={"displaylogo": False})


def kpi_card(valor, etiqueta):
    return f'<div class="kpi"><div class="kpi-valor">{valor}</div><div class="kpi-etiqueta">{etiqueta}</div></div>'


def build_risk_table_html(df_pred, n=20):
    """Tabla ordenable (click en encabezado) con las n celdas de mayor riesgo."""
    if df_pred is None:
        return '<div class="no-data">no disponible: corre predict_heatmap.py primero</div>'

    top = df_pred.sort_values("probabilidad", ascending=False).head(n).reset_index(drop=True)
    filas = ""
    for i, r in enumerate(top.itertuples(), start=1):
        filas += (
            f"<tr><td>{i}</td><td data-v='{r.lat:.4f}'>{r.lat:.4f}</td>"
            f"<td data-v='{r.lon:.4f}'>{r.lon:.4f}</td>"
            f"<td data-v='{r.probabilidad:.6f}'>{r.probabilidad:.4f}</td>"
            f"<td data-v='{r.probabilidad / TASA_BASE:.4f}'>{r.probabilidad / TASA_BASE:.1f}x</td></tr>"
        )

    return f"""
    <table class="risk-table" id="risk-table">
      <thead><tr>
        <th data-sort="none">#</th>
        <th data-sort="num">Latitud</th>
        <th data-sort="num">Longitud</th>
        <th data-sort="num">Probabilidad</th>
        <th data-sort="num">Riesgo relativo</th>
      </tr></thead>
      <tbody>{filas}</tbody>
    </table>
    <script>
    (function() {{
      const table = document.getElementById('risk-table');
      const headers = table.querySelectorAll('th');
      headers.forEach((th, idx) => {{
        if (th.dataset.sort === 'none') return;
        th.style.cursor = 'pointer';
        let asc = false;
        th.addEventListener('click', () => {{
          asc = !asc;
          const rows = Array.from(table.querySelectorAll('tbody tr'));
          rows.sort((a, b) => {{
            const av = parseFloat(a.children[idx].dataset.v);
            const bv = parseFloat(b.children[idx].dataset.v);
            return asc ? av - bv : bv - av;
          }});
          const tbody = table.querySelector('tbody');
          rows.forEach(r => tbody.appendChild(r));
        }});
      }});
    }})();
    </script>
    """


def build_css():
    grad = ", ".join(RISK_GRADIENT)
    return f"""
    :root {{
      --bg: {BG}; --panel: {PANEL}; --ink: {INK}; --muted: {MUTED}; --line: {LINE};
      --cpu: {CPU_COLOR}; --gpu: {GPU_COLOR};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg); color: var(--ink); margin: 0;
      font-family: 'IBM Plex Mono', monospace;
    }}
    h1, h2 {{ font-family: 'Newsreader', serif; font-weight: 500; margin: 0; }}
    header {{
      padding: 40px clamp(20px, 5vw, 64px) 24px; border-bottom: 1px solid var(--line);
    }}
    header .eyebrow {{
      color: var(--muted); letter-spacing: .12em; text-transform: uppercase; font-size: 12px;
    }}
    header h1 {{ font-size: clamp(28px, 4vw, 42px); margin-top: 6px; }}
    header p {{ color: var(--muted); max-width: 60ch; line-height: 1.5; margin-top: 10px; }}
    .scale {{
      height: 4px; width: 100%; margin-top: 24px;
      background: linear-gradient(90deg, {grad});
    }}
    main {{ padding: 32px clamp(20px, 5vw, 64px) 64px; display: flex; flex-direction: column; gap: 28px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 2px; }}
    section .head {{
      padding: 18px 22px; border-bottom: 1px solid var(--line);
      display: flex; align-items: baseline; justify-content: space-between;
    }}
    section .head h2 {{ font-size: 18px; }}
    section .head span {{ color: var(--muted); font-size: 12px; }}
    section .body {{ padding: 18px 22px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1px;
             background: var(--line); border: 1px solid var(--line); }}
    .kpi {{ background: var(--panel); padding: 20px; }}
    .kpi-valor {{ font-family: 'Newsreader', serif; font-size: 30px; color: var(--gpu); }}
    .kpi-etiqueta {{ color: var(--muted); font-size: 12px; margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }}
    @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .map-frame {{ width: 100%; height: 560px; border: 0; display: block; }}
    .no-data {{ color: var(--muted); padding: 40px; text-align: center; font-size: 13px; }}
    footer {{ padding: 20px clamp(20px, 5vw, 64px) 40px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); }}

    nav.tabs {{
      position: sticky; top: 0; z-index: 10; background: rgba(18,22,26,.92);
      backdrop-filter: blur(6px); border-bottom: 1px solid var(--line);
      display: flex; gap: 4px; padding: 0 clamp(20px, 5vw, 64px);
    }}
    nav.tabs a {{
      color: var(--muted); text-decoration: none; font-size: 12px;
      text-transform: uppercase; letter-spacing: .06em; padding: 14px 4px;
      border-bottom: 2px solid transparent;
    }}
    nav.tabs a:hover {{ color: var(--ink); border-bottom-color: var(--gpu); }}

    .risk-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .risk-table th, .risk-table td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--line); }}
    .risk-table th:first-child, .risk-table td:first-child {{ text-align: left; color: var(--muted); }}
    .risk-table th {{ color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: .05em;
                      user-select: none; }}
    .risk-table th:hover {{ color: var(--gpu); }}
    .risk-table tbody tr:hover {{ background: rgba(226, 98, 27, 0.08); }}
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dashboard_incendios.html")
    args = parser.parse_args()

    df_pred = cargar_csv_seguro(PRED_CSV)
    df_grid = cargar_csv_seguro(GRID_CSV)
    map_html, kpis = build_map_html(df_pred, df_grid)
    tabla_riesgo_html = build_risk_table_html(df_pred)
    fig_importancia = build_feature_importance_fig(MODEL_DIR_XGB)
    fig_metricas = build_metrics_fig(METRICS_CPU_CSV, METRICS_GPU_CSV)
    fig_speedup = build_speedup_fig(BENCH_CPU_CSV, BENCH_GPU_CSV)

    kpi_html = ""
    if kpis:
        kpi_html = "".join([
            kpi_card(f"{kpis['celdas']:,}", "Celdas evaluadas"),
            kpi_card(f"{kpis['prob_media']:.4f}", "Probabilidad promedio"),
            kpi_card(f"{kpis['prob_max']:.4f}", "Probabilidad maxima"),
            kpi_card(f"{kpis['riesgo_relativo_max']:.1f}x", "Riesgo relativo max. vs base"),
        ])
    else:
        kpi_html = '<div class="no-data">no disponible: corre predict_heatmap.py y ajusta PRED_CSV</div>'

    map_section = (
        f'<iframe class="map-frame" srcdoc="{map_html.replace(chr(34), "&quot;")}"></iframe>'
        if map_html else '<div class="no-data">no disponible: corre predict_heatmap.py + generar_mapa.py primero</div>'
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Riesgo de incendios forestales -- Centroamerica</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{build_css()}</style>
</head>
<body>
<header>
  <div class="eyebrow">Computacion Paralela y Distribuida &middot; LEAD University</div>
  <h1>Riesgo de incendios forestales en Centroamerica</h1>
  <p>Ensemble de regresion logistica, random forest y XGBoost entrenado por bootstrapping
     balanceado sobre ~15.6M de observaciones geoespaciales. Escenario: marzo (epoca seca).
     Entrenamiento acelerado en el cluster Kabre (CENAT), CPU multinucleo vs GPU (RAPIDS cuML).</p>
  <div class="scale"></div>
</header>
<nav class="tabs">
  <a href="#resumen">Resumen</a>
  <a href="#mapa">Mapa</a>
  <a href="#riesgo">Top celdas</a>
  <a href="#modelo">Modelo</a>
  <a href="#rendimiento">Rendimiento</a>
</nav>
<main>

  <section id="resumen">
    <div class="head"><h2>Resumen del escenario</h2><span>marzo &middot; epoca seca</span></div>
    <div class="body"><div class="kpis">{kpi_html}</div></div>
  </section>

  <section id="mapa">
    <div class="head"><h2>Mapa de calor de riesgo</h2><span>Folium &middot; click en un punto para ver sus features</span></div>
    <div class="body">{map_section}</div>
  </section>

  <section id="riesgo">
    <div class="head"><h2>Celdas de mayor riesgo</h2><span>click en un encabezado para ordenar</span></div>
    <div class="body">{tabla_riesgo_html}</div>
  </section>

  <div class="grid-2" id="modelo">
    <section>
      <div class="head"><h2>Importancia de variables</h2><span>promedio XGBoost, 25 samples</span></div>
      <div class="body">{fig_to_div(fig_importancia, first=True)}</div>
    </section>
    <section>
      <div class="head"><h2>Calidad del modelo</h2><span>CPU vs GPU</span></div>
      <div class="body">{fig_to_div(fig_metricas)}</div>
    </section>
  </div>

  <section id="rendimiento">
    <div class="head"><h2>Rendimiento computacional (Kabre)</h2><span>tiempo, speedup, eficiencia</span></div>
    <div class="body">{fig_to_div(fig_speedup)}</div>
  </section>

</main>
<footer>
  Cluster: Kabre (CENAT) &middot; CPU: particion kura &middot; GPU: particion nukwa (Tesla V100, 32GB) &middot; RAPIDS cuML.
  Generado con generar_dashboard.py.
</footer>
</body>
</html>"""

    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Dashboard guardado en {args.output}")


if __name__ == "__main__":
    main()