"""
generar_mapa.py
---------------
Toma el CSV de predicciones (lat, lon, probabilidad) generado en Kabre
y pinta el mapa de calor de riesgo sobre Centroamerica, listo para el
dashboard. Esto se corre en tu computadora, no en Kabre.

Uso:
    python generar_mapa.py --input predicciones_marzo.csv \
                            --output mapa_riesgo_marzo.html
"""

import argparse
import pandas as pd
import folium
from folium.plugins import HeatMap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="mapa_riesgo.html")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    mapa = folium.Map(location=[12.5, -85.0], zoom_start=6, tiles="CartoDB positron")

    puntos = df[["lat", "lon", "probabilidad"]].values.tolist()

    HeatMap(
        puntos,
        radius=8,
        blur=6,
        max_zoom=8,
        gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
    ).add_to(mapa)

    # Marcadores sobre las celdas de mayor riesgo, para poder identificarlas
    # a simple vista (el heatmap solo muestra el patron general, no permite
    # distinguir puntos exactos por el gradiente continuo).
    top_n = df.sort_values("probabilidad", ascending=False).head(10)
    for _, fila in top_n.iterrows():
        folium.CircleMarker(
            location=[fila["lat"], fila["lon"]],
            radius=6,
            color="black",
            weight=1,
            fill=True,
            fill_color="white",
            fill_opacity=0.9,
            tooltip=f"Probabilidad: {fila['probabilidad']:.4f}",
            popup=f"lat: {fila['lat']:.3f}, lon: {fila['lon']:.3f}<br>probabilidad: {fila['probabilidad']:.4f}",
        ).add_to(mapa)

    mapa.save(args.output)
    print(f"Mapa guardado en {args.output}")

    print("\n--- Resumen para el informe ---")
    print(f"Celdas totales: {len(df)}")
    print(f"Probabilidad promedio: {df['probabilidad'].mean():.4f}")
    print(f"Probabilidad maxima: {df['probabilidad'].max():.4f}")
    print("\nTop 10 celdas de mayor riesgo (lat, lon, probabilidad):")
    print(top_n[["lat", "lon", "probabilidad"]].to_string(index=False))


if __name__ == "__main__":
    main()
