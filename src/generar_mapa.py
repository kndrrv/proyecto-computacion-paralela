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

    mapa.save(args.output)
    print(f"Mapa guardado en {args.output}")

    print("\n--- Resumen para el informe ---")
    print(f"Celdas totales: {len(df)}")
    print(f"Probabilidad promedio: {df['probabilidad'].mean():.4f}")
    print(f"Probabilidad maxima: {df['probabilidad'].max():.4f}")
    top10 = df.sort_values('probabilidad', ascending=False).head(10)
    print("\nTop 10 celdas de mayor riesgo (lat, lon, probabilidad):")
    print(top10[["lat", "lon", "probabilidad"]].to_string(index=False))


if __name__ == "__main__":
    main()
