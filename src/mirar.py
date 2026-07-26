import polars as pl

# nombre del csv
ruta = "dataset_gee_centroamerica_completo.csv"

# primeras 10 filas, sin cargar el archivo completo
vistazo = pl.read_csv(ruta, n_rows=10)
print("=== primeras filas ===")
print(vistazo)

print("\n=== columnas y tipos ===")
print(vistazo.schema)

# contar filas totales en streaming 
total = pl.scan_csv(ruta).select(pl.len()).collect(streaming=True).item()
print(f"\nfilas totales: {total:,}")