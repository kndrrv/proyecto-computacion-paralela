var MES = 3;  // marzo
var ESCALA_METROS = 5000;  // ~5 km entre celdas

var aoi = ee.Geometry.Rectangle([-92.5, 7.0, -77.0, 18.5]);

var ndvi = ee.ImageCollection('MODIS/061/MOD13A2')
  .filter(ee.Filter.calendarRange(MES, MES, 'month'))
  .select('NDVI')
  .mean()
  .multiply(0.0001)
  .rename('ndvi');

var era5Marzo = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filter(ee.Filter.calendarRange(MES, MES, 'month'));

var temperatura = era5Marzo.select('temperature_2m').mean().subtract(273.15).rename('temperatura');
var precipitacion = era5Marzo.select('total_precipitation_sum').mean().multiply(1000).rename('precipitacion');
var vientoU = era5Marzo.select('u_component_of_wind_10m').mean().rename('viento_u');
var vientoV = era5Marzo.select('v_component_of_wind_10m').mean().rename('viento_v');

var srtm = ee.Image('USGS/SRTMGL1_003');
var elevation = srtm.rename('elevation');
var pendiente = ee.Terrain.slope(srtm).rename('pendiente');

var cobertura = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('cobertura');

var coords = ee.Image.pixelLonLat().rename(['lon', 'lat']);

var imagen = coords
  .addBands(ndvi)
  .addBands(temperatura)
  .addBands(precipitacion)
  .addBands(vientoU)
  .addBands(vientoV)
  .addBands(elevation)
  .addBands(pendiente)
  .addBands(cobertura);

var grid = imagen.sample({
  region: aoi,
  scale: ESCALA_METROS,
  geometries: false,
  dropNulls: true
});

Export.table.toDrive({
  collection: grid,
  description: 'grid_marzo_centroamerica',
  fileFormat: 'CSV',
  selectors: ['lat', 'lon', 'ndvi', 'temperatura', 'precipitacion',
              'viento_u', 'viento_v', 'elevation', 'pendiente', 'cobertura']
});

print('Script listo. Anda a la pestana Tasks arriba a la derecha y dale RUN a la tarea.');
