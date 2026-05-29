import { fetchDemandPrediction } from '../services/api';

export const ROUTES = [
  { id: 0, name: 'Ruta A' },
  { id: 1, name: 'Ruta B' },
  { id: 2, name: 'Ruta C' },
  { id: 3, name: 'Ruta D' },
  { id: 4, name: 'Ruta E' },
];

const CLIMA_MAP = { 'Lluvia': 0, 'Nublado': 1, 'Soleado': 2 };

const FS_MIN = [0, 1, 0];
const FS_MAX = [6, 12, 1];
const PAS_MIN = 346;
const PAS_MAX = 4039;

function normFeature(val, colIdx) {
  return (val - FS_MIN[colIdx]) / (FS_MAX[colIdx] - FS_MIN[colIdx]);
}

let cachedCsv = null;

async function loadCsv() {
  if (cachedCsv) return cachedCsv;
  const res = await fetch('/data/demanda_transporte.csv');
  const text = await res.text();
  const lines = text.trim().split('\n');
  const rows = lines.slice(1).map((line) => {
    const c = line.split(',');
    return {
      fecha: c[0],
      ruta: c[1],
      pasajeros: parseInt(c[2], 10),
      dia_semana: parseInt(c[4], 10),
      mes: parseInt(c[5], 10),
      festivo: parseInt(c[6], 10),
      clima: c[7].trim(),
    };
  });
  cachedCsv = rows;
  return rows;
}

export async function predictDemand(routeId) {
  const allRows = await loadCsv();
  const routeName = ROUTES.find((r) => r.id === Number(routeId))?.name ?? 'Ruta A';
  const routeRows = allRows
    .filter((r) => r.ruta === routeName)
    .sort((a, b) => new Date(a.fecha) - new Date(b.fecha));

  // need 60 rows: last 30 for input + 30 to predict
  const recent = routeRows.slice(-60);
  if (recent.length < 60) {
    throw new Error(`No hay suficientes datos históricos para ${routeName}`);
  }

  // first 30 rows = input sequence (30 days before the prediction window)
  const sequence = recent.slice(0, 30).map((r) => [
    normFeature(r.dia_semana, 0),
    normFeature(r.mes, 1),
    normFeature(r.festivo, 2),
    (r.pasajeros - PAS_MIN) / (PAS_MAX - PAS_MIN),
  ]);

  // next 30 rows = target days to predict
  const targetRows = recent.slice(30, 60);

  // normalized features + clima for each future step
  const futureFeatures = targetRows.map((r) => [
    normFeature(r.dia_semana, 0),
    normFeature(r.mes, 1),
    normFeature(r.festivo, 2),
  ]);
  const futureClimaIds = targetRows.map((r) => CLIMA_MAP[r.clima] ?? 1);

  const apiResult = await fetchDemandPrediction(
    sequence,
    routeId,
    futureClimaIds[0],
    futureFeatures,
    futureClimaIds,
  );
  const predictedValues = apiResult.predicciones;

  // build 60-day chart: 30 previous real values + 30 real vs predicted
  const prevDays = recent.slice(0, 30).map((r, i) => {
    const date = new Date(r.fecha);
    return {
      day: i + 1,
      date: date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' }),
      real: r.pasajeros,
    };
  });

  const predDays = targetRows.map((r, i) => {
    const date = new Date(r.fecha);
    return {
      day: i + 31,
      date: date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' }),
      real: r.pasajeros,
      predicted: Math.round(predictedValues[i]),
    };
  });

  const days = [...prevDays, ...predDays];

  const rmse = Math.round(
    Math.sqrt(predDays.reduce((sum, d) => sum + (d.real - d.predicted) ** 2, 0) / predDays.length),
  );
  const mae = Math.round(
    predDays.reduce((sum, d) => sum + Math.abs(d.real - d.predicted), 0) / predDays.length,
  );

  return { days, rmse, mae, routeName };
}

export const DRIVER_CLASSES = [
  { id: 'safe', label: 'Conducción Segura', icon: '✅', color: '#b8bb26' },
  { id: 'phone', label: 'Uso de Teléfono Móvil', icon: '📱', color: '#fb4934' },
  { id: 'drowsy', label: 'Somnolencia', icon: '😴', color: '#fe8019' },
  { id: 'eating', label: 'Comiendo o Bebiendo', icon: '🍔', color: '#fabd2f' },
  { id: 'looking', label: 'Mirando a los Lados', icon: '👀', color: '#d3869b' },
  { id: 'radio', label: 'Ajustando Radio', icon: '🎵', color: '#8ec07c' },
];

export function classifyDriver() {
  const idx = Math.floor(Math.random() * DRIVER_CLASSES.length);
  const driverClass = DRIVER_CLASSES[idx];
  const confidence = Math.round((0.72 + Math.random() * 0.22) * 100) / 100;
  const latency = 1500 + Math.random() * 1000;
  return { ...driverClass, confidence, latency };
}

export const DESTINATIONS = [
  { id: 'd1', name: 'Centro Histórico', icon: '🏛️', match: 95, category: 'Cultura' },
  { id: 'd2', name: 'Malecón Turístico', icon: '🌊', match: 88, category: 'Turismo' },
  { id: 'd3', name: 'Zona Comercial Norte', icon: '🛍️', match: 82, category: 'Compras' },
  { id: 'd4', name: 'Parque Industrial', icon: '🏭', match: 76, category: 'Negocios' },
  { id: 'd5', name: 'Plaza Central', icon: '⛲', match: 91, category: 'Recreación' },
  { id: 'd6', name: 'Terminal de Autobuses', icon: '🚌', match: 73, category: 'Transporte' },
  { id: 'd7', name: 'Hospital General', icon: '🏥', match: 68, category: 'Salud' },
  { id: 'd8', name: 'Ciudad Universitaria', icon: '🎓', match: 85, category: 'Educación' },
  { id: 'd9', name: 'Estadio Olímpico', icon: '🏟️', match: 78, category: 'Deportes' },
  { id: 'd10', name: 'Mercado Municipal', icon: '🏪', match: 80, category: 'Comercio' },
];

export function recommendDestinations(_clientId) {
  const shuffled = [...DESTINATIONS].sort(() => Math.random() - 0.5);
  const numRecs = 3 + Math.floor(Math.random() * 3);
  return shuffled.slice(0, numRecs).map((d) => ({
    ...d,
    match: Math.max(58, d.match - Math.floor(Math.random() * 18)),
  }));
}

export const FEATURE_IMPORTANCE = [
  { name: "Demanda Histórica", key: "hist_demand", importance: 0.321 },
  { name: "Incidentes Viales", key: "road_incidents", importance: 0.265 },
  { name: "Capacidad de Rutas", key: "route_capacity", importance: 0.184 },
  { name: "Tasa de Ocupación", key: "occupancy_rate", importance: 0.058 },
  { name: "Tiempo de Espera", key: "waiting_time", importance: 0.045 },
  { name: "Flujo de Pasajeros", key: "passenger_flow", importance: 0.038 },
  { name: "Eficiencia Operativa", key: "op_efficiency", importance: 0.031 },
  { name: "Tipo de Ruta", key: "route_type", importance: 0.029 },
  { name: "Distancia Recorrida", key: "travel_distance", importance: 0.019 },
  { name: "Periodo (30/90 Días)", key: "time_period", importance: 0.010 },
];

export { checkServerHealth } from '../services/api';
