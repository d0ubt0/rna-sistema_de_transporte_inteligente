export const ROUTES = [
  { id: 'r1', name: 'Ruta 1: Centro - Norte' },
  { id: 'r2', name: 'Ruta 2: Centro - Sur' },
  { id: 'r3', name: 'Ruta 3: Oriente - Poniente' },
  { id: 'r4', name: 'Ruta 4: Periférico' },
  { id: 'r5', name: 'Ruta 5: Aeropuerto - Centro' },
  { id: 'r6', name: 'Ruta 6: Terminal - Universidad' },
];

export function predictDemand(routeId) {
  const days = [];
  const now = new Date();
  let base = 3000 + Math.random() * 7000;

  for (let i = 0; i < 30; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() + i);
    const real = Math.round(base + (Math.random() - 0.5) * base * 0.3);
    const predicted = Math.round(base + (Math.random() - 0.5) * base * 0.18);
    days.push({
      day: i + 1,
      date: date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' }),
      real,
      predicted,
    });
    base += (Math.random() - 0.5) * 600;
    base = Math.max(800, base);
  }

  const rmse = Math.round(
    Math.sqrt(days.reduce((sum, d) => sum + (d.real - d.predicted) ** 2, 0) / days.length)
  );
  const mae = Math.round(
    days.reduce((sum, d) => sum + Math.abs(d.real - d.predicted), 0) / days.length
  );

  const routeName = ROUTES.find((r) => r.id === routeId)?.name ?? 'Ruta desconocida';

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

export async function checkServerHealth() {
  return true;
}
