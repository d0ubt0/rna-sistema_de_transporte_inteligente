const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchDemandPrediction(sequence, routeId, climaId) {
  const res = await fetch(`${API_BASE}/demand/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sequence,
      route_id: routeId,
      clima_id: climaId,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}

export async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE}/`);
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === 'online';
  } catch {
    return false;
  }
}
