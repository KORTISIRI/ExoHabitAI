/**
 * ExoHabitAI — Centralized API Service
 * All backend interactions go through this module.
 */

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// ── Custom error class ────────────────────────────────────────────────────────
export class ApiError extends Error {
  constructor(message, statusCode = 0, code = 'UNKNOWN', details = null) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
  }
}

// ── Generic fetch wrapper ─────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  let response;

  try {
    response = await fetch(url, { ...options });
  } catch (networkErr) {
    throw new ApiError(
      `Cannot reach the ExoHabitAI backend at ${BASE_URL}. ` +
        'Make sure the Flask server is running: python app.py',
      0,
      'NETWORK_ERROR'
    );
  }

  const ct = response.headers.get('content-type') || '';
  let body;
  try {
    body = ct.includes('application/json') ? await response.json() : await response.text();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const message =
      (body && body.message) ||
      (body && body.error) ||
      `Server returned HTTP ${response.status}`;
    throw new ApiError(message, response.status, 'HTTP_ERROR', body);
  }

  if (body && body.status === 'error') {
    throw new ApiError(body.message || 'Backend returned an error.', response.status, 'API_ERROR', body);
  }

  return body;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

/** Health check — resolves { model_loaded, message } */
export async function healthCheck() {
  return apiFetch('/health');
}

/** Single planet prediction */
export async function predictSingle(payload) {
  return apiFetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

/** Batch prediction from a JSON array */
export async function predictBatchJson(items) {
  return apiFetch('/predict-batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(items),
  });
}

/** Batch prediction from a File object (CSV or JSON) */
export async function predictBatchFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  return apiFetch('/predict-batch', { method: 'POST', body: fd });
}

/** Train-and-predict from raw CSV text */
export async function trainAndPredictCsv(csvText) {
  return apiFetch('/train-and-predict', {
    method: 'POST',
    headers: { 'Content-Type': 'text/csv' },
    body: csvText,
  });
}

/** Train-and-predict from a parsed JSON array */
export async function trainAndPredictJson(items) {
  return apiFetch('/train-and-predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(items),
  });
}

/** Train-and-predict from a File object */
export async function trainAndPredictFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  return apiFetch('/train-and-predict', { method: 'POST', body: fd });
}

/** GET /rank — fetch top-N pre-ranked exoplanets */
export async function getRankings(top = 10) {
  return apiFetch(`/rank?top=${top}`);
}

/** POST /rank — rank a submitted list of exoplanets */
export async function rankCustom(items) {
  return apiFetch('/rank', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(items),
  });
}
