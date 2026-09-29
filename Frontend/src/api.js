import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 min — large CSV inference can take time
});

/**
 * Upload a CSV file for AML prediction.
 * POST /predict (multipart/form-data)
 */
export async function uploadCSV(file) {
  const formData = new FormData();
  formData.append('file', file);
  const resp = await api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return resp.data;
}

/**
 * Load and predict on a preloaded sample dataset.
 * GET /samples/{sampleId}
 */
export async function loadSample(sampleId) {
  const resp = await api.get(`/samples/${sampleId}`);
  return resp.data;
}

/**
 * List available sample datasets.
 * GET /samples
 */
export async function listSamples() {
  const resp = await api.get('/samples');
  return resp.data;
}

/**
 * Get model architecture and performance info.
 * GET /model-info
 */
export async function getModelInfo() {
  const resp = await api.get('/model-info');
  return resp.data;
}

/**
 * Health check.
 * GET /health
 */
export async function getHealth() {
  const resp = await api.get('/health');
  return resp.data;
}

export default api;
