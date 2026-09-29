import React, { useRef, useState, useCallback } from 'react';
import { Upload, FileText, Database, AlertTriangle } from 'lucide-react';
import { uploadCSV, loadSample } from '../api';

const SAMPLES = [
  {
    id: 'sample_100k_test',
    title: '100k Gold Standard Test',
    description: '120 transactions — Peel + Smurf + Circular + Normal (full features)',
  },
  {
    id: 'sample_raw_transfers',
    title: 'Raw Ethereum Transfers',
    description: '120 raw transfers — tx_hash, from, to, amount, block_time only',
  },
];

export default function UploadPanel({ onResults, setLoading, loading }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.name.endsWith('.csv')) {
      setSelectedFile(file);
      setError(null);
    } else {
      setError('Please upload a .csv file');
    }
  }, []);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await uploadCSV(selectedFile);
      onResults(data);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setError(`Upload failed: ${detail}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleLoad = async (sampleId) => {
    setLoading(true);
    setError(null);
    setSelectedFile(null);
    try {
      const data = await loadSample(sampleId);
      onResults(data);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setError(`Failed to load sample: ${detail}`);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <p className="loading-overlay__text">
          Running ML inference engine... Analyzing transactions with XGBoost + GNN + NTS hybrid model and computing SHAP explanations
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Upload Zone */}
      <div
        className={`upload-zone ${dragActive ? 'upload-zone--active' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload className="upload-zone__icon" />
        {selectedFile ? (
          <>
            <div className="upload-zone__label">
              <FileText size={16} style={{ display: 'inline', marginRight: 6, verticalAlign: 'text-bottom' }} />
              {selectedFile.name}
            </div>
            <div className="upload-zone__sublabel">
              {(selectedFile.size / 1024).toFixed(1)} KB — Ready to analyze
            </div>
          </>
        ) : (
          <>
            <div className="upload-zone__label">
              Drop your transaction CSV here, or click to browse
            </div>
            <div className="upload-zone__sublabel">
              Supports raw transfers (tx_hash, from, to, amount) or feature-engineered datasets
            </div>
          </>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />
      </div>

      {/* Upload Button */}
      {selectedFile && (
        <div style={{ textAlign: 'center', marginTop: 16 }}>
          <button className="btn btn--primary" onClick={handleUpload} disabled={loading}>
            <Upload size={16} />
            Analyze Transactions
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="error-banner mt-4">
          <AlertTriangle />
          {error}
        </div>
      )}

      {/* Sample Datasets */}
      <div className="mt-6">
        <div className="section-header">
          <span className="text-sm text-muted">Or quick-load a sample dataset for testing:</span>
        </div>
        <div className="samples-row">
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              className="sample-btn"
              onClick={() => handleSampleLoad(s.id)}
              disabled={loading}
            >
              <Database />
              <div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>{s.title}</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>{s.description}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
