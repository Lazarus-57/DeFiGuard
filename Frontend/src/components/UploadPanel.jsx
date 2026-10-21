import React, { useRef, useState, useCallback } from 'react';
import { Upload, FileText, Database, AlertTriangle, Sparkles, Layers, Cpu } from 'lucide-react';
import { uploadCSV, loadSample } from '../api';

const SAMPLES = [
  {
    id: 'sample_100k_test',
    title: '100k Gold Standard Test',
    tag: 'Native Processed',
    description: '120 transactions with full pre-computed features (Peel + Smurf + Circular + Normal)',
  },
  {
    id: 'sample_etherscan_raw',
    title: 'Etherscan CSV Export',
    tag: 'Etherscan Schema',
    description: 'Raw Txhash, From, To, Value_IN(ETH), DateTime UTC — auto-aliased and graphed',
  },
  {
    id: 'sample_dune_wei',
    title: 'Dune Analytics Export',
    tag: 'Wei Scale (10¹⁸)',
    description: 'Raw transfers with amounts in Wei — automatically scaled to standard ETH',
  },
  {
    id: 'sample_bigquery_raw',
    title: 'Google BigQuery Export',
    tag: 'BigQuery / Unix Epoch',
    description: 'from_address, to_address, block_timestamp, uppercase hex — auto-harmonized',
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
          Universal Normalizer active: Harmonizing schema, auto-scaling Wei → ETH, computing graph topological metrics & running ML inference with SHAP explanations...
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Universal Normalizer Intelligence Banner */}
      <div className="normalizer-banner mb-6">
        <div className="normalizer-banner__icon">
          <Sparkles size={20} className="text-cyan" />
        </div>
        <div className="normalizer-banner__content">
          <div className="normalizer-banner__title">
            Universal Ethereum Dataset Normalization Engine Active
          </div>
          <div className="normalizer-banner__desc">
            DeFIGuard automatically standardizes arbitrary Ethereum CSVs: auto-detects Etherscan, Dune, BigQuery, or custom headers, scales Wei → ETH, and calculates graph topological metrics (PageRank, degrees, flow ratios) on the fly for raw datasets.
          </div>
        </div>
      </div>

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
              {(selectedFile.size / 1024).toFixed(1)} KB — Ready to normalize & analyze
            </div>
          </>
        ) : (
          <>
            <div className="upload-zone__label">
              Drop any Ethereum transaction CSV here, or click to browse
            </div>
            <div className="upload-zone__sublabel">
              Accepts Etherscan, Dune Analytics, BigQuery, or custom raw transfers (auto-scales Wei & extracts graph metrics)
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
            Normalize & Analyze Transactions
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

      {/* Multi-Format Sample Datasets */}
      <div className="mt-8">
        <div className="section-header mb-3">
          <div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Pre-Configured Multi-Format Test Datasets
            </div>
            <span className="text-xs text-muted">
              Click any dataset below to test DeFIGuard's multi-format auto-normalization capability:
            </span>
          </div>
        </div>
        <div className="samples-grid">
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              className="sample-btn"
              onClick={() => handleSampleLoad(s.id)}
              disabled={loading}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Database size={16} className="text-cyan" />
                  <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.85rem' }}>{s.title}</span>
                </div>
                <span className="badge badge--neutral" style={{ fontSize: '0.65rem', padding: '2px 6px' }}>
                  {s.tag}
                </span>
              </div>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textAlign: 'left', lineHeight: 1.4 }}>
                {s.description}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
