import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  Upload,
  FileText,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
} from 'lucide-react';
import { uploadCSV, loadSample } from '../api';

const SAMPLES = [
  {
    id: 'sample_100k_test',
    title: 'Gold Standard Test',
    tag: 'Native',
    description: '120 pre-featured transactions covering peel, smurfing, circular and normal flows.',
  },
  {
    id: 'sample_etherscan_raw',
    title: 'Etherscan Export',
    tag: 'Etherscan',
    description: 'Raw Etherscan CSV columns, auto-mapped and graphed.',
  },
  {
    id: 'sample_dune_wei',
    title: 'Dune Export',
    tag: 'Wei scale',
    description: 'Amounts in Wei, automatically scaled to ETH.',
  },
  {
    id: 'sample_bigquery_raw',
    title: 'BigQuery Export',
    tag: 'BigQuery',
    description: 'Uppercase hex and Unix timestamps, auto-harmonized.',
  },
];

const STAGES = [
  'Detecting schema',
  'Building wallet graph',
  'Scoring transactions',
  'Computing SHAP explanations',
];

function LoadingState() {
  const [stage, setStage] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 2500);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="loading-overlay">
      <div className="loading-cyber-box">
        <div className="spinner" />
        <h3 className="loading-cyber-title">Analyzing</h3>
        <div className="loading-steps-list">
          {STAGES.map((label, i) => (
            <div
              key={label}
              className={`loading-step ${i < stage ? 'loading-step--done' : ''} ${i === stage ? 'loading-step--active' : ''}`}
            >
              <CheckCircle2 size={14} style={{ opacity: i <= stage ? 1 : 0.25 }} />
              {label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function UploadPanel({ onResults, setLoading, loading }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preflightInfo, setPreflightInfo] = useState(null);
  const [error, setError] = useState(null);

  // Read the header locally to preview the detected schema
  const inspectCSV = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const lines = String(e.target.result).split(/\r\n|\n/).filter((l) => l.trim().length > 0);
        if (lines.length === 0) return;
        const header = lines[0].toLowerCase();
        const cols = lines[0].split(',').map((c) => c.replace(/['"]+/g, '').trim());

        let detectedType = 'Generic CSV';
        if (header.includes('value_in(eth)') || header.includes('txnhash') || header.includes('datetime (utc)')) {
          detectedType = 'Etherscan';
        } else if (header.includes('from_address') && header.includes('to_address') && header.includes('block_timestamp')) {
          detectedType = 'BigQuery / Dune';
        } else if (header.includes('transactionhash') && header.includes('callingfunction')) {
          detectedType = 'XBlock-ETH';
        }
        setPreflightInfo({ columns: cols.length, detectedType });
      } catch {
        setPreflightInfo(null);
      }
    };
    reader.readAsText(file.slice(0, 16384));
  }, []);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  }, []);

  const acceptFile = useCallback((file) => {
    setSelectedFile(file);
    setError(null);
    inspectCSV(file);
  }, [inspectCSV]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.name.toLowerCase().endsWith('.csv')) acceptFile(file);
    else setError('Please upload a .csv file.');
  }, [acceptFile]);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) acceptFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      onResults(await uploadCSV(selectedFile));
    } catch (err) {
      setError(`Upload failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleLoad = async (sampleId) => {
    setLoading(true);
    setError(null);
    setSelectedFile(null);
    setPreflightInfo(null);
    try {
      onResults(await loadSample(sampleId));
    } catch (err) {
      setError(`Failed to load sample: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingState />;

  return (
    <div>
      <div
        className={`upload-zone ${dragActive ? 'upload-zone--active' : ''} ${selectedFile ? 'upload-zone--has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="upload-zone__icon-glow">
          <Upload className="upload-zone__icon" />
        </div>

        {selectedFile ? (
          <>
            <div className="flex items-center gap-2">
              <FileText size={16} className="text-secondary" />
              <span className="upload-zone__label">{selectedFile.name}</span>
            </div>
            <div className="upload-zone__sublabel">
              {(selectedFile.size / 1024).toFixed(1)} KB
            </div>
            {preflightInfo && (
              <div className="preflight-tags mt-2">
                <span className="badge badge--info">{preflightInfo.detectedType}</span>
                <span className="badge badge--neutral">{preflightInfo.columns} columns</span>
              </div>
            )}
          </>
        ) : (
          <>
            <div className="upload-zone__label">Drop a CSV here, or click to browse</div>
            <div className="upload-zone__sublabel">
              Etherscan, BigQuery, Dune or any raw EVM transfer export
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

      {selectedFile && (
        <div style={{ textAlign: 'center', marginTop: 24 }}>
          <button className="btn btn--primary btn--lg" onClick={handleUpload} disabled={loading}>
            Run Analysis
          </button>
        </div>
      )}

      {error && (
        <div className="error-banner mt-4">
          <AlertTriangle />
          <div>{error}</div>
        </div>
      )}

      <div className="mt-8">
        <div className="label mb-4">Sample datasets</div>
        <div className="samples-grid">
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              className="sample-btn"
              onClick={() => handleSampleLoad(s.id)}
              disabled={loading}
            >
              <div className="sample-btn__header">
                <span className="sample-btn__title">{s.title}</span>
                <span className="badge badge--neutral">{s.tag}</span>
              </div>
              <div className="sample-btn__desc">{s.description}</div>
              <div className="sample-btn__footer">
                <span className="sample-btn__cta">
                  Load <ArrowRight size={12} />
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
