import React, { useEffect, useState } from 'react';
import { getModelInfo } from '../api';
import { Cpu, Target, Gauge, BarChart3, Layers, AlertTriangle, Loader } from 'lucide-react';

export default function ModelInfoPanel() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await getModelInfo();
        if (!cancelled) setInfo(data);
      } catch (err) {
        if (!cancelled) setError(err.response?.data?.detail || err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="loading-overlay" style={{ padding: 60 }}>
        <div className="spinner" />
        <p className="loading-overlay__text">Loading model information...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-banner">
        <AlertTriangle />
        Failed to load model info: {error}. Make sure FastAPI is running at localhost:8000.
      </div>
    );
  }

  if (!info) return null;

  const metrics = info.gold_standard_metrics || {};

  const metricCards = [
    { label: 'ROC-AUC', value: metrics.roc_auc?.toFixed(4) ?? '—', color: 'var(--accent-cyan)' },
    { label: 'Recall', value: `${((metrics.recall ?? 0) * 100).toFixed(1)}%`, color: 'var(--accent-emerald)' },
    { label: 'Precision', value: `${((metrics.precision ?? 0) * 100).toFixed(1)}%`, color: 'var(--accent-amber)' },
    { label: 'F1 Score', value: metrics.f1_score?.toFixed(4) ?? '—', color: 'var(--accent-purple)' },
    { label: 'PR-AUC', value: metrics.pr_auc?.toFixed(4) ?? '—', color: 'var(--text-secondary)' },
    { label: 'Threshold', value: metrics.decision_threshold?.toFixed(4) ?? '—', color: 'var(--accent-rose)' },
  ];

  return (
    <div>
      {/* Architecture Header */}
      <div className="card mb-6">
        <div className="card__header">
          <span className="card__title">
            <Cpu /> Model Architecture
          </span>
          <span className="badge badge--info">{info.feature_count ?? 51} Features</span>
        </div>
        <div className="card__body">
          <p className="text-sm" style={{ color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 12 }}>
            <strong style={{ color: 'var(--text-primary)' }}>{info.model_type}</strong>
          </p>
          <p className="text-sm text-muted" style={{ lineHeight: 1.6 }}>
            The hybrid model fuses three detection layers into a single 51-feature XGBoost classifier.
            Graph Neural Networks (GraphSAGE) capture wallet structural neighborhoods for smurfing detection.
            Normalised Transaction Scores (NTS) capture temporal burstiness for peel chain and circular ring detection.
            Baseline features (centrality, degree, flow ratio) provide volumetric and topological signals.
          </p>

          {/* ASCII Architecture */}
          <pre style={{
            marginTop: 16,
            padding: 16,
            background: 'rgba(6, 10, 20, 0.60)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.72rem',
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)',
            overflowX: 'auto',
            lineHeight: 1.5,
          }}>
{`Raw Transactions
       │
       ├── Baseline Features (15)  ─────────────────────────────────┐
       │   amount, centrality, degree, flow_ratio...                │
       │                                                            ▼
       ├── GNN Structural Embeddings (32) ── GraphSAGE ──► 51-Feature ──► XGBoost ──► Suspicion Score
       │   wallet graph position, neighbourhood...                  ▲       │              │
       │                                                            │       │         SHAP Explainer
       └── NTS Temporal Features (4) ───────────────────────────────┘       │              │
           from_nts, to_nts, nts_max, nts_mean                             │              ▼
                                                                            └──► Pattern Type Label`}
          </pre>
        </div>
      </div>

      {/* Gold Standard Metrics */}
      <div className="card mb-6">
        <div className="card__header">
          <span className="card__title">
            <Target /> Gold Standard Test Metrics (100k Dataset)
          </span>
        </div>
        <div className="card__body">
          <div className="metric-grid">
            {metricCards.map((m) => (
              <div className="metric-item" key={m.label}>
                <div className="metric-item__label">{m.label}</div>
                <div className="metric-item__value" style={{ color: m.color }}>{m.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Topology Coverage */}
      <div className="card mb-6">
        <div className="card__header">
          <span className="card__title">
            <Layers /> Laundering Topology Coverage
          </span>
        </div>
        <div className="card__body">
          <table className="topology-table">
            <thead>
              <tr>
                <th>Pattern</th>
                <th>Topology</th>
                <th>Primary Detector</th>
              </tr>
            </thead>
            <tbody>
              {(info.topology_coverage || []).map((t, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{t.pattern}</td>
                  <td>{t.topology}</td>
                  <td>
                    <span className="badge badge--info" style={{ fontSize: '0.72rem' }}>
                      {t.primary_detector}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature List */}
      <div className="card">
        <div className="card__header">
          <span className="card__title">
            <BarChart3 /> Feature Set ({info.feature_count ?? 0} features)
          </span>
        </div>
        <div className="card__body">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(info.features || []).map((f, i) => {
              let badgeCls = 'badge--info';
              if (f.startsWith('gnn_')) badgeCls = 'badge--danger';
              else if (f.startsWith('nts_') || f.endsWith('_nts')) badgeCls = 'badge--warn';
              return (
                <span key={i} className={`badge ${badgeCls}`} style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                  {f}
                </span>
              );
            })}
          </div>
          <div className="graph-legend mt-4">
            <div className="graph-legend__item">
              <div className="graph-legend__dot" style={{ background: 'var(--accent-cyan)' }} />
              Baseline
            </div>
            <div className="graph-legend__item">
              <div className="graph-legend__dot" style={{ background: 'var(--accent-rose)' }} />
              GNN Embeddings
            </div>
            <div className="graph-legend__item">
              <div className="graph-legend__dot" style={{ background: 'var(--accent-amber)' }} />
              NTS Temporal
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
