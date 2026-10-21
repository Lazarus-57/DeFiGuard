import React, { useEffect, useState, useMemo } from 'react';
import { getModelInfo } from '../api';
import {
  Cpu,
  Target,
  BarChart3,
  Layers,
  AlertTriangle,
  ArrowRight,
  ShieldCheck,
  Zap,
} from 'lucide-react';

export default function ModelInfoPanel() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeCategory, setActiveCategory] = useState('all');

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

  const features = useMemo(() => {
    if (!info?.features) return [];
    return info.features.map((name) => {
      let category = 'baseline';
      if (name.startsWith('gnn_')) category = 'gnn';
      else if (name.startsWith('nts_') || name.endsWith('_nts')) category = 'nts';
      return { name, category };
    });
  }, [info]);

  const filteredFeatures = useMemo(() => {
    if (activeCategory === 'all') return features;
    return features.filter((f) => f.category === activeCategory);
  }, [features, activeCategory]);

  const counts = useMemo(() => {
    return {
      all: features.length,
      gnn: features.filter((f) => f.category === 'gnn').length,
      baseline: features.filter((f) => f.category === 'baseline').length,
      nts: features.filter((f) => f.category === 'nts').length,
    };
  }, [features]);

  if (loading) {
    return (
      <div className="loading-overlay" style={{ padding: 60 }}>
        <div className="spinner" />
        <p className="loading-overlay__text">Loading model telemetry...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-banner">
        <AlertTriangle />
        Failed to load model specifications: {error}.
      </div>
    );
  }

  if (!info) return null;

  const m = info.gold_standard_metrics || {};

  // Formatted metric values with fallbacks to hardened test split
  const recallVal = ((m.recall ?? 0.9856) * 100).toFixed(1);
  const precisionVal = ((m.precision ?? 0.2671) * 100).toFixed(1);
  const rocAucVal = (m.roc_auc ?? 0.8681).toFixed(4);
  const prAucVal = (m.pr_auc ?? 0.2841).toFixed(4);
  const p01Val = ((m.precision_at_01 ?? 0.75) * 100).toFixed(1);
  const p05Val = ((m.precision_at_05 ?? 0.6951) * 100).toFixed(1);
  const p10Val = ((m.precision_at_10 ?? 0.3879) * 100).toFixed(1);
  const thresholdVal = (m.decision_threshold ?? 0.2000).toFixed(4);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* ── 1. Latest Detection Benchmarks ── */}
      <div className="card">
        <div className="card__header">
          <span className="card__title">
            <Target /> Evaluation Benchmarks
          </span>
        </div>

        <div className="card__body">
          {/* Row 1: Primary Detection Performance */}
          <div className="grid-4 mb-4">
            <div className="stat-card" style={{ padding: '14px 16px', background: 'rgba(6,10,18,0.5)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-card__label">Test Recall</div>
              <div className="stat-card__value" style={{ color: 'var(--accent-emerald)', fontSize: '1.75rem' }}>
                {recallVal}%
              </div>
              <div className="stat-card__sub">Illicit detection sensitivity</div>
            </div>

            <div className="stat-card" style={{ padding: '14px 16px', background: 'rgba(6,10,18,0.5)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-card__label">Overall Precision</div>
              <div className="stat-card__value" style={{ color: 'var(--accent-amber)', fontSize: '1.75rem' }}>
                {precisionVal}%
              </div>
              <div className="stat-card__sub">Decision threshold {thresholdVal}</div>
            </div>

            <div className="stat-card" style={{ padding: '14px 16px', background: 'rgba(6,10,18,0.5)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-card__label">ROC-AUC</div>
              <div className="stat-card__value" style={{ color: 'var(--accent-cyan)', fontSize: '1.75rem' }}>
                {rocAucVal}
              </div>
              <div className="stat-card__sub">Area under ROC curve</div>
            </div>

            <div className="stat-card" style={{ padding: '14px 16px', background: 'rgba(6,10,18,0.5)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-card__label">PR-AUC</div>
              <div className="stat-card__value" style={{ color: 'var(--accent-purple)', fontSize: '1.75rem' }}>
                {prAucVal}
              </div>
              <div className="stat-card__sub">Area under Precision-Recall curve</div>
            </div>
          </div>

          {/* Row 2: Precision across @ K (Risk Based Triage) */}
          <div style={{ padding: '14px 16px', background: 'rgba(10, 13, 19, 0.4)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-muted text-uppercase tracking-wider">
                Precision @ K (Risk Based Triage)
              </span>
            </div>

            <div className="grid-3" style={{ gap: 12 }}>
              <div style={{ padding: '10px 14px', background: 'rgba(6,10,18,0.6)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                <div className="stat-card__label">Precision @ Top 0.1%</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', marginTop: 2 }}>
                  {p01Val}%
                </div>
                <div className="text-xs text-muted mt-1">Top 0.1% ranked transactions</div>
              </div>

              <div style={{ padding: '10px 14px', background: 'rgba(6,10,18,0.6)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                <div className="stat-card__label">Precision @ Top 0.5%</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', marginTop: 2 }}>
                  {p05Val}%
                </div>
                <div className="text-xs text-muted mt-1">Top 0.5% ranked transactions</div>
              </div>

              <div style={{ padding: '10px 14px', background: 'rgba(6,10,18,0.6)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                <div className="stat-card__label">Precision @ Top 1.0%</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', marginTop: 2 }}>
                  {p10Val}%
                </div>
                <div className="text-xs text-muted mt-1">Top 1.0% ranked transactions</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── 2. Minimalist Architecture Pipeline ── */}
      <div className="card">
        <div className="card__header">
          <span className="card__title">
            <Cpu /> Architecture &amp; Dataflow
          </span>
          <span className="badge badge--info">{info.feature_count ?? 51} Features</span>
        </div>

        <div className="card__body">
          <div className="arch-pipeline">
            {/* Box 1: Inputs */}
            <div className="arch-card">
              <div className="arch-card__title">
                <Layers size={13} style={{ color: 'var(--accent-cyan)' }} />
                1. Feature Signals (51)
              </div>
              <div className="arch-card__item">
                <span>Baseline &amp; Topology</span>
                <span className="badge badge--cyan" style={{ fontSize: '0.62rem' }}>15</span>
              </div>
              <div className="arch-card__item">
                <span>GraphSAGE 2-Hop GNN</span>
                <span className="badge badge--rose" style={{ fontSize: '0.62rem' }}>32</span>
              </div>
              <div className="arch-card__item">
                <span>NTS Temporal Cadence</span>
                <span className="badge badge--amber" style={{ fontSize: '0.62rem' }}>4</span>
              </div>
            </div>

            <ArrowRight className="arch-pipeline__arrow" />

            {/* Box 2: Classifier */}
            <div className="arch-card">
              <div className="arch-card__title">
                <Cpu size={13} style={{ color: 'var(--accent-emerald)' }} />
                2. Decision Engine
              </div>
              <div className="arch-card__item">
                <span>Classifier</span>
                <span className="font-mono text-primary text-xs">XGBoost (Hist)</span>
              </div>
              <div className="arch-card__item">
                <span>Decision Threshold</span>
                <span className="font-mono text-xs" style={{ color: 'var(--accent-rose)' }}>{thresholdVal}</span>
              </div>
              <div className="arch-card__item">
                <span>Calibration</span>
                <span className="font-mono text-xs text-emerald">Sigmoid Calibrated</span>
              </div>
            </div>

            <ArrowRight className="arch-pipeline__arrow" />

            {/* Box 3: Outputs */}
            <div className="arch-card">
              <div className="arch-card__title">
                <ShieldCheck size={13} style={{ color: 'var(--accent-purple)' }} />
                3. Forensic Outputs
              </div>
              <div className="arch-card__item">
                <span>Suspicion Score</span>
                <span className="font-mono text-primary text-xs">[0.00 – 1.00]</span>
              </div>
              <div className="arch-card__item">
                <span>Attribution Explainer</span>
                <span className="font-mono text-xs text-purple">TreeSHAP</span>
              </div>
              <div className="arch-card__item">
                <span>Topology Label</span>
                <span className="font-mono text-xs text-secondary">3 Heuristics</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── 3. Laundering Topology Coverage ── */}
      <div className="card">
        <div className="card__header">
          <span className="card__title">
            <Layers /> Laundering Topology Defense
          </span>
        </div>
        <div className="card__body" style={{ padding: 0 }}>
          <table className="topology-table">
            <thead>
              <tr>
                <th>Pattern</th>
                <th>Topology Structure</th>
                <th>Primary Detection Layer</th>
              </tr>
            </thead>
            <tbody>
              {(info.topology_coverage || []).map((t, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{t.pattern}</td>
                  <td className="font-mono text-xs">{t.topology}</td>
                  <td>
                    <span className="badge badge--info" style={{ fontSize: '0.68rem' }}>
                      {t.primary_detector}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── 4. Feature Set (51 Features) ── */}
      <div className="card">
        <div className="card__header">
          <span className="card__title">
            <BarChart3 /> Feature Set ({info.feature_count ?? 51})
          </span>
          <div className="flex gap-2">
            <button
              className={`feature-filter-tab ${activeCategory === 'all' ? 'feature-filter-tab--active' : ''}`}
              onClick={() => setActiveCategory('all')}
            >
              All ({counts.all})
            </button>
            <button
              className={`feature-filter-tab ${activeCategory === 'gnn' ? 'feature-filter-tab--active' : ''}`}
              onClick={() => setActiveCategory('gnn')}
            >
              GNN ({counts.gnn})
            </button>
            <button
              className={`feature-filter-tab ${activeCategory === 'baseline' ? 'feature-filter-tab--active' : ''}`}
              onClick={() => setActiveCategory('baseline')}
            >
              Baseline ({counts.baseline})
            </button>
            <button
              className={`feature-filter-tab ${activeCategory === 'nts' ? 'feature-filter-tab--active' : ''}`}
              onClick={() => setActiveCategory('nts')}
            >
              NTS ({counts.nts})
            </button>
          </div>
        </div>

        <div className="card__body">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {filteredFeatures.map((f) => {
              let chipCls = 'feature-tag-chip--baseline';
              if (f.category === 'gnn') chipCls = 'feature-tag-chip--gnn';
              else if (f.category === 'nts') chipCls = 'feature-tag-chip--nts';

              return (
                <span key={f.name} className={`feature-tag-chip ${chipCls}`}>
                  {f.name}
                </span>
              );
            })}
          </div>

          <div className="flex items-center gap-4 mt-4" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            <div className="flex items-center gap-1.5">
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-cyan)' }} />
              <span>Baseline Centrality (15)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-rose)' }} />
              <span>GraphSAGE Embeddings (32)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-amber)' }} />
              <span>NTS Temporal (4)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
