import React from 'react';
import { Brain, Layers, Zap, TrendingUp, AlertTriangle } from 'lucide-react';

export default function ShapExplainer({ transaction }) {
  if (!transaction) return null;

  const breakdown = transaction.shap_breakdown;
  const isFlagged = transaction.aml_flag === 1;

  if (!breakdown) {
    return (
      <div className="shap-panel card" style={{ padding: 20 }}>
        <p className="text-muted text-sm">No SHAP breakdown available for this transaction.</p>
      </div>
    );
  }

  const topDrivers = breakdown.top_drivers || [];
  const maxAbsImp = Math.max(...topDrivers.map((d) => Math.abs(d.importance)), 0.01);

  // Component scores
  const components = [
    {
      label: 'GNN Structural',
      score: breakdown.gnn_structural_score ?? 0,
      color: 'var(--accent-rose)',
      bg: 'var(--accent-rose-dim)',
      icon: Layers,
    },
    {
      label: 'NTS Temporal',
      score: breakdown.nts_temporal_score ?? 0,
      color: 'var(--accent-amber)',
      bg: 'var(--accent-amber-dim)',
      icon: Zap,
    },
    {
      label: 'Base Features',
      score: breakdown.base_features_score ?? 0,
      color: 'var(--accent-cyan)',
      bg: 'var(--accent-cyan-dim)',
      icon: TrendingUp,
    },
  ];

  return (
    <div className="shap-panel card">
      <div className="card__header">
        <span className="card__title">
          <Brain /> SHAP Explanation
        </span>
        {isFlagged ? (
          <span className="badge badge--danger">
            <AlertTriangle size={12} /> {transaction.pattern_type}
          </span>
        ) : (
          <span className="badge badge--safe">Normal</span>
        )}
      </div>

      <div className="card__body">
        {/* Dominant Layer */}
        <div style={{ marginBottom: 16 }}>
          <span className="text-sm text-muted">Dominant Detection Layer: </span>
          <span className="badge badge--info" style={{ fontSize: '0.78rem' }}>
            {breakdown.dominant_layer}
          </span>
        </div>

        {/* Component Decomposition */}
        <div className="shap-component">
          {components.map((c) => (
            <div
              key={c.label}
              className="shap-component__item"
              style={{ background: c.bg, border: `1px solid ${c.color}22` }}
            >
              <c.icon size={18} style={{ color: c.color, marginBottom: 4 }} />
              <div className="shap-component__label" style={{ color: c.color }}>
                {c.label}
              </div>
              <div className="shap-component__score" style={{ color: c.color }}>
                {c.score >= 0 ? '+' : ''}{c.score.toFixed(4)}
              </div>
            </div>
          ))}
        </div>

        {/* Top Feature Drivers */}
        {topDrivers.length > 0 && (
          <div className="mt-4">
            <div className="text-sm text-muted mb-4" style={{ fontWeight: 600 }}>
              Top Risk Drivers
            </div>
            {topDrivers.map((d, i) => {
              const pct = Math.min(Math.abs(d.importance) / maxAbsImp * 100, 100);
              const isPositive = d.importance >= 0;
              return (
                <div className="shap-bar" key={i}>
                  <span className="shap-bar__label" title={d.feature}>
                    {d.feature}
                  </span>
                  <div className="shap-bar__track">
                    <div
                      className={`shap-bar__fill ${isPositive ? 'shap-bar__fill--positive' : 'shap-bar__fill--negative'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="shap-bar__value">
                    {isPositive ? '+' : ''}{d.importance.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* Narrative Explanation */}
        {breakdown.narrative && (
          <div className={`shap-narrative ${isFlagged ? '' : 'shap-narrative--clean'}`}>
            <strong style={{ display: 'block', marginBottom: 4, fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Investigator Narrative
            </strong>
            {breakdown.narrative}
          </div>
        )}

        {/* Feature Values */}
        {topDrivers.length > 0 && (
          <div className="mt-4">
            <details style={{ cursor: 'pointer' }}>
              <summary className="text-sm text-muted" style={{ fontWeight: 500 }}>
                Raw feature values
              </summary>
              <div style={{ marginTop: 8, overflowX: 'auto' }}>
                <table className="data-table" style={{ fontSize: '0.76rem' }}>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Value</th>
                      <th>SHAP Importance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topDrivers.map((d, i) => (
                      <tr key={i}>
                        <td className="text-mono">{d.feature}</td>
                        <td className="text-mono">{d.feature_value?.toFixed(4) ?? '—'}</td>
                        <td className="text-mono" style={{ color: d.importance >= 0 ? 'var(--accent-rose)' : 'var(--accent-cyan)' }}>
                          {d.importance >= 0 ? '+' : ''}{d.importance.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
