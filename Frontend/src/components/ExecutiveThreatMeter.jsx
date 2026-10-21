import React from 'react';

const LEVELS = {
  NOMINAL: {
    color: 'var(--accent-emerald)',
    badgeClass: 'threat-badge--nominal',
    title: 'Nominal risk',
    desc: 'Alert volume is within normal operating range.',
  },
  ELEVATED: {
    color: 'var(--accent-amber)',
    badgeClass: 'threat-badge--elevated',
    title: 'Elevated risk',
    desc: 'A notable share of transactions match laundering patterns.',
  },
  CRITICAL: {
    color: 'var(--accent-rose)',
    badgeClass: 'threat-badge--critical',
    title: 'Critical risk',
    desc: 'Most activity matches laundering patterns. Review immediately.',
  },
};

// Alert-rate bands: <3% nominal, 3-25% elevated, >25% critical.
function getLevel(rate) {
  if (rate > 25) return 'CRITICAL';
  if (rate >= 3) return 'ELEVATED';
  return 'NOMINAL';
}

// Map alert rate onto the three equal-width gauge segments.
function markerPosition(rate) {
  let pct;
  if (rate < 3) pct = (rate / 3) * (100 / 3);
  else if (rate <= 25) pct = 100 / 3 + ((rate - 3) / 22) * (100 / 3);
  else pct = 200 / 3 + (Math.min(rate - 25, 75) / 75) * (100 / 3);
  return Math.min(Math.max(pct, 1), 99);
}

export default function ExecutiveThreatMeter({ summary }) {
  if (!summary) return null;

  const alertRate = summary.alert_rate_percentage ?? 0;
  const flaggedVolume = summary.flagged_volume_eth ?? 0;
  const totalVolume = summary.total_volume_eth ?? 0;
  const volumeShare = totalVolume > 0 ? (flaggedVolume / totalVolume) * 100 : 0;

  const level = getLevel(alertRate);
  const meta = LEVELS[level];

  return (
    <div className="card threat-meter-card mb-6" style={{ borderLeftColor: meta.color }}>
      <div className="threat-meter__content">
        <div className="threat-meter__badge-row">
          <span className={`threat-badge ${meta.badgeClass}`}>
            <span className="threat-badge__beacon" />
            Risk posture · {level}
          </span>
        </div>

        <h2 className="threat-meter__headline">{meta.title}</h2>
        <p className="threat-meter__desc">{meta.desc}</p>

        <div className="threat-gauge-container">
          <div className="threat-gauge-labels">
            <span className={level === 'NOMINAL' ? 'active-level' : ''}>Nominal &lt; 3%</span>
            <span className={level === 'ELEVATED' ? 'active-level' : ''}>Elevated 3–25%</span>
            <span className={level === 'CRITICAL' ? 'active-level' : ''}>Critical &gt; 25%</span>
          </div>
          <div className="threat-gauge-track">
            <div className="threat-gauge-fill--nominal" />
            <div className="threat-gauge-fill--elevated" />
            <div className="threat-gauge-fill--critical" />
            <div className="threat-gauge-marker" style={{ left: `${markerPosition(alertRate)}%` }} />
          </div>
        </div>

        <div className="threat-chips-grid">
          <div className="threat-chip">
            <span className="threat-chip__label">Alert rate</span>
            <span className="threat-chip__val" style={{ color: meta.color }}>{alertRate}%</span>
          </div>
          <div className="threat-chip">
            <span className="threat-chip__label">Flagged volume</span>
            <span className="threat-chip__val">
              {volumeShare.toFixed(2)}%
              <span className="text-muted text-xs"> · {flaggedVolume.toLocaleString(undefined, { maximumFractionDigits: 2 })} ETH</span>
            </span>
          </div>
          <div className="threat-chip">
            <span className="threat-chip__label">Flagged transactions</span>
            <span className="threat-chip__val">{(summary.flagged_transactions ?? 0).toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
