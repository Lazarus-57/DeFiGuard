import React, { useMemo, useRef } from 'react';
import { Doughnut, Bar, getElementAtEvent } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js';
import {
  UserCheck,
  ShieldAlert,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
} from 'lucide-react';
import ExecutiveThreatMeter from './ExecutiveThreatMeter';

ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Tooltip, Legend);

ChartJS.defaults.color = '#7d8696';
ChartJS.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';
ChartJS.defaults.font.family = "'Inter', sans-serif";

const PATTERN_COLORS = {
  'Normal': '#34d399',
  'Smurfing (Structural)': '#ff5468',
  'Peel/Circular (Temporal)': '#f5a524',
  'Generic Baseline': '#9b8cff',
};

const SCORE_COLORS = ['#34d399', '#5fd0ff', '#f5a524', '#fb7a3c', '#ff5468'];

const fmt = (n, d = 2) =>
  (n ?? 0).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

export default function DashboardOverview({
  summary,
  results = [],
  normalization,
  reviews = {},
  onDrillDown,
}) {
  const donutRef = useRef(null);
  const barRef = useRef(null);

  const statCards = useMemo(() => {
    if (!summary) return [];
    const total = summary.total_transactions || 1;
    return [
      {
        label: 'Transactions',
        value: summary.total_transactions?.toLocaleString() ?? '—',
        color: 'cyan',
        sub: 'Analyzed',
      },
      {
        label: 'Flagged',
        value: summary.flagged_transactions?.toLocaleString() ?? '—',
        color: 'rose',
        sub: `${summary.alert_rate_percentage ?? 0}% alert rate`,
      },
      {
        label: 'Clean',
        value: summary.clean_transactions?.toLocaleString() ?? '—',
        color: 'emerald',
        sub: `${(((summary.clean_transactions ?? 0) / total) * 100).toFixed(1)}% of total`,
      },
      {
        label: 'High-risk wallets',
        value: summary.high_risk_wallets_count?.toLocaleString() ?? '—',
        color: 'amber',
        sub: `Avg score ${(summary.average_suspicion_score ?? 0).toFixed(4)}`,
      },
    ];
  }, [summary]);

  const patternData = useMemo(() => {
    if (!summary?.pattern_distribution) return null;
    const labels = Object.keys(summary.pattern_distribution);
    return {
      labels,
      datasets: [{
        data: Object.values(summary.pattern_distribution),
        backgroundColor: labels.map((l) => PATTERN_COLORS[l] || '#6b7280'),
        borderWidth: 3,
        borderColor: '#0a0d13',
        hoverOffset: 4,
      }],
    };
  }, [summary]);

  const scoreData = useMemo(() => {
    if (!summary?.score_distribution) return null;
    return {
      labels: Object.keys(summary.score_distribution),
      datasets: [{
        label: 'Transactions',
        data: Object.values(summary.score_distribution),
        backgroundColor: SCORE_COLORS.map((c) => c + '99'),
        hoverBackgroundColor: SCORE_COLORS,
        borderRadius: 2,
        borderSkipped: false,
      }],
    };
  }, [summary]);

  const handleDonutClick = (event) => {
    if (!donutRef.current || !onDrillDown) return;
    const els = getElementAtEvent(donutRef.current, event);
    if (els?.length) onDrillDown({ type: 'pattern', value: patternData.labels[els[0].index] });
  };

  const handleBarClick = (event) => {
    if (!barRef.current || !onDrillDown) return;
    const els = getElementAtEvent(barRef.current, event);
    if (els?.length) onDrillDown({ type: 'score', value: scoreData.labels[els[0].index] });
  };

  const triageStats = useMemo(() => {
    if (!results || results.length === 0) return null;
    let needsReview = 0;
    let critical = 0;
    let cleared = 0;
    let escalated = 0;

    results.forEach((r) => {
      const rev = reviews[r.tx_hash];
      if (rev?.status === 'cleared') {
        cleared++;
      } else if (rev?.status === 'escalated') {
        escalated++;
      } else if (r.aml_flag === 1) {
        if ((r.suspicion_score ?? 0) < 0.45) {
          needsReview++;
        } else {
          critical++;
        }
      }
    });

    const totalReviewed = cleared + escalated;
    const totalFlagged = summary?.flagged_transactions || 1;
    const progressPct = Math.round((totalReviewed / totalFlagged) * 100);

    return {
      needsReview,
      critical,
      cleared,
      escalated,
      totalReviewed,
      progressPct: Math.min(100, progressPct),
    };
  }, [results, reviews, summary]);

  if (!summary) return null;

  const normItems = normalization?.detected_source
    ? [
        ['Source', normalization.detected_source],
        ['Units', normalization.unit_conversion],
        [
          'Graph',
          normalization.graph_features_generated
            ? `Built on the fly · ${normalization.wallet_count?.toLocaleString()} wallets`
            : 'Pre-computed',
        ],
        [
          'Columns mapped',
          normalization.columns_mapped ? Object.keys(normalization.columns_mapped).length : 0,
        ],
      ]
    : [];

  return (
    <div>
      <ExecutiveThreatMeter summary={summary} />

      {/* Analyst Triage Strip */}
      {triageStats && (
        <div className="card dashboard-triage-strip mb-6">
          <div className="flex items-center gap-3">
            <div
              className="p-2 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(95, 208, 255, 0.1)', color: '#5fd0ff', width: 36, height: 36 }}
            >
              <UserCheck size={18} />
            </div>
            <div>
              <div className="text-sm font-semibold flex items-center gap-2">
                <span>Analyst Adjudication & Triage</span>
                {triageStats.totalReviewed > 0 && (
                  <span className="badge badge--safe" style={{ fontSize: '0.68rem', padding: '2px 8px' }}>
                    {triageStats.totalReviewed} Decided ({triageStats.progressPct}%)
                  </span>
                )}
              </div>
              <div className="text-xs text-muted">
                Human-in-the-loop review queue for ambiguous borderline transactions (score 0.20 – 0.45)
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <button
              className="triage-strip-stat triage-strip-stat--amber"
              onClick={() => onDrillDown && onDrillDown({ type: 'triage', value: 'needs_review' })}
              title="Filter transactions needing analyst review"
            >
              <AlertTriangle size={13} />
              <span>Needs Review:</span>
              <strong>{triageStats.needsReview.toLocaleString()}</strong>
            </button>

            <button
              className="triage-strip-stat triage-strip-stat--rose"
              onClick={() => onDrillDown && onDrillDown({ type: 'triage', value: 'critical' })}
              title="Filter high-confidence critical alerts"
            >
              <ShieldAlert size={13} />
              <span>Critical:</span>
              <strong>{triageStats.critical.toLocaleString()}</strong>
            </button>

            <button
              className="triage-strip-stat triage-strip-stat--emerald"
              onClick={() => onDrillDown && onDrillDown({ type: 'triage', value: 'cleared' })}
              title="Filter analyst-cleared false positives"
            >
              <CheckCircle2 size={13} />
              <span>Cleared:</span>
              <strong>{triageStats.cleared.toLocaleString()}</strong>
            </button>

            <button
              className="triage-strip-stat triage-strip-stat--purple"
              onClick={() => onDrillDown && onDrillDown({ type: 'triage', value: 'escalated' })}
              title="Filter analyst-escalated SAR alerts"
            >
              <ShieldAlert size={13} />
              <span>Escalated:</span>
              <strong>{triageStats.escalated.toLocaleString()}</strong>
            </button>

            <button
              className="btn btn--secondary btn--xs flex items-center gap-1"
              onClick={() => onDrillDown && onDrillDown({ type: 'triage', value: 'needs_review' })}
              style={{ marginLeft: 6 }}
            >
              <span>Queue</span>
              <ArrowRight size={12} />
            </button>
          </div>
        </div>
      )}

      {/* Key figures */}
      <div className="grid-4 mb-6">
        {statCards.map((s) => (
          <div className="card stat-card stat-card--interactive" key={s.label}>
            <span className="stat-card__label">{s.label}</span>
            <span className={`stat-card__value stat-card__value--${s.color}`}>{s.value}</span>
            <span className="stat-card__sub">{s.sub}</span>
          </div>
        ))}
      </div>

      <div className="grid-3 mb-6">
        <div className="card stat-card">
          <span className="stat-card__label">Total volume</span>
          <span className="stat-card__value" style={{ fontSize: '1.4rem' }}>{fmt(summary.total_volume_eth)} ETH</span>
        </div>
        <div className="card stat-card">
          <span className="stat-card__label">Flagged volume</span>
          <span className="stat-card__value stat-card__value--rose" style={{ fontSize: '1.4rem' }}>{fmt(summary.flagged_volume_eth)} ETH</span>
        </div>
        <div className="card stat-card">
          <span className="stat-card__label">Detection</span>
          <span className="stat-card__value" style={{ fontSize: '1.4rem' }}>GNN · NTS · XGBoost</span>
        </div>
      </div>

      {/* Charts */}
      <div className="grid-2 mb-6">
        <div className="card">
          <div className="card__header">
            <span className="card__title">Pattern distribution</span>
            <span className="drilldown-hint">Click to filter</span>
          </div>
          <div className="card__body" style={{ display: 'flex', justifyContent: 'center', minHeight: 300 }}>
            {patternData ? (
              <div style={{ width: 280, height: 280, cursor: 'pointer' }}>
                <Doughnut
                  ref={donutRef}
                  data={patternData}
                  onClick={handleDonutClick}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '72%',
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { padding: 16, usePointStyle: true, pointStyle: 'rectRounded', font: { size: 11 } },
                      },
                    },
                  }}
                />
              </div>
            ) : (
              <span className="text-muted text-sm">No pattern data</span>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card__header">
            <span className="card__title">Risk score distribution</span>
            <span className="drilldown-hint">Click to filter</span>
          </div>
          <div className="card__body" style={{ height: 346, cursor: 'pointer' }}>
            {scoreData ? (
              <Bar
                ref={barRef}
                data={scoreData}
                onClick={handleBarClick}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } },
                  scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                    y: { ticks: { font: { size: 10 } } },
                  },
                }}
              />
            ) : (
              <span className="text-muted text-sm">No score data</span>
            )}
          </div>
        </div>
      </div>

      {/* Ingestion summary */}
      {normItems.length > 0 && (
        <div className="card">
          <div className="card__header">
            <span className="card__title">Ingestion</span>
          </div>
          <div className="card__body">
            <div className="grid-4">
              {normItems.map(([label, value]) => (
                <div className="intelligence-chip" key={label}>
                  <div className="intelligence-chip__label">{label}</div>
                  <div className="intelligence-chip__val">{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
