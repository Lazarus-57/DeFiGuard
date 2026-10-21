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

export default function DashboardOverview({ summary, normalization, onDrillDown }) {
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
