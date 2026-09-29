import React, { useMemo } from 'react';
import { Doughnut, Bar } from 'react-chartjs-2';
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
  Activity,
  AlertTriangle,
  Shield,
  Wallet,
  BarChart3,
  TrendingUp,
  Layers,
  Zap,
} from 'lucide-react';

ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Tooltip, Legend);

// Chart.js default dark theme overrides
ChartJS.defaults.color = '#8b97b0';
ChartJS.defaults.borderColor = 'rgba(100, 120, 160, 0.12)';

const PATTERN_COLORS = {
  'Normal': '#10b981',
  'Smurfing (Structural)': '#f43f5e',
  'Peel/Circular (Temporal)': '#f59e0b',
  'Generic Baseline': '#7c3aed',
};

export default function DashboardOverview({ summary, resultCount }) {
  const statCards = useMemo(() => {
    if (!summary) return [];
    return [
      {
        label: 'Total Transactions',
        value: summary.total_transactions?.toLocaleString() ?? '—',
        color: 'cyan',
        icon: Activity,
        sub: `Analyzed from uploaded dataset`,
      },
      {
        label: 'Flagged (Suspicious)',
        value: summary.flagged_transactions?.toLocaleString() ?? '—',
        color: 'rose',
        icon: AlertTriangle,
        sub: `${summary.alert_rate_percentage ?? 0}% alert rate`,
      },
      {
        label: 'Clean Transactions',
        value: summary.clean_transactions?.toLocaleString() ?? '—',
        color: 'emerald',
        icon: Shield,
        sub: 'Passed below AML threshold',
      },
      {
        label: 'High Risk Wallets',
        value: summary.high_risk_wallets_count?.toLocaleString() ?? '—',
        color: 'amber',
        icon: Wallet,
        sub: `Avg score: ${(summary.average_suspicion_score ?? 0).toFixed(4)}`,
      },
    ];
  }, [summary]);

  const patternData = useMemo(() => {
    if (!summary?.pattern_distribution) return null;
    const labels = Object.keys(summary.pattern_distribution);
    const values = Object.values(summary.pattern_distribution);
    return {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: labels.map((l) => PATTERN_COLORS[l] || '#6366f1'),
          borderWidth: 0,
          hoverOffset: 6,
        },
      ],
    };
  }, [summary]);

  const scoreData = useMemo(() => {
    if (!summary?.score_distribution) return null;
    const labels = Object.keys(summary.score_distribution);
    const values = Object.values(summary.score_distribution);
    const barColors = ['#10b981', '#22d3ee', '#f59e0b', '#f97316', '#f43f5e'];
    return {
      labels,
      datasets: [
        {
          label: 'Transactions',
          data: values,
          backgroundColor: barColors.map((c) => c + '99'),
          borderColor: barColors,
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    };
  }, [summary]);

  if (!summary) return null;

  return (
    <div>
      {/* Stat Cards */}
      <div className="grid-4 mb-6">
        {statCards.map((s) => (
          <div className="card stat-card" key={s.label}>
            <div className="flex items-center justify-between">
              <span className="stat-card__label">{s.label}</span>
              <s.icon size={18} style={{ color: `var(--accent-${s.color})`, opacity: 0.7 }} />
            </div>
            <span className={`stat-card__value stat-card__value--${s.color}`}>{s.value}</span>
            <span className="stat-card__sub">{s.sub}</span>
          </div>
        ))}
      </div>

      {/* Volume Stats Row */}
      <div className="grid-3 mb-6">
        <div className="card stat-card">
          <span className="stat-card__label">Total Volume</span>
          <span className="stat-card__value" style={{ fontSize: '1.3rem' }}>
            <TrendingUp size={18} style={{ color: 'var(--accent-cyan)', marginRight: 6, verticalAlign: 'text-bottom' }} />
            {(summary.total_volume_eth ?? 0).toFixed(4)} ETH
          </span>
        </div>
        <div className="card stat-card">
          <span className="stat-card__label">Flagged Volume</span>
          <span className="stat-card__value stat-card__value--rose" style={{ fontSize: '1.3rem' }}>
            <Zap size={18} style={{ color: 'var(--accent-rose)', marginRight: 6, verticalAlign: 'text-bottom' }} />
            {(summary.flagged_volume_eth ?? 0).toFixed(4)} ETH
          </span>
        </div>
        <div className="card stat-card">
          <span className="stat-card__label">Detection Layers</span>
          <span className="stat-card__value stat-card__value--purple" style={{ fontSize: '1.3rem' }}>
            <Layers size={18} style={{ color: 'var(--accent-purple)', marginRight: 6, verticalAlign: 'text-bottom' }} />
            3 Hybrid
          </span>
          <span className="stat-card__sub">GNN + NTS + XGBoost Baseline</span>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid-2">
        {/* Pattern Distribution Donut */}
        <div className="card">
          <div className="card__header">
            <span className="card__title">
              <BarChart3 /> Pattern Distribution
            </span>
          </div>
          <div className="card__body" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 260 }}>
            {patternData ? (
              <div style={{ width: 260, height: 260 }}>
                <Doughnut
                  data={patternData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: {
                          padding: 14,
                          usePointStyle: true,
                          pointStyleWidth: 10,
                          font: { size: 11, family: "'Inter', sans-serif" },
                        },
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

        {/* Score Distribution Bar */}
        <div className="card">
          <div className="card__header">
            <span className="card__title">
              <Activity /> Risk Score Distribution
            </span>
          </div>
          <div className="card__body" style={{ minHeight: 260 }}>
            {scoreData ? (
              <Bar
                data={scoreData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false },
                  },
                  scales: {
                    x: {
                      grid: { display: false },
                      ticks: { font: { size: 10 } },
                    },
                    y: {
                      grid: { color: 'rgba(100,120,160,0.08)' },
                      ticks: { font: { size: 10 } },
                    },
                  },
                }}
                height={220}
              />
            ) : (
              <span className="text-muted text-sm">No score data</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
