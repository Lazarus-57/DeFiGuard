import React, { useState, useMemo, useEffect } from 'react';
import {
  Table2,
  Filter,
  Download,
  ChevronDown,
  ChevronUp,
  Eye,
  Search,
  Copy,
  Check,
  ExternalLink,
  X,
  Sparkles,
  Layers,
  UserCheck,
  ShieldAlert,
  AlertTriangle,
  FileCheck,
} from 'lucide-react';
import ShapExplainer from './ShapExplainer';
import AnalystReviewPanel from './AnalystReviewPanel';

const PAGE_SIZE = 25;

function truncateAddr(addr) {
  if (!addr || addr.length <= 14) return addr || '—';
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`;
}

function ScoreBar({ score }) {
  const pct = Math.round(score * 100);
  const cls = score >= 0.7 ? 'high' : score >= 0.35 ? 'mid' : 'low';
  return (
    <div className="score-bar">
      <div className="score-bar__track">
        <div className={`score-bar__fill score-bar__fill--${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-mono" style={{ fontSize: '0.78rem', minWidth: 44, textAlign: 'right', fontWeight: 600 }}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}

function PatternBadge({ pattern }) {
  if (!pattern || pattern === 'Normal') return <span className="badge badge--safe">Normal</span>;
  if (pattern.includes('Smurfing')) return <span className="badge badge--danger">Smurfing</span>;
  if (pattern.includes('Peel') || pattern.includes('Circular')) return <span className="badge badge--warn">Peel / Circular</span>;
  return <span className="badge badge--purple">{pattern}</span>;
}

export default function AlertsTable({
  results = [],
  activeDrillDown,
  onClearDrillDown,
  reviews = {},
  onSaveReview = () => {},
  onClearReview = () => {},
}) {
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState('suspicion_score');
  const [sortAsc, setSortAsc] = useState(false);
  const [filterFlagged, setFilterFlagged] = useState(false);
  const [filterPattern, setFilterPattern] = useState('all');
  const [filterTriage, setFilterTriage] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedKey, setCopiedKey] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(null);

  // Sync active drill-down from parent (if user clicked chart or triage badge in Dashboard)
  useEffect(() => {
    if (activeDrillDown) {
      if (activeDrillDown.type === 'pattern') {
        setFilterPattern(activeDrillDown.value);
      } else if (activeDrillDown.type === 'triage') {
        setFilterTriage(activeDrillDown.value);
      }
      setPage(0);
    }
  }, [activeDrillDown]);

  // Unique pattern types
  const patternTypes = useMemo(() => {
    const s = new Set(results.map((r) => r.pattern_type).filter(Boolean));
    return Array.from(s);
  }, [results]);

  // Triage queue counts
  const triageCounts = useMemo(() => {
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

    return { needsReview, critical, cleared, escalated };
  }, [results, reviews]);

  // Filter & sort
  const filtered = useMemo(() => {
    let data = [...results];

    // Drilldown score bracket handling
    if (activeDrillDown && activeDrillDown.type === 'score') {
      const parts = activeDrillDown.value.split('-').map((s) => parseFloat(s.trim()));
      if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
        data = data.filter((r) => {
          const score = r.suspicion_score ?? 0;
          return score >= parts[0] && score <= parts[1];
        });
      }
    }

    if (filterFlagged) data = data.filter((r) => r.aml_flag === 1);
    if (filterPattern !== 'all') data = data.filter((r) => r.pattern_type === filterPattern);

    // Triage Queue filtering
    if (filterTriage === 'needs_review') {
      data = data.filter(
        (r) => r.aml_flag === 1 && (r.suspicion_score ?? 0) < 0.45 && !reviews[r.tx_hash]
      );
    } else if (filterTriage === 'critical') {
      data = data.filter(
        (r) => r.aml_flag === 1 && (r.suspicion_score ?? 0) >= 0.45 && !reviews[r.tx_hash]
      );
    } else if (filterTriage === 'cleared') {
      data = data.filter((r) => reviews[r.tx_hash]?.status === 'cleared');
    } else if (filterTriage === 'escalated') {
      data = data.filter((r) => reviews[r.tx_hash]?.status === 'escalated');
    }

    // Search query matching (tx_hash, from, to)
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      data = data.filter(
        (r) =>
          (r.tx_hash && r.tx_hash.toLowerCase().includes(q)) ||
          (r.from && r.from.toLowerCase().includes(q)) ||
          (r.to && r.to.toLowerCase().includes(q))
      );
    }

    data.sort((a, b) => {
      const va = a[sortKey] ?? 0;
      const vb = b[sortKey] ?? 0;
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? va - vb : vb - va;
    });
    return data;
  }, [results, filterFlagged, filterPattern, filterTriage, searchQuery, sortKey, sortAsc, activeDrillDown, reviews]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paged = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
    setPage(0);
  };

  const SortArrow = ({ col }) => {
    if (sortKey !== col) return <span className="sort-arrow">⇅</span>;
    return <span className="sort-arrow active-sort">{sortAsc ? '↑' : '↓'}</span>;
  };

  const handleCopy = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1800);
  };

  const handleExport = () => {
    const headers = ['tx_hash', 'from', 'to', 'amount', 'suspicion_score', 'aml_flag', 'pattern_type', 'top_shap_reason'];
    const csv = [
      headers.join(','),
      ...filtered.map((r) =>
        headers.map((h) => JSON.stringify(r[h] ?? '')).join(',')
      ),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'defiguard_forensic_alerts.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportAuditTrail = () => {
    const reviewedRows = results
      .filter((r) => reviews[r.tx_hash])
      .map((r) => ({
        tx_hash: r.tx_hash,
        from_address: r.from,
        to_address: r.to,
        amount_eth: r.amount,
        block_time: r.block_time,
        model_risk_score: r.suspicion_score,
        model_pattern: r.pattern_type,
        model_top_driver: r.top_shap_reason,
        dominant_layer: r.shap_breakdown?.dominant_layer || 'N/A',
        analyst_decision: reviews[r.tx_hash]?.status,
        compliance_reason: reviews[r.tx_hash]?.reason,
        case_notes: reviews[r.tx_hash]?.note,
        reviewed_at: reviews[r.tx_hash]?.reviewedAt,
        analyst: reviews[r.tx_hash]?.analyst,
      }));

    if (reviewedRows.length === 0) {
      alert('No analyst determinations recorded yet. Click on any flagged transaction row below to clear or escalate it first.');
      return;
    }

    const payload = {
      report: 'DeFIGuard Human-in-the-Loop AML Triage Audit Trail',
      exported_at: new Date().toISOString(),
      total_determinations: reviewedRows.length,
      determinations: reviewedRows,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `defiguard_triage_audit_trail_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const selectedTransaction = selectedIdx !== null ? filtered[selectedIdx] : null;

  return (
    <div>
      {/* ── Drill-down Notification Banner (if triggered from dashboard) ── */}
      {activeDrillDown && (
        <div className="drilldown-active-banner mb-4">
          <div className="flex items-center gap-2">
            <span className="label">Filtered by</span>
            <span className="text-sm font-semibold text-primary">{activeDrillDown.value}</span>
          </div>
          <button
            className="btn btn--secondary btn--xs"
            onClick={() => {
              setFilterPattern('all');
              setFilterTriage('all');
              if (onClearDrillDown) onClearDrillDown();
            }}
          >
            <X size={12} /> Clear
          </button>
        </div>
      )}

      {/* ── Triage Queue Fast Selector Bar ── */}
      <div className="flex items-center gap-2 flex-wrap mb-3">
        <span className="label" style={{ marginRight: 6 }}>Triage Queue:</span>
        <button
          type="button"
          className={`triage-filter-pill ${filterTriage === 'all' ? 'triage-filter-pill--active' : ''}`}
          onClick={() => { setFilterTriage('all'); setPage(0); }}
        >
          <span>All ({results.length.toLocaleString()})</span>
        </button>

        <button
          type="button"
          className={`triage-filter-pill ${filterTriage === 'needs_review' ? 'triage-filter-pill--active' : ''}`}
          onClick={() => { setFilterTriage('needs_review'); setPage(0); }}
          style={{ borderColor: filterTriage === 'needs_review' ? 'var(--accent-amber)' : undefined }}
        >
          <AlertTriangle size={11} style={{ color: 'var(--accent-amber)' }} />
          <span>Needs Review ({triageCounts.needsReview.toLocaleString()})</span>
        </button>

        <button
          type="button"
          className={`triage-filter-pill ${filterTriage === 'critical' ? 'triage-filter-pill--active' : ''}`}
          onClick={() => { setFilterTriage('critical'); setPage(0); }}
          style={{ borderColor: filterTriage === 'critical' ? 'var(--accent-rose)' : undefined }}
        >
          <ShieldAlert size={11} style={{ color: 'var(--accent-rose)' }} />
          <span>Critical Threat ({triageCounts.critical.toLocaleString()})</span>
        </button>

        <button
          type="button"
          className={`triage-filter-pill ${filterTriage === 'cleared' ? 'triage-filter-pill--active' : ''}`}
          onClick={() => { setFilterTriage('cleared'); setPage(0); }}
          style={{ borderColor: filterTriage === 'cleared' ? 'var(--accent-emerald)' : undefined }}
        >
          <Check size={11} style={{ color: 'var(--accent-emerald)' }} />
          <span>Cleared by Analyst ({triageCounts.cleared.toLocaleString()})</span>
        </button>

        <button
          type="button"
          className={`triage-filter-pill ${filterTriage === 'escalated' ? 'triage-filter-pill--active' : ''}`}
          onClick={() => { setFilterTriage('escalated'); setPage(0); }}
          style={{ borderColor: filterTriage === 'escalated' ? 'var(--accent-rose)' : undefined }}
        >
          <FileCheck size={11} style={{ color: 'var(--accent-rose)' }} />
          <span>Escalated ({triageCounts.escalated.toLocaleString()})</span>
        </button>
      </div>

      {/* ── Filters & Search Control Bar ── */}
      <div className="filters-bar card mb-4">
        <div className="flex items-center gap-3 flex-wrap flex-1">
          {/* Search box */}
          <div className="search-input-wrap">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              placeholder="Search address or hash"
              className="search-input"
              value={searchQuery}
              onChange={(e) => { setSearchQuery(e.target.value); setPage(0); }}
            />
            {searchQuery && (
              <button className="search-clear-btn" onClick={() => setSearchQuery('')}>
                <X size={12} />
              </button>
            )}
          </div>

          <label className="filter-toggle">
            <input
              type="checkbox"
              checked={filterFlagged}
              onChange={(e) => { setFilterFlagged(e.target.checked); setPage(0); }}
            />
            <span>Flagged only</span>
          </label>

          <select
            className="filter-select"
            value={filterPattern}
            onChange={(e) => { setFilterPattern(e.target.value); setPage(0); }}
          >
            <option value="all">All patterns</option>
            {patternTypes.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted" style={{ marginRight: 6 }}>
            {filtered.length.toLocaleString()} of {results.length.toLocaleString()}
          </span>

          <button className="btn btn--secondary btn--sm" onClick={handleExport}>
            <Download size={13} /> Export CSV
          </button>

          <button
            className="btn btn--ghost btn--sm"
            onClick={handleExportAuditTrail}
            title="Download JSON compliance package with analyst determinations and SHAP drivers"
          >
            <FileCheck size={13} style={{ color: 'var(--accent-cyan)' }} />
            <span>Audit Trail JSON</span>
          </button>
        </div>
      </div>

      {/* ── Data Table ── */}
      <div className="card table-card" style={{ overflow: 'hidden' }}>
        <div className="data-table-wrapper" style={{ maxHeight: 560, overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 34 }}></th>
                <th onClick={() => handleSort('tx_hash')}>Tx Hash <SortArrow col="tx_hash" /></th>
                <th onClick={() => handleSort('from')}>From <SortArrow col="from" /></th>
                <th onClick={() => handleSort('to')}>To <SortArrow col="to" /></th>
                <th onClick={() => handleSort('amount')}>Amount <SortArrow col="amount" /></th>
                <th onClick={() => handleSort('suspicion_score')}>Risk Score <SortArrow col="suspicion_score" /></th>
                <th onClick={() => handleSort('aml_flag')}>Triage Status <SortArrow col="aml_flag" /></th>
                <th onClick={() => handleSort('pattern_type')}>Pattern <SortArrow col="pattern_type" /></th>
                <th>Top driver</th>
              </tr>
            </thead>
            <tbody>
              {paged.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
                    No transactions match the selected filter criteria.
                  </td>
                </tr>
              ) : (
                paged.map((r, i) => {
                  const globalIdx = page * PAGE_SIZE + i;
                  const isSelected = selectedIdx === globalIdx;
                  const rev = reviews[r.tx_hash];
                  return (
                    <React.Fragment key={globalIdx}>
                      <tr
                        className={`${r.aml_flag === 1 ? 'row--flagged' : ''} ${isSelected ? 'row--selected' : ''}`}
                        onClick={() => setSelectedIdx(isSelected ? null : globalIdx)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td style={{ textAlign: 'center' }}>
                          <button
                            className="btn btn--ghost btn--icon"
                            style={{ padding: 2, opacity: 0.7 }}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedIdx(isSelected ? null : globalIdx);
                            }}
                          >
                            <Eye size={14} className={isSelected ? 'text-primary' : ''} />
                          </button>
                        </td>

                        {/* Tx Hash */}
                        <td>
                          <div className="flex items-center gap-1.5 font-mono text-primary">
                            <span>{truncateAddr(r.tx_hash)}</span>
                            {r.tx_hash && (
                              <>
                                <button
                                  className="table-action-btn"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleCopy(r.tx_hash, `tx-${globalIdx}`);
                                  }}
                                  title="Copy Transaction Hash"
                                >
                                  {copiedKey === `tx-${globalIdx}` ? <Check size={11} className="text-emerald" /> : <Copy size={11} />}
                                </button>
                                <a
                                  href={`https://etherscan.io/tx/${r.tx_hash}`}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="table-action-btn"
                                  onClick={(e) => e.stopPropagation()}
                                  title="View on Etherscan"
                                >
                                  <ExternalLink size={11} />
                                </a>
                              </>
                            )}
                          </div>
                        </td>

                        {/* From Address */}
                        <td>
                          <div className="flex items-center gap-1.5 font-mono text-secondary">
                            <span>{truncateAddr(r.from)}</span>
                            {r.from && (
                              <button
                                className="table-action-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleCopy(r.from, `from-${globalIdx}`);
                                }}
                                title="Copy Sender Address"
                              >
                                {copiedKey === `from-${globalIdx}` ? <Check size={11} className="text-emerald" /> : <Copy size={11} />}
                              </button>
                            )}
                          </div>
                        </td>

                        {/* To Address */}
                        <td>
                          <div className="flex items-center gap-1.5 font-mono text-secondary">
                            <span>{truncateAddr(r.to)}</span>
                            {r.to && (
                              <button
                                className="table-action-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleCopy(r.to, `to-${globalIdx}`);
                                }}
                                title="Copy Recipient Address"
                              >
                                {copiedKey === `to-${globalIdx}` ? <Check size={11} className="text-emerald" /> : <Copy size={11} />}
                              </button>
                            )}
                          </div>
                        </td>

                        {/* Amount */}
                        <td className="font-mono">
                          {(r.amount ?? 0).toFixed(4)}
                        </td>

                        {/* Risk Score */}
                        <td>
                          <ScoreBar score={r.suspicion_score ?? 0} />
                        </td>

                        {/* Triage Status */}
                        <td>
                          {rev?.status === 'cleared' ? (
                            <span className="badge badge--safe" style={{ gap: 4 }}>
                              <Check size={10} /> Cleared (Analyst)
                            </span>
                          ) : rev?.status === 'escalated' ? (
                            <span className="badge badge--danger" style={{ gap: 4 }}>
                              <AlertTriangle size={10} /> Escalated (SAR)
                            </span>
                          ) : r.aml_flag === 1 && (r.suspicion_score ?? 0) < 0.45 ? (
                            <span className="badge badge--warn" style={{ gap: 4 }}>
                              Needs Review
                            </span>
                          ) : r.aml_flag === 1 ? (
                            <span className="badge badge--danger">Flagged</span>
                          ) : (
                            <span className="badge badge--safe">Clean</span>
                          )}
                        </td>

                        {/* Pattern */}
                        <td>
                          <PatternBadge pattern={r.pattern_type} />
                        </td>

                        {/* Top Driver */}
                        <td className="font-mono text-xs text-muted">
                          {r.top_shap_reason || 'Normal Baseline'}
                        </td>
                      </tr>

                      {/* Expandable SHAP & Analyst Triage Drawer Row */}
                      {isSelected && (
                        <tr className="row--expanded">
                          <td colSpan={9} style={{ padding: '20px 24px', background: 'var(--bg-primary)' }}>
                            <div className="flex items-center justify-between mb-3">
                              <span className="label">Forensic Evidence &amp; Human-in-the-Loop Determination</span>
                              <span className="text-xs font-mono text-muted">
                                {r.tx_hash}
                              </span>
                            </div>

                            {/* SHAP Feature Decomposition */}
                            <ShapExplainer transaction={r} />

                            {/* Analyst Triage Decision Panel */}
                            <AnalystReviewPanel
                              transaction={r}
                              currentReview={reviews[r.tx_hash]}
                              onSaveReview={onSaveReview}
                              onClearReview={onClearReview}
                            />
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* ── Pagination Footer ── */}
        {totalPages > 1 && (
          <div className="table-pagination">
            <button
              className="btn btn--ghost btn--xs"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
            >
              ← Prev
            </button>
            <div className="pagination-pages">
              {Array.from({ length: Math.min(totalPages, 9) }).map((_, pi) => {
                let targetPage = pi;
                if (totalPages > 9 && page > 4) {
                  targetPage = Math.min(page - 4 + pi, totalPages - 1);
                }
                return (
                  <button
                    key={targetPage}
                    className={`page-pill ${page === targetPage ? 'page-pill--active' : ''}`}
                    onClick={() => setPage(targetPage)}
                  >
                    {targetPage + 1}
                  </button>
                );
              })}
            </div>
            <button
              className="btn btn--ghost btn--xs"
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
            >
              Next →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
