import React, { useState, useMemo } from 'react';
import { Table2, Filter, Download, ChevronDown, ChevronUp, Eye } from 'lucide-react';
import ShapExplainer from './ShapExplainer';

const PAGE_SIZE = 20;

function truncateAddr(addr) {
  if (!addr || addr.length <= 14) return addr || '—';
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`;
}

function ScoreBar({ score }) {
  const pct = Math.round(score * 100);
  const cls = score >= 0.7 ? 'high' : score >= 0.4 ? 'mid' : 'low';
  return (
    <div className="score-bar">
      <div className="score-bar__track">
        <div className={`score-bar__fill score-bar__fill--${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-mono" style={{ fontSize: '0.78rem', minWidth: 40, textAlign: 'right' }}>
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

export default function AlertsTable({ results }) {
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState('suspicion_score');
  const [sortAsc, setSortAsc] = useState(false);
  const [filterFlagged, setFilterFlagged] = useState(false);
  const [filterPattern, setFilterPattern] = useState('all');
  const [selectedIdx, setSelectedIdx] = useState(null);

  // Unique pattern types
  const patternTypes = useMemo(() => {
    const s = new Set(results.map((r) => r.pattern_type).filter(Boolean));
    return Array.from(s);
  }, [results]);

  // Filter & sort
  const filtered = useMemo(() => {
    let data = [...results];
    if (filterFlagged) data = data.filter((r) => r.aml_flag === 1);
    if (filterPattern !== 'all') data = data.filter((r) => r.pattern_type === filterPattern);

    data.sort((a, b) => {
      const va = a[sortKey] ?? 0;
      const vb = b[sortKey] ?? 0;
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? va - vb : vb - va;
    });
    return data;
  }, [results, filterFlagged, filterPattern, sortKey, sortAsc]);

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
    return <span className="sort-arrow">{sortAsc ? '↑' : '↓'}</span>;
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
    a.download = 'defiguard_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const selectedTransaction = selectedIdx !== null ? filtered[selectedIdx] : null;

  return (
    <div>
      {/* Filters Bar */}
      <div className="filters-bar">
        <Filter size={16} style={{ color: 'var(--text-muted)' }} />

        <label className="filter-toggle">
          <input
            type="checkbox"
            checked={filterFlagged}
            onChange={(e) => { setFilterFlagged(e.target.checked); setPage(0); }}
          />
          Flagged only
        </label>

        <select
          className="filter-select"
          value={filterPattern}
          onChange={(e) => { setFilterPattern(e.target.value); setPage(0); }}
        >
          <option value="all">All Patterns</option>
          {patternTypes.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>

        <span className="text-sm text-muted" style={{ marginLeft: 'auto' }}>
          Showing {filtered.length} of {results.length} transactions
        </span>

        <button className="btn btn--ghost btn--sm" onClick={handleExport}>
          <Download size={14} /> Export CSV
        </button>
      </div>

      {/* Table */}
      <div className="card" style={{ overflow: 'hidden' }}>
        <div className="data-table-wrapper" style={{ maxHeight: 520, overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 36 }}></th>
                <th onClick={() => handleSort('tx_hash')}>Tx Hash <SortArrow col="tx_hash" /></th>
                <th onClick={() => handleSort('from')}>From <SortArrow col="from" /></th>
                <th onClick={() => handleSort('to')}>To <SortArrow col="to" /></th>
                <th onClick={() => handleSort('amount')}>Amount <SortArrow col="amount" /></th>
                <th onClick={() => handleSort('suspicion_score')}>Risk Score <SortArrow col="suspicion_score" /></th>
                <th onClick={() => handleSort('aml_flag')}>Flag <SortArrow col="aml_flag" /></th>
                <th onClick={() => handleSort('pattern_type')}>Pattern <SortArrow col="pattern_type" /></th>
                <th>Top Driver</th>
              </tr>
            </thead>
            <tbody>
              {paged.map((r, i) => {
                const globalIdx = page * PAGE_SIZE + i;
                const isSelected = selectedIdx === globalIdx;
                return (
                  <React.Fragment key={globalIdx}>
                    <tr
                      className={`${r.aml_flag === 1 ? 'row--flagged' : ''} ${isSelected ? 'row--selected' : ''}`}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setSelectedIdx(isSelected ? null : globalIdx)}
                    >
                      <td>
                        <Eye size={14} style={{ color: isSelected ? 'var(--accent-cyan)' : 'var(--text-muted)', opacity: isSelected ? 1 : 0.4 }} />
                      </td>
                      <td className="cell-address" title={r.tx_hash}>{truncateAddr(r.tx_hash)}</td>
                      <td className="cell-address" title={r.from}>{truncateAddr(r.from)}</td>
                      <td className="cell-address" title={r.to}>{truncateAddr(r.to)}</td>
                      <td className="text-mono">{Number(r.amount ?? 0).toFixed(4)}</td>
                      <td><ScoreBar score={Number(r.suspicion_score ?? 0)} /></td>
                      <td>
                        {r.aml_flag === 1
                          ? <span className="badge badge--danger">⚠ Flagged</span>
                          : <span className="badge badge--safe">Clean</span>
                        }
                      </td>
                      <td><PatternBadge pattern={r.pattern_type} /></td>
                      <td className="text-mono" style={{ fontSize: '0.76rem', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {r.top_shap_reason}
                      </td>
                    </tr>
                    {/* Expanded SHAP Detail */}
                    {isSelected && (
                      <tr>
                        <td colSpan={9} style={{ padding: 0, border: 'none' }}>
                          <div style={{ padding: '12px 16px' }}>
                            <ShapExplainer transaction={r} />
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <button
            className="pagination__btn"
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
          >
            ← Prev
          </button>
          {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
            let pageIdx;
            if (totalPages <= 7) {
              pageIdx = i;
            } else if (page < 4) {
              pageIdx = i;
            } else if (page >= totalPages - 4) {
              pageIdx = totalPages - 7 + i;
            } else {
              pageIdx = page - 3 + i;
            }
            return (
              <button
                key={pageIdx}
                className={`pagination__btn ${page === pageIdx ? 'pagination__btn--active' : ''}`}
                onClick={() => setPage(pageIdx)}
              >
                {pageIdx + 1}
              </button>
            );
          })}
          <button
            className="pagination__btn"
            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
            disabled={page >= totalPages - 1}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
