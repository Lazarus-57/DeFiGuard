import React, { useEffect, useRef, useMemo, useState } from 'react';
import { Network } from 'vis-network';
import { GitBranch, X, AlertTriangle, CheckCircle, Wallet } from 'lucide-react';
import ShapExplainer from './ShapExplainer';

function truncateAddr(addr) {
  if (!addr || addr.length <= 14) return addr || '—';
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`;
}

function ScorePill({ score }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.7 ? 'var(--accent-rose)' : score >= 0.4 ? 'var(--accent-amber)' : 'var(--accent-cyan)';
  return (
    <span
      className="text-mono"
      style={{ color, fontWeight: 700, fontSize: '0.85rem' }}
    >
      {pct}%
    </span>
  );
}

export default function NetworkGraph({ graphData, results = [] }) {
  const containerRef = useRef(null);
  const networkRef   = useRef(null);

  // Selected wallet state for sidebar
  const [selectedWallet, setSelectedWallet] = useState(null);

  // Build lookup: walletAddress -> top 3 flagged txns from results
  const walletTxLookup = useMemo(() => {
    const map = {};
    results.forEach((r) => {
      const from = r.from;
      const to   = r.to;
      [from, to].forEach((addr) => {
        if (!addr) return;
        if (!map[addr]) map[addr] = [];
        if (r.aml_flag === 1) map[addr].push(r);
      });
    });
    // Sort each wallet's txns by score desc, keep top 3
    Object.keys(map).forEach((addr) => {
      map[addr] = map[addr]
        .sort((a, b) => (b.suspicion_score ?? 0) - (a.suspicion_score ?? 0))
        .slice(0, 3);
    });
    return map;
  }, [results]);

  const { visNodes, visEdges, stats } = useMemo(() => {
    if (!graphData || !graphData.nodes || !graphData.edges) {
      return { visNodes: [], visEdges: [], stats: {} };
    }

    const nodes = graphData.nodes.map((n) => {
      const isSuspicious = n.isSuspicious;
      const score = n.maxScore ?? 0;

      let bgColor, borderColor, fontColor;
      if (isSuspicious) {
        const intensity = Math.min(score * 1.2, 1);
        bgColor     = `rgba(244, 63, 94, ${0.3 + intensity * 0.5})`;
        borderColor = `rgba(244, 63, 94, ${0.6 + intensity * 0.4})`;
        fontColor   = '#fecdd3';
      } else {
        bgColor     = 'rgba(0, 242, 254, 0.15)';
        borderColor = 'rgba(0, 242, 254, 0.50)';
        fontColor   = '#67e8f9';
      }

      const degree = (n.inDegree ?? 0) + (n.outDegree ?? 0);
      const size   = Math.max(12, Math.min(35, 12 + degree * 3 + Math.log1p(n.totalVolume ?? 0) * 2));

      return {
        id: n.id,
        label: n.label || n.id,
        title: `${isSuspicious ? '⚠ Suspicious' : '✓ Clean'} Wallet\n${n.address || n.id}\nIn: ${n.inDegree} · Out: ${n.outDegree}\nVolume: ${(n.totalVolume ?? 0).toFixed(4)} ETH\nMax Score: ${score.toFixed(4)}`,
        size,
        color: {
          background: bgColor,
          border:     borderColor,
          highlight: {
            background: isSuspicious ? 'rgba(244, 63, 94, 0.8)' : 'rgba(0, 242, 254, 0.5)',
            border:     isSuspicious ? '#f43f5e'                 : '#00f2fe',
          },
          hover: {
            background: isSuspicious ? 'rgba(244, 63, 94, 0.6)' : 'rgba(0, 242, 254, 0.35)',
            border:     isSuspicious ? '#f43f5e'                 : '#00f2fe',
          },
        },
        font:        { color: fontColor, size: 9, face: "'JetBrains Mono', monospace" },
        borderWidth: isSuspicious ? 2 : 1.5,
        shadow:      isSuspicious ? { enabled: true, color: 'rgba(244, 63, 94, 0.35)', size: 8 } : false,
        // Store original data for sidebar lookup
        _address: n.address || n.id,
        _isSuspicious: isSuspicious,
        _score: score,
        _inDegree:  n.inDegree  ?? 0,
        _outDegree: n.outDegree ?? 0,
        _volume:    n.totalVolume ?? 0,
      };
    });

    const edges = graphData.edges.map((e, i) => {
      const isFlagged = e.is_flagged;
      return {
        id:    e.id || `edge-${i}`,
        from:  e.source,
        to:    e.target,
        title: `Amount: ${(e.amount ?? 0).toFixed(4)} ETH\nScore: ${(e.suspicion_score ?? 0).toFixed(4)}\nPattern: ${e.pattern_type || 'Normal'}`,
        color: {
          color:     isFlagged ? 'rgba(244, 63, 94, 0.50)' : 'rgba(0, 242, 254, 0.15)',
          highlight: isFlagged ? '#f43f5e'                  : '#00f2fe',
          hover:     isFlagged ? 'rgba(244, 63, 94, 0.70)'  : 'rgba(0, 242, 254, 0.40)',
        },
        width:  isFlagged ? 2 : 1,
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
        smooth: { type: 'curvedCW', roundness: 0.15 },
        dashes: !isFlagged,
      };
    });

    return { visNodes: nodes, visEdges: edges, stats: graphData.stats || {} };
  }, [graphData]);

  useEffect(() => {
    if (!containerRef.current || visNodes.length === 0) return;

    const options = {
      nodes: { shape: 'dot', scaling: { min: 10, max: 35 } },
      edges: { smooth: { type: 'curvedCW', roundness: 0.15 } },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -80,
          centralGravity:        0.01,
          springLength:          80,
          springConstant:        0.05,
          damping:               0.45,
        },
        stabilization: { enabled: true, iterations: 300, updateInterval: 20 },
      },
      interaction: {
        hover:             true,
        tooltipDelay:      100,
        zoomView:          true,
        dragView:          true,
        navigationButtons: false,
        keyboard:          false,
      },
      layout: { improvedLayout: true },
    };

    const network = new Network(
      containerRef.current,
      { nodes: visNodes, edges: visEdges },
      options
    );
    networkRef.current = network;

    network.once('stabilizationIterationsDone', () => {
      network.setOptions({ physics: { enabled: false } });
    });

    // ── Click handler: open sidebar for selected node ──
    network.on('click', (params) => {
      if (params.nodes && params.nodes.length > 0) {
        const nodeId   = params.nodes[0];
        const nodeData = visNodes.find((n) => n.id === nodeId);
        if (nodeData) {
          setSelectedWallet({
            address:     nodeData._address,
            isSuspicious: nodeData._isSuspicious,
            score:       nodeData._score,
            inDegree:    nodeData._inDegree,
            outDegree:   nodeData._outDegree,
            volume:      nodeData._volume,
          });
        }
      } else {
        // Click on empty canvas — deselect
        setSelectedWallet(null);
      }
    });

    return () => {
      network.destroy();
      networkRef.current = null;
    };
  }, [visNodes, visEdges]);

  if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
    return (
      <div className="empty-state">
        <GitBranch className="empty-state__icon" />
        <div className="empty-state__title">No Graph Data</div>
        <div className="empty-state__text">
          Upload transactions to generate the wallet network graph visualization.
        </div>
      </div>
    );
  }

  // Txns associated with the selected wallet
  const walletTxns = selectedWallet ? (walletTxLookup[selectedWallet.address] || []) : [];

  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>

      {/* ── Left: graph + legend ── */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Legend */}
        <div className="graph-legend mb-4">
          <div className="graph-legend__item">
            <div className="graph-legend__dot" style={{ background: 'rgba(244, 63, 94, 0.7)' }} />
            Suspicious Wallet
          </div>
          <div className="graph-legend__item">
            <div className="graph-legend__dot" style={{ background: 'rgba(0, 242, 254, 0.5)' }} />
            Clean Wallet
          </div>
          <div className="graph-legend__item">
            <div className="graph-legend__dot" style={{ background: 'rgba(244, 63, 94, 0.5)', width: 20, height: 3, borderRadius: 2 }} />
            Flagged Edge
          </div>
          <div className="graph-legend__item">
            <div className="graph-legend__dot" style={{ background: 'rgba(0, 242, 254, 0.2)', width: 20, height: 3, borderRadius: 2 }} />
            Clean Edge
          </div>
          <span className="text-sm text-muted" style={{ marginLeft: 'auto' }}>
            {stats.total_nodes ?? 0} nodes · {stats.total_edges ?? 0} edges · {stats.suspicious_nodes ?? 0} suspicious
            {' · '}
            <span style={{ color: 'rgba(0,242,254,0.7)' }}>
              {(stats.total_nodes ?? 0) - (stats.suspicious_nodes ?? 0)} clean
            </span>
          </span>
        </div>

        {/* Click-to-inspect hint */}
        <p className="text-sm text-muted mb-4" style={{ opacity: 0.6 }}>
          Click any node to inspect why it was flagged →
        </p>

        {/* Network Canvas */}
        <div ref={containerRef} className="graph-container" />
      </div>

      {/* ── Right: wallet SHAP sidebar ── */}
      {selectedWallet && (
        <div
          style={{
            width: 360,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          }}
        >
          {/* Header card */}
          <div className="card">
            <div className="card__header" style={{ justifyContent: 'space-between' }}>
              <span className="card__title">
                <Wallet size={16} /> Wallet Details
              </span>
              <button
                onClick={() => setSelectedWallet(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
                title="Close"
              >
                <X size={16} />
              </button>
            </div>
            <div className="card__body" style={{ fontSize: '0.82rem', lineHeight: 1.8 }}>
              <div style={{ fontFamily: 'monospace', wordBreak: 'break-all', color: 'var(--text-secondary)', marginBottom: 8 }}>
                {selectedWallet.address}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                {selectedWallet.isSuspicious ? (
                  <><AlertTriangle size={14} style={{ color: 'var(--accent-rose)' }} />
                  <span className="badge badge--danger">Suspicious</span></>
                ) : (
                  <><CheckCircle size={14} style={{ color: 'var(--accent-cyan)' }} />
                  <span className="badge badge--safe">Clean</span></>
                )}
                {selectedWallet.isSuspicious && (
                  <span style={{ marginLeft: 'auto' }}>
                    Max risk: <ScorePill score={selectedWallet.score} />
                  </span>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <div style={{ color: 'var(--text-muted)' }}>In-degree</div>
                <div className="text-mono">{selectedWallet.inDegree}</div>
                <div style={{ color: 'var(--text-muted)' }}>Out-degree</div>
                <div className="text-mono">{selectedWallet.outDegree}</div>
                <div style={{ color: 'var(--text-muted)' }}>Volume</div>
                <div className="text-mono">{selectedWallet.volume.toFixed(4)} ETH</div>
              </div>
            </div>
          </div>

          {/* Top 3 flagged transactions with SHAP */}
          {selectedWallet.isSuspicious && walletTxns.length > 0 ? (
            <>
              <div className="text-sm" style={{ color: 'var(--text-muted)', fontWeight: 600, marginTop: 4 }}>
                Top {walletTxns.length} Flagged Transaction{walletTxns.length > 1 ? 's' : ''}
              </div>
              {walletTxns.map((tx, i) => (
                <div key={i} style={{ borderLeft: '3px solid var(--accent-rose)', paddingLeft: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', marginBottom: 6 }}>
                    <span className="text-mono" style={{ color: 'var(--text-muted)' }}>
                      {truncateAddr(tx.tx_hash)}
                    </span>
                    <ScorePill score={tx.suspicion_score ?? 0} />
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginBottom: 8 }}>
                    {truncateAddr(tx.from)} → {truncateAddr(tx.to)} · {Number(tx.amount ?? 0).toFixed(4)} ETH · {tx.pattern_type}
                  </div>
                  <ShapExplainer transaction={tx} />
                </div>
              ))}
            </>
          ) : selectedWallet.isSuspicious ? (
            <div className="card" style={{ padding: 16 }}>
              <p className="text-sm text-muted">
                No flagged transactions found in this batch for this wallet.
                It may have been marked suspicious via a connected wallet's activity.
              </p>
            </div>
          ) : (
            <div className="card" style={{ padding: 16 }}>
              <p className="text-sm" style={{ color: 'var(--accent-cyan)', opacity: 0.8 }}>
                ✓ This wallet has no flagged transactions in the current dataset sample.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
