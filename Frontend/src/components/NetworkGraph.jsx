import React, { useEffect, useRef, useMemo, useState } from 'react';
import { Network } from 'vis-network';
import {
  GitBranch,
  X,
  AlertTriangle,
  CheckCircle,
  Wallet,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  Eye,
  EyeOff,
  Copy,
  Check,
  ExternalLink,
  Layers,
  Sparkles,
} from 'lucide-react';
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
  const wrapperRef = useRef(null);
  const networkRef = useRef(null);

  // States
  const [selectedWallet, setSelectedWallet] = useState(null);
  const [showCleanNodes, setShowCleanNodes] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [copiedAddr, setCopiedAddr] = useState(false);

  // Build lookup: walletAddress -> top 3 flagged txns from results
  const walletTxLookup = useMemo(() => {
    const map = {};
    results.forEach((r) => {
      const from = r.from;
      const to = r.to;
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

    // Filter nodes based on showCleanNodes toggle
    const filteredRawNodes = showCleanNodes
      ? graphData.nodes
      : graphData.nodes.filter((n) => n.isSuspicious);

    const validNodeIds = new Set(filteredRawNodes.map((n) => n.id));

    const nodes = filteredRawNodes.map((n) => {
      const isSuspicious = n.isSuspicious;
      const score = n.maxScore ?? 0;

      let bgColor, borderColor, fontColor;
      if (isSuspicious) {
        const intensity = Math.min(score * 1.2, 1);
        bgColor = `rgba(244, 63, 94, ${0.35 + intensity * 0.5})`;
        borderColor = `rgba(244, 63, 94, ${0.7 + intensity * 0.3})`;
        fontColor = '#fecdd3';
      } else {
        bgColor = 'rgba(0, 242, 254, 0.15)';
        borderColor = 'rgba(0, 242, 254, 0.55)';
        fontColor = '#67e8f9';
      }

      const degree = (n.inDegree ?? 0) + (n.outDegree ?? 0);
      const size = Math.max(12, Math.min(36, 12 + degree * 3 + Math.log1p(n.totalVolume ?? 0) * 2));

      return {
        id: n.id,
        label: n.label || n.id,
        title: `${isSuspicious ? '⚠ Suspicious' : '✓ Clean'} Wallet\n${n.address || n.id}\nIn: ${n.inDegree} · Out: ${n.outDegree}\nVolume: ${(n.totalVolume ?? 0).toFixed(4)} ETH\nMax Score: ${score.toFixed(4)}`,
        size,
        color: {
          background: bgColor,
          border: borderColor,
          highlight: {
            background: isSuspicious ? 'rgba(244, 63, 94, 0.9)' : 'rgba(0, 242, 254, 0.6)',
            border: isSuspicious ? '#f43f5e' : '#00f2fe',
          },
          hover: {
            background: isSuspicious ? 'rgba(244, 63, 94, 0.7)' : 'rgba(0, 242, 254, 0.4)',
            border: isSuspicious ? '#f43f5e' : '#00f2fe',
          },
        },
        font: { color: fontColor, size: 9, face: "'JetBrains Mono', monospace" },
        borderWidth: isSuspicious ? 2 : 1.5,
        shadow: false,
        _address: n.address || n.id,
        _isSuspicious: isSuspicious,
        _score: score,
        _inDegree: n.inDegree ?? 0,
        _outDegree: n.outDegree ?? 0,
        _volume: n.totalVolume ?? 0,
      };
    });

    const edges = graphData.edges
      .filter((e) => validNodeIds.has(e.source) && validNodeIds.has(e.target))
      .map((e, i) => {
        const isFlagged = e.is_flagged;
        return {
          id: e.id || `edge-${i}`,
          from: e.source,
          to: e.target,
          title: `Amount: ${(e.amount ?? 0).toFixed(4)} ETH\nScore: ${(e.suspicion_score ?? 0).toFixed(4)}\nPattern: ${e.pattern_type || 'Normal'}`,
          color: {
            color: isFlagged ? 'rgba(244, 63, 94, 0.55)' : 'rgba(0, 242, 254, 0.18)',
            highlight: isFlagged ? '#f43f5e' : '#00f2fe',
            hover: isFlagged ? 'rgba(244, 63, 94, 0.80)' : 'rgba(0, 242, 254, 0.45)',
          },
          width: isFlagged ? 2 : 1,
          arrows: { to: { enabled: true, scaleFactor: 0.5 } },
          smooth: { type: 'curvedCW', roundness: 0.15 },
          dashes: !isFlagged,
        };
      });

    return { visNodes: nodes, visEdges: edges, stats: graphData.stats || {} };
  }, [graphData, showCleanNodes]);

  // Vis-network instance lifecycle
  useEffect(() => {
    if (!containerRef.current || visNodes.length === 0) return;

    const options = {
      nodes: { shape: 'dot', scaling: { min: 10, max: 36 } },
      edges: { smooth: { type: 'curvedCW', roundness: 0.15 } },
      physics: {
        // Layout is solved once during stabilization, then frozen (see below).
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -60,
          centralGravity: 0.02,
          springLength: 70,
          springConstant: 0.06,
          damping: 0.6,
        },
        stabilization: { enabled: true, iterations: 400, updateInterval: 100, fit: true },
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        zoomView: true,
        dragView: true,
        navigationButtons: false,
        keyboard: false,
      },
      layout: { improvedLayout: true },
    };

    const network = new Network(
      containerRef.current,
      { nodes: visNodes, edges: visEdges },
      options
    );
    networkRef.current = network;

    // Freeze nodes once the layout has settled so nothing drifts or reacts to dragging.
    network.once('stabilizationIterationsDone', () => {
      network.setOptions({ physics: { enabled: false } });
      network.fit({ animation: false });
    });

    // Node click handler -> open wallet inspection drawer
    network.on('click', (params) => {
      if (params.nodes && params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const clickedNode = visNodes.find((n) => n.id === nodeId);
        if (clickedNode) {
          const addr = clickedNode._address;
          const topTxns = walletTxLookup[addr] || [];
          setSelectedWallet({
            address: addr,
            isSuspicious: clickedNode._isSuspicious,
            score: clickedNode._score,
            inDegree: clickedNode._inDegree,
            outDegree: clickedNode._outDegree,
            volume: clickedNode._volume,
            topTxns,
          });
        }
      } else if (params.edges && params.edges.length > 0) {
        // Edge clicked
      } else {
        // Empty space clicked
      }
    });

    return () => {
      network.destroy();
      networkRef.current = null;
    };
  }, [visNodes, visEdges, walletTxLookup]);

  // Zoom controls
  const handleZoomIn = () => {
    if (!networkRef.current) return;
    const currentScale = networkRef.current.getScale();
    networkRef.current.moveTo({ scale: currentScale * 1.35, animation: { duration: 300 } });
  };

  const handleZoomOut = () => {
    if (!networkRef.current) return;
    const currentScale = networkRef.current.getScale();
    networkRef.current.moveTo({ scale: currentScale * 0.75, animation: { duration: 300 } });
  };

  const handleFitView = () => {
    if (!networkRef.current) return;
    networkRef.current.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
  };

  // Fullscreen handler
  const handleToggleFullscreen = () => {
    if (!wrapperRef.current) return;
    if (!document.fullscreenElement) {
      wrapperRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  const handleCopyAddr = (address) => {
    navigator.clipboard.writeText(address);
    setCopiedAddr(true);
    setTimeout(() => setCopiedAddr(false), 2000);
  };

  return (
    <div 
      ref={wrapperRef} 
      className={`network-graph-wrapper ${isFullscreen ? 'network-graph--fullscreen' : ''}`}
    >
      {/* ── Top Legend & HUD Status Bar ── */}
      <div className="network-hud-bar">
        <div className="network-legend">
          <span className="network-legend__item">
            <span className="network-legend__dot network-legend__dot--suspicious" />
            <span>Suspicious Wallet</span>
          </span>
          <span className="network-legend__item">
            <span className="network-legend__dot network-legend__dot--clean" />
            <span>Clean Wallet</span>
          </span>
          <span className="network-legend__item">
            <span className="network-legend__line network-legend__line--flagged" />
            <span>Flagged Edge</span>
          </span>
          <span className="network-legend__item">
            <span className="network-legend__line network-legend__line--clean" />
            <span>Clean Edge</span>
          </span>
        </div>

        <div className="network-stats-pill">
          <span>{stats.node_count ?? visNodes.length} nodes</span>
          <span className="network-stats-pill__sep">·</span>
          <span>{stats.edge_count ?? visEdges.length} edges</span>
          <span className="network-stats-pill__sep">·</span>
          <span className="text-rose font-semibold">{stats.suspicious_nodes ?? visNodes.filter(n => n._isSuspicious).length} suspicious</span>
          <span className="network-stats-pill__sep">·</span>
          <span className="text-cyan font-semibold">{stats.clean_nodes ?? visNodes.filter(n => !n._isSuspicious).length} clean</span>
        </div>
      </div>

      {/* ── Canvas Container with Floating HUD Controls ── */}
      <div className="network-canvas-container" style={{ position: 'relative' }}>
        <div 
          ref={containerRef} 
          style={{ width: '100%', height: isFullscreen ? 'calc(100vh - 60px)' : '720px' }} 
        />

        {/* Floating HUD Controls */}
        <div className="network-floating-hud">
          <button 
            className="hud-btn" 
            onClick={handleZoomIn} 
            title="Zoom In (+)"
          >
            <ZoomIn size={16} />
          </button>
          <button 
            className="hud-btn" 
            onClick={handleZoomOut} 
            title="Zoom Out (-)"
          >
            <ZoomOut size={16} />
          </button>
          <button 
            className="hud-btn" 
            onClick={handleFitView} 
            title="Fit to Center"
          >
            <Maximize2 size={16} />
          </button>
          <div className="hud-btn-sep" />
          <button 
            className={`hud-btn ${!showCleanNodes ? 'hud-btn--active' : ''}`} 
            onClick={() => setShowCleanNodes(!showCleanNodes)} 
            title={showCleanNodes ? "Hide clean wallets" : "Show all wallets"}
          >
            {showCleanNodes ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <div className="hud-btn-sep" />
          <button 
            className="hud-btn" 
            onClick={handleToggleFullscreen} 
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Graph View"}
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>

        {/* Hint overlay */}
        <div className="network-hint-overlay">
          <span>Select a wallet to inspect</span>
        </div>

        {/* ── Slide-in Forensic Wallet Drawer ── */}
        {selectedWallet && (
          <div className="network-wallet-drawer">
            <div className="network-wallet-drawer__header">
              <div className="flex items-center gap-2">
                <Wallet size={16} className="text-cyan" />
                <span className="font-semibold text-sm">Wallet</span>
              </div>
              <button 
                className="btn btn--ghost btn--icon" 
                onClick={() => setSelectedWallet(null)}
              >
                <X size={16} />
              </button>
            </div>

            <div className="network-wallet-drawer__body">
              {/* Address with copy & Etherscan */}
              <div className="wallet-addr-box">
                <div className="text-mono text-xs text-primary" style={{ wordBreak: 'break-all' }}>
                  {selectedWallet.address}
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <button 
                    className="btn btn--secondary btn--xs"
                    onClick={() => handleCopyAddr(selectedWallet.address)}
                  >
                    {copiedAddr ? <Check size={12} className="text-emerald" /> : <Copy size={12} />}
                    {copiedAddr ? 'Copied' : 'Copy'}
                  </button>
                  <a 
                    href={`https://etherscan.io/address/${selectedWallet.address}`}
                    target="_blank"
                    rel="noreferrer"
                    className="btn btn--secondary btn--xs"
                  >
                    <ExternalLink size={12} />
                    Etherscan
                  </a>
                </div>
              </div>

              {/* Status and Risk */}
              <div className="flex items-center justify-between mt-3 mb-3">
                <span className={`badge ${selectedWallet.isSuspicious ? 'badge--danger' : 'badge--safe'}`}>
                  {selectedWallet.isSuspicious ? (
                    <><AlertTriangle size={12} style={{ marginRight: 4 }} /> Suspicious Nexus</>
                  ) : (
                    <><CheckCircle size={12} style={{ marginRight: 4 }} /> Clean Wallet</>
                  )}
                </span>
                <div>
                  <span className="text-xs text-muted mr-1">Max Risk:</span>
                  <ScorePill score={selectedWallet.score} />
                </div>
              </div>

              {/* Graph Centrality Metrics */}
              <div className="wallet-metrics-grid">
                <div className="wallet-metric">
                  <span className="wallet-metric__label">In-degree</span>
                  <span className="wallet-metric__val font-mono">{selectedWallet.inDegree}</span>
                </div>
                <div className="wallet-metric">
                  <span className="wallet-metric__label">Out-degree</span>
                  <span className="wallet-metric__val font-mono">{selectedWallet.outDegree}</span>
                </div>
                <div className="wallet-metric">
                  <span className="wallet-metric__label">Cumulative Vol</span>
                  <span className="wallet-metric__val font-mono">{selectedWallet.volume.toFixed(4)} ETH</span>
                </div>
              </div>

              {/* Flagged Transactions & SHAP Explanations */}
              <div className="mt-4">
                <div className="text-xs font-semibold text-muted text-uppercase mb-2">
                  Flagged transactions ({selectedWallet.topTxns.length})
                </div>

                {selectedWallet.topTxns.length === 0 ? (
                  <div className="text-xs text-muted py-3">
                    No flagged transactions for this wallet.
                  </div>
                ) : (
                  <div className="flex flex-col gap-3">
                    {selectedWallet.topTxns.map((tx, idx) => (
                      <div key={idx} className="wallet-tx-item">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="text-mono text-cyan">{truncateAddr(tx.tx_hash)}</span>
                          <ScorePill score={tx.suspicion_score ?? 0} />
                        </div>
                        <div className="text-xs text-muted mb-2">
                          {truncateAddr(tx.from)} → {truncateAddr(tx.to)} · {(tx.amount ?? 0).toFixed(4)} ETH · {tx.pattern_type}
                        </div>
                        <ShapExplainer transaction={tx} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
