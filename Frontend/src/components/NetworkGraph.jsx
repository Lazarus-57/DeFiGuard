import React, { useEffect, useRef, useMemo } from 'react';
import { Network } from 'vis-network';
import { GitBranch } from 'lucide-react';

export default function NetworkGraph({ graphData }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  const { visNodes, visEdges, stats } = useMemo(() => {
    if (!graphData || !graphData.nodes || !graphData.edges) {
      return { visNodes: [], visEdges: [], stats: {} };
    }

    const nodes = graphData.nodes.map((n) => {
      const isSuspicious = n.isSuspicious;
      const score = n.maxScore ?? 0;

      // Node color based on suspicion
      let bgColor, borderColor, fontColor;
      if (isSuspicious) {
        const intensity = Math.min(score * 1.2, 1);
        bgColor = `rgba(244, 63, 94, ${0.3 + intensity * 0.5})`;
        borderColor = `rgba(244, 63, 94, ${0.6 + intensity * 0.4})`;
        fontColor = '#fecdd3';
      } else {
        bgColor = 'rgba(0, 242, 254, 0.12)';
        borderColor = 'rgba(0, 242, 254, 0.35)';
        fontColor = '#8b97b0';
      }

      // Size based on degree + volume
      const degree = (n.inDegree ?? 0) + (n.outDegree ?? 0);
      const size = Math.max(12, Math.min(35, 12 + degree * 3 + Math.log1p(n.totalVolume ?? 0) * 2));

      return {
        id: n.id,
        label: n.label || n.id,
        title: `Wallet: ${n.address || n.id}\nIn-degree: ${n.inDegree}\nOut-degree: ${n.outDegree}\nVolume: ${(n.totalVolume ?? 0).toFixed(4)} ETH\nMax Score: ${score.toFixed(4)}\nSuspicious: ${isSuspicious ? 'YES' : 'No'}`,
        size,
        color: {
          background: bgColor,
          border: borderColor,
          highlight: {
            background: isSuspicious ? 'rgba(244, 63, 94, 0.7)' : 'rgba(0, 242, 254, 0.4)',
            border: isSuspicious ? '#f43f5e' : '#00f2fe',
          },
        },
        font: { color: fontColor, size: 9, face: "'JetBrains Mono', monospace" },
        borderWidth: isSuspicious ? 2 : 1,
        shadow: isSuspicious ? { enabled: true, color: 'rgba(244, 63, 94, 0.3)', size: 8 } : false,
      };
    });

    const edges = graphData.edges.map((e, i) => {
      const isFlagged = e.is_flagged;
      return {
        id: e.id || `edge-${i}`,
        from: e.source,
        to: e.target,
        title: `Amount: ${(e.amount ?? 0).toFixed(4)} ETH\nScore: ${(e.suspicion_score ?? 0).toFixed(4)}\nPattern: ${e.pattern_type || 'Normal'}`,
        color: {
          color: isFlagged ? 'rgba(244, 63, 94, 0.50)' : 'rgba(100, 120, 160, 0.18)',
          highlight: isFlagged ? '#f43f5e' : '#00f2fe',
          hover: isFlagged ? 'rgba(244, 63, 94, 0.70)' : 'rgba(0, 242, 254, 0.40)',
        },
        width: isFlagged ? 2 : 1,
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
      nodes: {
        shape: 'dot',
        scaling: { min: 10, max: 35 },
      },
      edges: {
        smooth: { type: 'curvedCW', roundness: 0.15 },
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -60,
          centralGravity: 0.008,
          springLength: 120,
          springConstant: 0.04,
          damping: 0.4,
        },
        stabilization: {
          enabled: true,
          iterations: 200,
          updateInterval: 25,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        zoomView: true,
        dragView: true,
        navigationButtons: false,
        keyboard: false,
      },
      layout: {
        improvedLayout: true,
      },
    };

    const network = new Network(
      containerRef.current,
      { nodes: visNodes, edges: visEdges },
      options
    );

    networkRef.current = network;

    // Stabilize then stop physics to improve performance
    network.once('stabilizationIterationsDone', () => {
      network.setOptions({ physics: { enabled: false } });
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

  return (
    <div>
      {/* Legend */}
      <div className="graph-legend mb-4">
        <div className="graph-legend__item">
          <div className="graph-legend__dot" style={{ background: 'rgba(244, 63, 94, 0.7)' }} />
          Suspicious Wallet
        </div>
        <div className="graph-legend__item">
          <div className="graph-legend__dot" style={{ background: 'rgba(0, 242, 254, 0.4)' }} />
          Clean Wallet
        </div>
        <div className="graph-legend__item">
          <div className="graph-legend__dot" style={{ background: 'rgba(244, 63, 94, 0.5)', width: 20, height: 3, borderRadius: 2 }} />
          Flagged Edge
        </div>
        <div className="graph-legend__item">
          <div className="graph-legend__dot" style={{ background: 'rgba(100, 120, 160, 0.25)', width: 20, height: 3, borderRadius: 2 }} />
          Clean Edge
        </div>

        <span className="text-sm text-muted" style={{ marginLeft: 'auto' }}>
          {stats.total_nodes ?? 0} nodes · {stats.total_edges ?? 0} edges · {stats.suspicious_nodes ?? 0} suspicious
        </span>
      </div>

      {/* Network Canvas */}
      <div ref={containerRef} className="graph-container" />
    </div>
  );
}
